#!/usr/bin/env python3
"""
Batch test all tasks from dataloader configuration
Runs test_simulator.py for each task and collects success rates
"""

import os
import sys
import argparse
import yaml
import subprocess
import signal
import tempfile
import time
import re
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime


def load_config(config_path):
    """Load the dataloader configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def stop_process_group(process):
    """Stop a task subprocess and any xvfb/CoppeliaSim children it spawned."""
    # The group leader may have exited while descendants are still alive.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    if process.poll() is None:
        process.wait()


def run_task_test(task_name, args):
    """
    Run test for a single task. If an episode crashes after it has started,
    count that episode as a failure and resume from the next stored demo.
    Errors before the first episode starts are treated as setup errors.

    Returns: (success_rate, total_tests)
    """
    script_dir = Path(__file__).parent
    test_script = script_dir / 'test_simulator.py'

    base_cmd = [
        sys.executable, '-u', str(test_script),
        '--task', task_name,
        '--actor', args.actor,
        '--device', str(args.device),
    ]

    if args.disp:
        base_cmd.append('--disp')

    if args.pose_method:
        base_cmd.extend(['--pose_method', args.pose_method])

    if args.debug:
        base_cmd.append('--debug')

    if args.actor == 'expert':
        base_cmd.extend(['--traj_length', str(args.traj_length)])

    if args.add_noise != 'none' and args.pose_method in ['fp', 'foundation_pose']:
        base_cmd.extend(['--add_noise', args.add_noise])

    results = []
    next_start = args.start_test
    remaining = args.n_tests

    while remaining > 0:
        cmd = base_cmd + [
            '--n_tests', str(remaining),
            '--start_test', str(next_start),
        ]
        print(f"\n{'='*80}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'='*80}")

        process = None
        current_episode = None
        chunk_results = []
        output_tail = []
        summary_total = None
        summary_rate = None
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as child_output:
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=child_output,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                return_code = process.wait()
            finally:
                if process is not None:
                    stop_process_group(process)

            child_output.seek(0)
            for line in child_output:
                output_tail.append(line.rstrip())
                output_tail = output_tail[-20:]
                if args.verbose:
                    print(line, end='', flush=True)
                match = re.search(r'Task: .*?, Test: (\d+)/(\d+)', line)
                if match:
                    current_episode = int(match.group(1))
                if re.search(r'Test \d+ result: SUCCESS', line):
                    chunk_results.append(1.0)
                elif re.search(r'Test \d+ result: FAILURE', line):
                    chunk_results.append(0.0)
                elif line.startswith('Total tests:'):
                    summary_total = int(line.split(':', 1)[1].strip())
                elif line.startswith('Success rate:'):
                    summary_rate = float(line.split(':', 1)[1].strip().rstrip('%'))

        if return_code == 0:
            if len(chunk_results) != remaining:
                if summary_total == remaining and summary_rate is not None:
                    num_successes = round(summary_rate * summary_total / 100.0)
                    chunk_results = (
                        [1.0] * num_successes
                        + [0.0] * (summary_total - num_successes)
                    )
                    print(
                        f"⚠ Parsed {len(chunk_results)} results from the complete "
                        "test summary because one or more per-episode result lines "
                        "were missing.",
                        file=sys.stderr,
                    )
                else:
                    raise RuntimeError(
                        f"test_simulator.py finished but produced {len(chunk_results)} "
                        f"episode results for {remaining} requested tests, and no "
                        "matching complete summary was found.\nLast subprocess output:\n  "
                        + "\n  ".join(output_tail)
                    )
            results.extend(chunk_results)
            break

        if current_episode is None:
            raise RuntimeError(
                f"test_simulator.py exited with code {return_code} before an episode started.\n"
                f"Re-run this task to inspect the setup error:\n  {' '.join(cmd)}"
            )

        # Preserve completed results. If the current episode did not emit a
        # result before the crash, count it as a failure. This also guarantees
        # that retry always makes forward progress.
        current_had_result = len(chunk_results) >= current_episode
        accounted = max(current_episode, len(chunk_results))
        accounted = min(accounted, remaining)
        chunk_results.extend([0.0] * (accounted - len(chunk_results)))
        results.extend(chunk_results)
        crashed_demo = next_start + current_episode - 1
        next_start += accounted
        remaining -= accounted

        crash_outcome = (
            "its recorded result was preserved"
            if current_had_result else "the episode was counted as failure"
        )
        print(
            f"\n⚠ {task_name} demo {crashed_demo} crashed (exit {return_code}); "
            f"{crash_outcome}. Resuming with {remaining} episode(s) remaining.",
            file=sys.stderr,
        )
        if not args.verbose:
            print("Last subprocess output:", file=sys.stderr)
            for line in output_tail:
                print(f"  {line}", file=sys.stderr)

    success_rate = 100.0 * sum(results) / len(results) if results else 0.0
    return success_rate, len(results)


def main():
    parser = argparse.ArgumentParser(
        description='Batch test all tasks from dataloader configuration'
    )
    parser.add_argument('--config_dir', type=str, default='../config',
                       help='Configuration directory')
    parser.add_argument('--n_tests', type=int, default=25,
                       help='Number of test episodes per task per run')
    parser.add_argument('--start_test', type=int, default=5,
                       help='Starting episode index')
    parser.add_argument('--num_runs', type=int, default=3,
                       help='Number of runs to average over')
    parser.add_argument('--disp', action='store_true', default=False,
                       help='Display visualization')
    parser.add_argument('--actor', type=str, default='foci', choices=['foci', 'expert'],
                       help='Actor type: foci, or expert')
    parser.add_argument('--traj_length', type=int, default=3,
                       help='Trajectory length for expert mode (default: 3)')
    parser.add_argument('--device', type=str, default=None,
                       help='CUDA device override, e.g. cuda:1 or 1 (default: dataloader.yaml)')
    parser.add_argument('--pose_method', type=str, default='fp',
                       choices=['gt', 'fp', 'foundation_pose'],
                       help='Pose estimation method')
    parser.add_argument('--add_noise', type=str, default='none',
                       choices=['none', 'mild', 'medium', 'hard'],
                       help='Add noise to Foundation Pose estimates (only with --pose_method fp)')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Enable debug mode')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Print all output from test_simulator.py')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                       help='Specific tasks to test (if not specified, test all)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    args = parser.parse_args()
    
    # Auto-generate output filename if not specified
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'test_results_{args.actor}_{timestamp}.json'
    
    # Get script directory and config path
    script_dir = Path(__file__).parent
    config_dir = script_dir / args.config_dir
    dataloader_config = config_dir / 'dataloader.yaml'
    
    if not dataloader_config.exists():
        print(f"✗ Config file not found: {dataloader_config}")
        return 1
    
    dataloader_cfg = load_config(dataloader_config)
    args.device = args.device or str(dataloader_cfg.get('device', 'cuda:0'))
    if args.device.isdigit():
        args.device = f'cuda:{args.device}'
    if args.device != 'cuda' and not args.device.startswith('cuda:'):
        parser.error(f"Invalid CUDA device '{args.device}'; use cuda:N or N")

    # Load task list
    if args.tasks:
        task_list = args.tasks
        print(f"Testing specified tasks: {task_list}")
    else:
        task_list = dataloader_cfg.get('task_list', [])
        print(f"Loaded {len(task_list)} tasks from config")
    
    if not task_list:
        print("✗ No tasks found")
        return 1
    
    print(f"\n{'='*80}")
    print(f"BATCH TEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"Tasks: {len(task_list)}")
    print(f"Tests per task: {args.n_tests}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Total tests: {len(task_list) * args.n_tests * args.num_runs}")
    print(f"Actor: {args.actor}")
    print(f"Pose method: {args.pose_method}")
    print(f"CUDA device: {args.device}")
    if args.add_noise != 'none' and args.pose_method in ['fp', 'foundation_pose']:
        print(f"Noise level: {args.add_noise}")
    print(f"Display: {args.disp}")
    print(f"Output file: {args.output}")
    print(f"{'='*80}\n")
    
    # Storage for results
    all_results = defaultdict(list)  # task_name -> [success_rates]
    
    # Run tests for each task, multiple runs
    for run_idx in range(args.num_runs):
        print(f"\n{'#'*80}")
        print(f"# RUN {run_idx + 1}/{args.num_runs}")
        print(f"{'#'*80}\n")
        
        for task_idx, task_name in enumerate(task_list):
            print(f"\n[Task {task_idx + 1}/{len(task_list)}] {task_name}")
            print(f"Run {run_idx + 1}/{args.num_runs}")
            
            try:
                success_rate, total_tests = run_task_test(task_name, args)
            except KeyboardInterrupt:
                print(f"\nBatch test interrupted while running '{task_name}'.", file=sys.stderr)
                return 130
            except Exception as exc:
                print(f"\n✗ Error while testing '{task_name}'; batch stopped.", file=sys.stderr)
                print(str(exc), file=sys.stderr)
                return 1
            all_results[task_name].append(success_rate)
            
            print(f"✓ {task_name}: {success_rate:.1f}% ({run_idx + 1}/{args.num_runs})")
    
    # Compute statistics
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS - {args.num_runs} runs averaged")
    print(f"{'='*80}\n")
    
    task_stats = {}
    total_mean = 0.0
    
    for task_name in task_list:
        rates = all_results[task_name]
        if rates:
            mean_rate = sum(rates) / len(rates)
            std_rate = (sum((r - mean_rate) ** 2 for r in rates) / len(rates)) ** 0.5
            task_stats[task_name] = {
                'mean': mean_rate,
                'std': std_rate,
                'runs': rates,
                'num_runs': len(rates)
            }
            total_mean += mean_rate
            
            print(f"{task_name:40s}: {mean_rate:5.1f}% (±{std_rate:4.1f}%)  {rates}")
        else:
            print(f"{task_name:40s}: No results")
            task_stats[task_name] = {
                'mean': 0.0,
                'std': 0.0,
                'runs': [],
                'num_runs': 0
            }
    
    overall_mean = total_mean / len(task_list) if task_list else 0.0
    
    print(f"\n{'='*80}")
    print(f"OVERALL SUCCESS RATE: {overall_mean:.1f}%")
    print(f"{'='*80}\n")
    
    # Save results to JSON
    output_path = Path(args.output)
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'actor': args.actor,
            'pose_method': args.pose_method,
            'add_noise': args.add_noise,
            'demo_num': args.n_tests,
            'n_tests': args.n_tests,
            'start_test': args.start_test,
            'num_runs': args.num_runs,
            'device': args.device,
            'traj_length': args.traj_length,
            'config_dir': str(config_dir),
            'dataloader': dataloader_cfg,
        },
        'task_results': task_stats,
        'overall_mean': overall_mean,
        'num_tasks': len(task_list),
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"✓ Results saved to: {output_path.absolute()}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Task':<40s} {'Mean':<8s} {'Std':<8s} {'Runs':<20s}")
    print(f"{'-'*80}")
    for task_name in task_list:
        stats = task_stats[task_name]
        runs_str = ', '.join([f"{r:.1f}" for r in stats['runs']])
        print(f"{task_name:<40s} {stats['mean']:>6.1f}% {stats['std']:>6.1f}% [{runs_str}]")
    print(f"{'-'*80}")
    print(f"{'OVERALL':<40s} {overall_mean:>6.1f}%")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
