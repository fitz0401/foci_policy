import os
import sys
import yaml
import torch
import torch.nn as nn
import math
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foci_policy.model.foci_dataloader import FOCI_Dataset, AddLanguageEmbedding
from foci_policy.model.foci_model import FOCIModel, FOCILoss
from foci_policy.utils.transform_utils import matrix_to_pose_9d


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_point_encoder_config(pcd_encoder_path):
    """Load point encoder configuration from YAML"""
    with open(pcd_encoder_path, 'r') as f:
        pcd_config = yaml.safe_load(f)

    # Convert to the format expected by MaskedPointNetEncoder
    network_kwargs = pcd_config['network_kwargs']

    # Create config object with proper structure
    point_encoder_cfg = {
        'group_cfg': type('GroupCfg', (), {
            'num_group': network_kwargs['group_cfg']['num_group'],
            'group_size': network_kwargs['group_cfg']['group_size']
        })(),
        'masked_encoder_cfg': type('MaskCfg', (), {
            'mask_ratio': network_kwargs['masked_encoder_cfg']['mask_ratio'],
            'mask_type': network_kwargs['masked_encoder_cfg']['mask_type'],
            'embed_dim': network_kwargs['masked_encoder_cfg']['embed_dim']  # Will be set later
        })(),
        'output_size': network_kwargs['output_size'],  # Will be set later
        'skip_group': False,
    }
    return point_encoder_cfg


def load_dataloader_config(dataloader_path):
    """Load dataloader configuration from YAML"""
    with open(dataloader_path, 'r') as f:
        dataloader_config = yaml.safe_load(f)
    return dataloader_config


def create_optimizer(model, optimizer_config):
    """Create optimizer from configuration"""
    if optimizer_config['type'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config.get('lr', 1e-4),
            weight_decay=optimizer_config.get('weight_decay', 0.01),
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            eps=optimizer_config.get('eps', 1e-8),
        )
    elif optimizer_config['type'].lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config.get('lr', 1e-4),
            weight_decay=optimizer_config.get('weight_decay', 0.01),
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            eps=optimizer_config.get('eps', 1e-8),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_config['type']}")
    return optimizer


def create_scheduler(optimizer, scheduler_config, training_config):
    """Create learning rate scheduler from configuration"""
    if scheduler_config['type'].lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', training_config['num_epochs']),
            eta_min=scheduler_config.get('eta_min', 1e-6),
        )
    elif scheduler_config['type'].lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.1),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_config['type']}")
    return scheduler


def collate_fn(batch):
    '''Custom collate function to batch data samples'''
    class BatchData:
        pass
    
    result = BatchData()
    result.pa_points = torch.stack([item.pa_points for item in batch])
    result.pa_colors = torch.stack([item.pa_colors for item in batch])
    result.pb_points = torch.stack([item.pb_points for item in batch])
    result.pb_colors = torch.stack([item.pb_colors for item in batch])
    result.current_pose = torch.stack([item.current_pose for item in batch])
    result.trajectory_poses = torch.stack([item.trajectory_poses for item in batch])
    
    if hasattr(batch[0], 'language_embedding'):
        result.language_embedding = torch.stack([item.language_embedding for item in batch])
    
    return result


def create_model(model_config, point_encoder_cfg, dataloader_cfg):
    """Create FOCI model from configuration"""
    point_encoder_cfg['masked_encoder_cfg'].embed_dim = model_config['pcd_dim']
    point_encoder_cfg['output_size'] = model_config['pcd_dim']
    model = FOCIModel(
        point_encoder_cfg=point_encoder_cfg,
        pcd_dim=model_config['pcd_dim'],
        pose_dim=model_config['pose_dim'],
        prediction_length=dataloader_cfg['prediction_length'],
        nhead=model_config['nhead'],
        num_decoder_layers=model_config['num_decoder_layers'],
        dropout=model_config['dropout'],
        use_language=model_config['use_language'],
        mode=model_config['mode'],
        use_pcd_features=model_config.get('use_pcd_features', True),
        use_pose_features=model_config.get('use_pose_features', True),
        num_gaussians=model_config['num_gaussians'],
        goal_hidden_dim=model_config['goal_hidden_dim'],
        waypoint_nhead=model_config.get('waypoint_nhead', model_config.get('action_nhead', 4)),
        waypoint_num_layers=model_config.get('waypoint_num_layers', model_config.get('action_num_layers', 2)),
    )
    return model


def determine_training_phase(epoch, config):
    """Determine current training phase based on epoch (3-stage)"""
    phase_config = config['training'].get('phase', {})

    if isinstance(phase_config, dict):
        phase_type = phase_config.get('type', 'auto')
        stage1_epochs = phase_config.get('stage1_epochs', 50)
        stage2_epochs = phase_config.get('stage2_epochs', 50)
    else:
        phase_type = phase_config
        stage1_epochs = config['training'].get('stage1_epochs', 50)
        stage2_epochs = config['training'].get('stage2_epochs', 50)

    if phase_type in ('stage1', 'stage2', 'stage3'):
        return phase_type
    elif phase_type == 'auto':
        if epoch < stage1_epochs:
            return 'stage1'
        elif epoch < stage1_epochs + stage2_epochs:
            return 'stage2'
        else:
            return 'stage3'
    else:
        raise ValueError(f"Unknown phase type: {phase_type}")


def freeze_parameters(model, phase):
    """Freeze/unfreeze parameters based on training phase (3-stage)"""
    for param in model.parameters():
        param.requires_grad = True

    if phase == 'stage1':
        # Train encoder + GMM, freeze waypoint decoder
        for name, param in model.named_parameters():
            if 'waypoint_decoder' in name:
                param.requires_grad = False
    elif phase == 'stage2':
        # Train only waypoint decoder with GT goal; freeze everything else
        for name, param in model.named_parameters():
            if 'waypoint_decoder' not in name:
                param.requires_grad = False
    # stage3: all params unfrozen


def train_epoch(model, train_loader, criterion, optimizer, epoch, config, device, use_wandb=False):
    """Train for one epoch"""
    model.train()
    
    # Determine training phase
    phase = determine_training_phase(epoch, config)
    
    # Freeze/unfreeze parameters based on phase
    freeze_parameters(model, phase)
    
    # Initialize loss tracking
    epoch_loss_dict = {'total': []}
    if phase == 'stage1':
        epoch_loss_dict['nll'] = []
    elif phase == 'stage2':
        epoch_loss_dict.update({'waypoint_pos': [], 'waypoint_rot': []})
    elif phase == 'stage3':
        epoch_loss_dict.update({'nll': [], 'waypoint_pos': [], 'waypoint_rot': []})
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [{phase}]", 
                leave=False, dynamic_ncols=True, position=0)
    
    for batch in pbar:
        # Move data to device
        pa_points = batch.pa_points.to(device)
        pb_points = batch.pb_points.to(device)
        pa_pose_matrix = batch.current_pose.to(device)  # (B, 4, 4)
        gt_trajectory_matrix = batch.trajectory_poses.to(device)  # (B, T, 4, 4)
        
        # Convert to 9D representation
        pa_pose = matrix_to_pose_9d(pa_pose_matrix)  # (B, 9)
        pb_pose = torch.zeros_like(pa_pose)  # pb is at origin in canonical frame
        gt_trajectory = matrix_to_pose_9d(gt_trajectory_matrix)  # (B, T, 9)
        
        language_embedding = None
        if hasattr(batch, 'language_embedding'):
            language_embedding = batch.language_embedding.to(device)
        
        # Forward pass: teacher-force waypoint decoder with GT goal in stage2
        gt_goal = gt_trajectory[:, -1, :] if phase == 'stage2' else None
        goal_pose, pred_waypoints, pred_trajectory, gmm_params = model(
            pa_points, pb_points, pa_pose, pb_pose,
            language_embedding=language_embedding,
            return_gmm_params=True,
            deterministic_goal=False,
            gt_goal=gt_goal,
        )
        
        # Compute loss
        loss, losses = criterion(
            phase=phase,
            goal_pose=goal_pose,
            pred_waypoints=pred_waypoints,
            pred_trajectory=pred_trajectory,
            gmm_params=gmm_params,
            gt_trajectory=gt_trajectory
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config['training'].get('grad_clip_norm', None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip_norm'])
        
        optimizer.step()
        
        # Track losses
        epoch_loss_dict['total'].append(loss.item())
        for key, value in losses.items():
            if key != 'total' and key in epoch_loss_dict:
                epoch_loss_dict[key].append(value.item())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'phase': phase})
    
    # Compute average losses
    avg_losses = {key: sum(values) / len(values) if len(values) > 0 else 0.0 
                  for key, values in epoch_loss_dict.items()}
    
    # Log to wandb
    if use_wandb:
        try:
            import wandb
            log_dict = {f'train/{key}': value for key, value in avg_losses.items()}
            log_dict['train/epoch'] = epoch
            log_dict['train/phase'] = phase
            log_dict['train/lr'] = optimizer.param_groups[0]['lr']
            wandb.log(log_dict)
        except ImportError:
            pass
    
    return avg_losses, phase


def evaluate(model, test_loader, criterion, device, use_wandb=False, epoch=0, phase='joint'):
    """Evaluate model on test set"""
    model.eval()
    
    # Initialize loss tracking
    eval_loss_dict = {'total': []}
    if phase == 'stage1':
        eval_loss_dict['nll'] = []
    elif phase == 'stage2':
        eval_loss_dict.update({'waypoint_pos': [], 'waypoint_rot': []})
    elif phase == 'stage3':
        eval_loss_dict.update({'nll': [], 'waypoint_pos': [], 'waypoint_rot': []})
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False, dynamic_ncols=True):
            # Move data to device
            pa_points = batch.pa_points.to(device)
            pb_points = batch.pb_points.to(device)
            pa_pose_matrix = batch.current_pose.to(device)  # (B, 4, 4)
            gt_trajectory_matrix = batch.trajectory_poses.to(device)  # (B, T, 4, 4)
            
            # Convert to 9D representation
            pa_pose = matrix_to_pose_9d(pa_pose_matrix)  # (B, 9)
            pb_pose = torch.zeros_like(pa_pose)  # pb is at origin in canonical frame
            gt_trajectory = matrix_to_pose_9d(gt_trajectory_matrix)  # (B, T, 9)
            
            language_embedding = None
            if hasattr(batch, 'language_embedding'):
                language_embedding = batch.language_embedding.to(device)
            
            # Forward pass
            goal_pose, pred_waypoints, pred_trajectory, gmm_params = model(
                pa_points, pb_points, pa_pose, pb_pose,
                language_embedding=language_embedding,
                return_gmm_params=True,
                deterministic_goal=True,
            )
            
            # Compute loss
            loss, losses = criterion(
                phase=phase,
                goal_pose=goal_pose,
                pred_waypoints=pred_waypoints,
                pred_trajectory=pred_trajectory,
                gmm_params=gmm_params,
                gt_trajectory=gt_trajectory
            )
            
            # Track losses
            eval_loss_dict['total'].append(loss.item())
            for key, value in losses.items():
                if key != 'total' and key in eval_loss_dict:
                    eval_loss_dict[key].append(value.item())
    
    # Compute average losses
    avg_losses = {key: sum(values) / len(values) if len(values) > 0 else 0.0 
                  for key, values in eval_loss_dict.items()}
    
    # Log to wandb
    if use_wandb:
        try:
            import wandb
            log_dict = {f'test/{key}': value for key, value in avg_losses.items()}
            log_dict['test/epoch'] = epoch
            log_dict['test/phase'] = phase
            wandb.log(log_dict)
        except ImportError:
            pass
    
    return avg_losses


def save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, config, mode, phase, data_source='simu'):
    """Save model checkpoint"""
    checkpoint_config = config.get('checkpoint', {})
    save_dir = checkpoint_config.get('save_dir', '../checkpoints/foci')
    # Add data_source to checkpoint path
    checkpoint_dir = Path(save_dir) / data_source / mode
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}_{phase}.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss,
        'config': config,
        'mode': mode,
        'phase': phase,
        'data_source': data_source,
    }, checkpoint_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Keep only last N checkpoints
    keep_last_n = checkpoint_config.get('keep_last_n', 5)
    checkpoints = sorted(
        checkpoint_dir.glob('checkpoint_epoch_*.pth'),
        key=lambda x: int(x.stem.split('_')[2])
    )
    if len(checkpoints) > keep_last_n:
        for old_checkpoint in checkpoints[:-keep_last_n]:
            old_checkpoint.unlink()


def train_single_mode(config_dir, dataset_root, mode, data_source='simu'):
    """ Train single mode (grasp or manip) model """
    # Load configurations
    config_dir = Path(config_dir)
    foci_config_path = config_dir / 'traj_decoder' / 'foci_model.yaml'
    pcd_encoder_path = config_dir / 'pcd_encoder' / 'masked_pointnet_encoder.yaml'
    dataloader_path = config_dir / 'dataloader.yaml'
    
    foci_config = load_config(foci_config_path)
    point_encoder_cfg = load_point_encoder_config(pcd_encoder_path)
    dataloader_config = load_dataloader_config(dataloader_path)
    
    # Set mode
    foci_config['model']['mode'] = mode
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(foci_config['model'], point_encoder_cfg, dataloader_config)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Initialize wandb if enabled
    use_wandb = foci_config['logging'].get('use_wandb', False)
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=foci_config['logging']['wandb_project'],
                entity=foci_config['logging'].get('wandb_entity'),
                name=f'foci_{mode}_{data_source}',
                config={
                    'model': foci_config['model'],
                    'training': foci_config['training'],
                    'loss': foci_config['loss'],
                    'mode': mode,
                    'data_source': data_source,
                }
            )
        except ImportError:
            print("wandb not available, skipping wandb logging")
            use_wandb = False
    
    # Create dataset and dataloader
    print("Loading dataset...")
    
    # Add language embedding preprocessing if use_language is enabled
    pre_transform = None
    if foci_config['model']['use_language']:
        pre_transform = AddLanguageEmbedding(device=device)
    
    # Get task list from config
    task_list = None
    if data_source == 'simu':
        task_list = dataloader_config.get('task_list', None)
    elif data_source == 'real':
        task_list = dataloader_config.get('task_list_realworld', None)
    _demo_key = f'num_demos_{mode}'
    num_demos = dataloader_config.get(_demo_key, 1)
    optimal = dataloader_config.get('optimal', False)
    print(f"[DataLoader] mode={mode}, num_demos={num_demos} (key: '{_demo_key}'), optimal={optimal}")

    train_dataset = FOCI_Dataset(
        root=dataset_root,
        mode=mode,
        train=True,
        prediction_length=dataloader_config['prediction_length'],
        num_demos=num_demos,
        task_list=task_list,
        sampling_ratio=dataloader_config.get('sampling_ratio', 0.6),
        max_demos_per_task=dataloader_config.get('max_demos_per_task', 50),
        pre_transform=pre_transform,
        optimal=optimal,
    )
    
    test_dataset = FOCI_Dataset(
        root=dataset_root,
        mode=mode,
        train=False,
        prediction_length=dataloader_config['prediction_length'],
        num_demos=num_demos,
        task_list=task_list,
        sampling_ratio=dataloader_config.get('sampling_ratio', 0.6),
        max_demos_per_task=dataloader_config.get('max_demos_per_task', 50),
        pre_transform=pre_transform,
        optimal=optimal,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=foci_config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=foci_config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create loss function
    criterion = FOCILoss(
        lambda_nll=foci_config['loss']['lambda_nll'],
        lambda_waypoint_pos=foci_config['loss'].get('lambda_waypoint_pos', foci_config['loss'].get('lambda_action_pos', 1.0)),
        lambda_waypoint_rot=foci_config['loss'].get('lambda_waypoint_rot', foci_config['loss'].get('lambda_action_rot', 5.0)),
        lambda_stage1=foci_config['loss']['lambda_stage1'],
        lambda_stage2=foci_config['loss']['lambda_stage2'],
        gamma_geo=foci_config['loss']['gamma_geo'],
    )
    
    # Create optimizer
    optimizer_config = {
        'type': 'adamw',
        'lr': foci_config['training']['learning_rate'],
        'weight_decay': foci_config['training']['weight_decay']
    }
    optimizer = create_optimizer(model, optimizer_config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, foci_config['training']['scheduler'], foci_config['training'])
    
    # Training loop
    import time
    print("Starting training...\n")
    best_test_loss = float('inf')
    start_time = time.time()

    for epoch in range(foci_config['training']['num_epochs']):
        # Train
        train_losses, phase = train_epoch(model, train_loader, criterion, optimizer, 
                                         epoch, foci_config, device, use_wandb)
        # Evaluate
        eval_interval = foci_config.get('logging', {}).get('eval_interval', 1)
        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            test_losses = evaluate(model, test_loader, criterion, device, use_wandb, epoch, phase)
        else:
            test_losses = None
        # Update learning rate
        scheduler.step()
        # Log to wandb
        if use_wandb:
            try:
                import wandb
                wandb.log({'learning_rate': scheduler.get_last_lr()[0], 'epoch': epoch})
            except ImportError:
                pass
        # Print progress at intervals
        log_interval = foci_config.get('logging', {}).get('log_interval', 10)
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            # Format loss details based on phase
            train_details = f"total={train_losses['total']:.4f}"
            test_details = f"total={test_losses['total']:.4f}" if test_losses else "skipped"
            if phase == 'stage1' and 'nll' in train_losses:
                train_details += f", nll={train_losses['nll']:.4f}"
                if test_losses:
                    test_details += f", nll={test_losses['nll']:.4f}"
            elif phase == 'stage2' and 'waypoint_pos' in train_losses:
                train_details += f", pos={train_losses['waypoint_pos']:.4f}, rot={train_losses['waypoint_rot']:.4f}"
                if test_losses:
                    test_details += f", pos={test_losses['waypoint_pos']:.4f}, rot={test_losses['waypoint_rot']:.4f}"
            elif phase == 'stage3' and 'nll' in train_losses:
                train_details += f", nll={train_losses['nll']:.4f}, pos={train_losses['waypoint_pos']:.4f}, rot={train_losses['waypoint_rot']:.4f}"
                if test_losses:
                    test_details += f", nll={test_losses['nll']:.4f}, pos={test_losses['waypoint_pos']:.4f}, rot={test_losses['waypoint_rot']:.4f}"
            print(f"Epoch {epoch+1:3d}/{foci_config['training']['num_epochs']} [{phase.upper()}] | "
                  f"Train: {train_details} | "
                  f"Test: {test_details} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
        # Save periodic checkpoint
        save_interval = foci_config.get('checkpoint', {}).get('save_interval', 10)
        if (epoch + 1) % save_interval == 0:
            ckpt_loss = test_losses['total'] if test_losses else float('inf')
            save_checkpoint(model, optimizer, scheduler, epoch, ckpt_loss,
                          foci_config, mode, phase, data_source)
        # Save best model (only in stage3 where joint loss is comparable across epochs)
        if test_losses and phase == 'stage3' and test_losses['total'] < best_test_loss:
            best_test_loss = test_losses['total']
            save_dir = foci_config.get('checkpoint', {}).get('save_dir', '../checkpoints/foci')
            best_checkpoint_dir = Path(save_dir) / data_source / mode
            best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_checkpoint_path = best_checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_losses['total'],
                'test_loss': best_test_loss,
                'config': foci_config,
                'mode': mode,
                'phase': phase,
                'data_source': data_source,
            }, best_checkpoint_path)
            # print(f"Best model saved to {best_checkpoint_path} with test loss: {best_test_loss:.4f}")
            if use_wandb:
                try:
                    import wandb
                    wandb.log({'best_test_loss': best_test_loss, 'epoch': epoch})
                except ImportError:
                    pass
    elapsed = time.time() - start_time
    print(f"\n{mode.upper()} mode training completed!")
    print(f"Best test loss: {best_test_loss:.4f}")
    print(f"Total training time: {elapsed/60:.2f} min ({elapsed:.1f} sec)")
    # Close wandb
    if use_wandb:
        wandb.finish()
    return best_test_loss, elapsed


def train_foci_models(config_dir='../config', dataset_root='../../foci_dataset', data_source='simu'):
    """ Train both grasp and manip models sequentially """
    print("="*80)
    print(f"Training FOCI Models - Grasp and Manip ({data_source.upper()})")
    print("="*80)
    
    results = {}
    # Train grasp model
    print("\n" + "="*80)
    print("PHASE 1/2: GRASP Model")
    print("="*80)
    try:
        grasp_loss, grasp_time = train_single_mode(config_dir, dataset_root, mode='grasp', data_source=data_source)
        results['grasp'] = {'status': 'success', 'best_loss': grasp_loss, 'time': grasp_time}
    except Exception as e:
        print(f"Error training grasp model: {e}")
        import traceback
        traceback.print_exc()
        results['grasp'] = {'status': 'failed', 'error': str(e)}
    # Train manip model
    print("\n" + "="*80)
    print("PHASE 2/2: MANIP Model")
    print("="*80)
    try:
        manip_loss, manip_time = train_single_mode(config_dir, dataset_root, mode='manip', data_source=data_source)
        results['manip'] = {'status': 'success', 'best_loss': manip_loss, 'time': manip_time}
    except Exception as e:
        print(f"Error training manip model: {e}")
        import traceback
        traceback.print_exc()
        results['manip'] = {'status': 'failed', 'error': str(e)}
    # Print final summary
    print("\n" + "="*80)
    print("Training Summary")
    print("="*80)
    for mode, result in results.items():
        print(f"\n{mode.upper()} Model:")
        if result['status'] == 'success':
            print(f"  Status: ✓ Success")
            print(f"  Best Test Loss: {result['best_loss']:.4f}")
            print(f"  Total Training Time: {result['time']/60:.2f} min ({result['time']:.1f} sec)")
        else:
            print(f"  Status: ✗ Failed")
            print(f"  Error: {result['error']}")
    print("\n" + "="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train FOCI models for FOCI policy')
    parser.add_argument('--config_dir', type=str, default='../config',
                       help='Directory containing configuration files')
    parser.add_argument('--mode', type=str, choices=['grasp', 'manip', 'both'], default='both',
                       help='Training mode: grasp, manip, or both')
    parser.add_argument('--data_source', type=str, choices=['simu', 'real'], default='simu',
                       help='Data source: simu for RLBench simulation, real for real-world data')
    
    args = parser.parse_args()
    
    # Adjust dataset root for real-world data if not explicitly specified
    if args.data_source == 'real':
        dataset_root = '../../foci_dataset_realworld'
    else:
        dataset_root = '../../foci_dataset'
    print(f"Using dataset root: {dataset_root}")
    
    if args.mode == 'both':
        train_foci_models(args.config_dir, dataset_root, args.data_source)
    else:
        train_single_mode(args.config_dir, dataset_root, args.mode, args.data_source)
