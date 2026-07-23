"""Point-cloud operators implemented with native PyTorch."""

from __future__ import annotations

import torch


@torch.no_grad()
def farthest_point_indices(points: torch.Tensor, count: int) -> torch.Tensor:
    """Return iterative farthest-point-sampling indices.

    Args:
        points: Point coordinates shaped ``(batch, num_points, dims)``.
        count: Number of points to sample. The first selected point is index 0,
            matching the PointNet2 CUDA operator previously used by FOCI.
    """
    if points.ndim != 3:
        raise ValueError(f"Expected (B, N, D) points, got {tuple(points.shape)}")
    batch_size, num_points, _ = points.shape
    if not 0 < count <= num_points:
        raise ValueError(f"count must be in [1, {num_points}], got {count}")

    indices = torch.empty(
        (batch_size, count), dtype=torch.long, device=points.device)
    min_distances = torch.full(
        (batch_size, num_points), float("inf"),
        dtype=points.dtype, device=points.device)
    farthest = torch.zeros(batch_size, dtype=torch.long, device=points.device)
    batch_indices = torch.arange(batch_size, device=points.device)

    for sample_idx in range(count):
        indices[:, sample_idx] = farthest
        centroid = points[batch_indices, farthest].unsqueeze(1)
        distances = torch.sum((points - centroid) ** 2, dim=-1)
        min_distances = torch.minimum(min_distances, distances)
        farthest = torch.max(min_distances, dim=1).indices
    return indices


def gather_points(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``(B, C, N)`` features at ``(B, S)`` point indices."""
    if features.ndim != 3 or indices.ndim != 2:
        raise ValueError(
            "Expected features (B, C, N) and indices (B, S), got "
            f"{tuple(features.shape)} and {tuple(indices.shape)}")
    if features.shape[0] != indices.shape[0]:
        raise ValueError("features and indices must have the same batch size")
    expanded = indices.unsqueeze(1).expand(-1, features.shape[1], -1)
    return torch.gather(features, dim=2, index=expanded)


@torch.no_grad()
def knn_indices(
    reference: torch.Tensor,
    query: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Return indices of the ``k`` nearest reference points for each query.

    ``reference`` is shaped ``(B, N, D)`` and ``query`` is ``(B, Q, D)``;
    the returned tensor is shaped ``(B, Q, k)``.
    """
    if reference.ndim != 3 or query.ndim != 3:
        raise ValueError(
            "Expected reference (B, N, D) and query (B, Q, D), got "
            f"{tuple(reference.shape)} and {tuple(query.shape)}")
    if reference.shape[0] != query.shape[0] or reference.shape[2] != query.shape[2]:
        raise ValueError("reference and query batch/coordinate dimensions must match")
    if not 0 < k <= reference.shape[1]:
        raise ValueError(f"k must be in [1, {reference.shape[1]}], got {k}")

    distances = torch.cdist(query.float(), reference.float(), p=2)
    return torch.topk(
        distances, k=k, dim=-1, largest=False, sorted=True).indices
