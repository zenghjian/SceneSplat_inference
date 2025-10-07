"""
Misc utility functions for point cloud operations
"""

import torch


def offset2batch(offset):
    """Convert offset to batch indices"""
    return torch.cat([
        torch.full((offset[i + 1] - offset[i],), i, dtype=torch.long, device=offset.device)
        for i in range(len(offset) - 1)
    ])


def batch2offset(batch):
    """Convert batch indices to offset"""
    return torch.cat([
        torch.tensor([0], dtype=torch.long, device=batch.device),
        torch.cumsum(torch.bincount(batch), dim=0)
    ])


def offset2bincount(offset):
    """Convert offset to bin count"""
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


def bincount2offset(bincount):
    """Convert bin count to offset"""
    return torch.cat([
        torch.tensor([0], dtype=torch.long, device=bincount.device),
        torch.cumsum(bincount, dim=0)
    ])


def off_diagonal(x):
    """Return a flattened view of the off-diagonal elements of a square matrix"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten() 