from __future__ import print_function, division
import torch
import scipy.io as scp
import numpy as np
import random
from scipy.interpolate import interp1d



def augment_video_frames(frames, augmenter):
    """
    frames: torch.Tensor [T, C, H, W]
    augmenter: tsaug augmenter
    """
    T, C, H, W = frames.shape
    # Flatten frames to (T, D)
    flat = frames.view(T, -1).cpu().numpy()[None, :, :]  # shape [1, T, D]

    # Apply tsaug temporal augmentation
    flat_aug = augmenter.augment(flat)  # shape [1, T, D]

    # Convert back to torch.Tensor [T, C, H, W]
    flat_aug = torch.tensor(flat_aug[0]).float().view(T, C, H, W)

    return flat_aug


def temporal_jittering_fixed(frames, T, drop_prob=0.1):
    """
    Temporal jittering but ensures fixed length T.
    frames: Tensor [T, C, H, W]
    """
    new_frames = []
    for f in frames:
        if random.random() < drop_prob:
            # Drop this frame
            print("frame dropped")
            continue
        new_frames.append(f)
        if random.random() < drop_prob / 2:
            # Duplicate this frame
            new_frames.append(f.clone())

    # Handle cases where length changes
    if len(new_frames) == 0:  
        new_frames = [frames[0]] * T   # fallback if all dropped

    # Pad or trim back to T
    if len(new_frames) < T:
        pad = [new_frames[-1]] * (T - len(new_frames))
        new_frames.extend(pad)
    elif len(new_frames) > T:
        # randomly select T frames from sequence
        idxs = sorted(random.sample(range(len(new_frames)), T))
        new_frames = [new_frames[i] for i in idxs]

    return torch.stack(new_frames)


def temporal_time_warp(frames, T, sigma=0.2):
    """
    Apply temporal time warping to a sequence while keeping fixed length T.
    
    Args:
        frames: torch.Tensor of shape [T, ...] (video frames or features)
        T: int, number of frames (sequence length to keep)
        sigma: float, strength of warping (0.0 = no warp, higher = more warp)
    
    Returns:
        warped_frames: torch.Tensor of shape [T, ...]
    """
    device = frames.device
    orig_T = frames.shape[0]

    # Original timeline: [0, 1, 2, ..., T-1]
    orig_steps = np.arange(orig_T)

    # Generate a smooth random warping curve using cumulative sum of noise
    warp_noise = np.random.normal(loc=0, scale=sigma, size=orig_T)
    warp_curve = np.cumsum(warp_noise)
    warp_curve = (warp_curve - warp_curve.min()) / (warp_curve.max() - warp_curve.min() + 1e-8)  # normalize [0,1]
    warp_curve = warp_curve * (orig_T - 1)

    # Interpolator to map from warped timeline back to original
    f = interp1d(warp_curve, orig_steps, kind='linear', fill_value="extrapolate")

    # New warped indices (still T points, evenly spaced)
    new_steps = np.linspace(warp_curve.min(), warp_curve.max(), T)
    new_indices = f(new_steps)

    # Convert to torch indices and interpolate
    new_indices = torch.tensor(new_indices, dtype=torch.float32, device=device)

    # If frames are images: [T, C, H, W] → interpolate per frame index
    if frames.ndim == 4:  # video frames
        warped_frames = torch.stack([
            frames[int(torch.clamp(idx, 0, orig_T - 1))] for idx in new_indices
        ])
    else:  # features [T, D]
        warped_frames = torch.stack([
            frames[int(torch.clamp(idx, 0, orig_T - 1))] for idx in new_indices
        ])

    return warped_frames



def temporal_jitter(sequence, max_offset=2):
    """Randomly shift temporal window within a small offset."""
    if len(sequence) <= max_offset * 2:
        return sequence
    offset = random.randint(-max_offset, max_offset)
    start = max(0, offset)
    end = len(sequence) - (max(0, -offset))
    return sequence[start:end]