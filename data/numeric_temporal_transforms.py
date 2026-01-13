import torch
import matplotlib.pyplot as plt
from pathlib import Path


def augment_numeric_timeseries(
    kinematic: torch.Tensor,   # [T, K]
    bbox: torch.Tensor,        # [T, B]
    noise_std=0.01,
    scale_range=(0.95, 1.05),
    max_time_shift=1,
    debug=False,
    sample_index=None,
    feature_names=None,        # optional list of kinematic feature names
):
    """
    Numeric time-series augmentation with optional debugging plots.
    """

    augment_save_path = Path(r"C:\Projects\hazard_prediction_project\output\visualizations\Numeric Time Series Augmentation")
    # -------------------------
    # Preserve originals
    # -------------------------
    kin_orig = kinematic.clone()
    bbox_orig = bbox.clone()

    kinematic = kinematic.clone()
    bbox = bbox.clone()

    T, K = kinematic.shape

    # Utility for plotting
    def plot_compare(orig, aug, title, feature_idx=0):
        plt.figure(figsize=(10, 4))
        plt.plot(orig[:, feature_idx].cpu(), label="original", linewidth=2)
        plt.plot(aug[:, feature_idx].cpu(), label="augmented", linestyle="--")
        fname = feature_names[feature_idx] if feature_names else f"feature_{feature_idx}"
        plt.title(f"{title} — {fname}")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(augment_save_path / f"{title} — {fname}_{sample_index}.png", dpi=300, format='png')
        #plt.show()

    # -------------------------
    # 1. Kinematic noise
    # -------------------------
    noise = torch.randn_like(kinematic) * noise_std
    kinematic_noisy = kinematic + noise

    if debug:
        plot_compare(kin_orig, kinematic_noisy, "Kinematic Gaussian Noise")

    kinematic = kinematic_noisy

    # -------------------------
    # 2. Dynamics scaling
    # -------------------------
    scale = torch.empty(1).uniform_(*scale_range)
    kinematic_scaled = kinematic * scale

    if debug:
        plot_compare(kin_orig, kinematic_scaled, f"Dynamics Scaling (×{scale.item():.3f})")

    kinematic = kinematic_scaled

    # -------------------------
    # 3. Mild bbox noise
    # -------------------------
    bbox_noise = torch.randn_like(bbox) * (noise_std * 0.5)
    bbox_noisy = bbox + bbox_noise

    if debug:
        plt.figure(figsize=(10, 4))
        plt.plot(bbox_orig[:, 0].cpu(), label="original bbox[0]", linewidth=2)
        plt.plot(bbox_noisy[:, 0].cpu(), label="augmented bbox[0]", linestyle="--")
        plt.title("BBox Noise Injection")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(augment_save_path / f"BBox Noise Injection_{sample_index}.png", dpi=300, format='png')
        #plt.show()

    bbox = bbox_noisy

    # -------------------------
    # 4. Temporal jitter is not ideal when using both numeric and frames.
    # -------------------------
    if max_time_shift > 0:
        shift = torch.randint(-max_time_shift, max_time_shift + 1, (1,)).item()
        kinematic_shifted = torch.roll(kinematic, shift, dims=0)
        bbox_shifted = torch.roll(bbox, shift, dims=0)

        if debug:
            plot_compare(
                kin_orig,
                kinematic_shifted,
                f"Temporal Jitter (shift={shift})"
            )

        kinematic = kinematic_shifted
        bbox = bbox_shifted

    return kinematic, bbox
