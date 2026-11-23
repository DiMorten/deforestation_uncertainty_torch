import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm


def compute_fixed_grid_patch_coverage(
    image: np.ndarray,
    top_percent: float,
    patch_size: int | tuple[int, int],
    return_boxes: bool = False,
) -> float | tuple[float, list[tuple[int, int, int, int]]]:
    """Fixed-grid (non-overlapping) patch selection."""
    if not (0 <= top_percent <= 100):
        raise ValueError("top_percent must be between 0 and 100.")
    if top_percent == 0:
        return (0.0, []) if return_boxes else 0.0

    uncertainty_map = image.mean(axis=-1) if image.ndim == 3 else image
    H, W = uncertainty_map.shape
    ph, pw = (
        (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    )

    pad_h = (ph - H % ph) % ph
    pad_w = (pw - W % pw) % pw
    padded = np.pad(
        uncertainty_map, ((0, pad_h), (0, pad_w)), constant_values=-1.0
    )
    grid_h, grid_w = padded.shape[0] // ph, padded.shape[1] // pw

    total_pixels = H * W
    num_target = int(np.ceil(total_pixels * (top_percent / 100.0)))
    threshold = np.partition(uncertainty_map.ravel(), -num_target)[-num_target]
    target_mask = padded >= threshold

    patches = target_mask.reshape(grid_h, ph, grid_w, pw)
    selected_patches = patches.any(axis=(1, 3))

    patch_areas = np.ones((grid_h, grid_w), dtype=np.float64) * (ph * pw)
    if pad_h > 0:
        patch_areas[-1, :] *= (ph - pad_h) / ph
    if pad_w > 0:
        patch_areas[:, -1] *= (pw - pad_w) / pw

    covered_area_pct = float(
        (patch_areas[selected_patches].sum() / total_pixels) * 100.0
    )

    if not return_boxes:
        return covered_area_pct

    # Extract (y_min, x_min, y_max, x_max) clipped to original image bounds
    ys, xs = np.where(selected_patches)
    boxes = [
        (
            int(y * ph),
            int(x * pw),
            int(min((y + 1) * ph, H)),
            int(min((x + 1) * pw, W)),
        )
        for y, x in zip(ys, xs)
    ]
    return covered_area_pct, boxes


def compute_adaptive_patch_coverage(
    image: np.ndarray,
    top_percent: float,
    patch_size: int | tuple[int, int],
    border_margin: int = 0,
    overlap_penalty: float = 0.05,
    max_overlap_ratio: float | None = None,
    return_boxes: bool = False,
) -> float | tuple[float, list[tuple[int, int, int, int]]]:
    """Fast greedy patch selection that minimizes overlapping patches.

    Args:
        image: 2D (H, W) or 3D (H, W, C) array with uncertainty values in [0,
          1].
        top_percent: Percentage of highest-value pixels to cover (0 to 100).
        patch_size: (patch_h, patch_w) or int for square patches.
        border_margin: Minimum context pixels required around each target pixel.
        overlap_penalty: Weight (0.0 to 1.0) penalizing already-covered pixels.
          Higher values (e.g. 0.05 - 0.2) force patches to tile edge-to-edge
          instead of overlapping.
        max_overlap_ratio: Optional hard cap (0.0 to 1.0) on how much of a new
          patch can overlap already-covered area (e.g., 0.25 for max 25%
          overlap). Falls back to allowing overlap only if uncovered target
          pixels would otherwise be stranded.
        return_boxes: If True, also returns [(y_min, x_min, y_max, x_max), ...].
    """
    if not (0 <= top_percent <= 100):
        raise ValueError("top_percent must be between 0 and 100.")
    if top_percent == 0:
        return (0.0, []) if return_boxes else 0.0

    uncertainty_map = image.mean(axis=-1) if image.ndim == 3 else image
    H, W = uncertainty_map.shape
    ph, pw = (
        (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    )
    patch_area = ph * pw

    if ph > H or pw > W:
        raise ValueError(
            f"Patch size {(ph, pw)} cannot exceed image size {(H, W)}."
        )
    if border_margin * 2 >= ph or border_margin * 2 >= pw:
        raise ValueError(
            "border_margin is too large for the given patch_size."
        )

    # 1. Threshold top X% pixels
    total_pixels = H * W
    num_target = int(np.ceil(total_pixels * (top_percent / 100.0)))
    threshold = np.partition(uncertainty_map.ravel(), -num_target)[-num_target]
    target_mask = uncertainty_map >= threshold

    # 2. Apply border margin via dilation
    if border_margin > 0:
        struct_size = 2 * border_margin + 1
        target_mask = binary_dilation(
            target_mask, structure=np.ones((struct_size, struct_size), dtype=bool)
        )

    uncovered = target_mask.astype(np.int32)
    covered_canvas = np.zeros((H, W), dtype=np.int32)
    valid_h, valid_w = H - ph + 1, W - pw + 1

    # 3. Initial integral image for uncovered target counts
    integral = (
        np.pad(uncovered, ((1, 0), (1, 0)), mode="constant")
        .cumsum(axis=0)
        .cumsum(axis=1)
    )
    uncovered_counts = (
        integral[ph:, pw:]
        - integral[:-ph, pw:]
        - integral[ph:, :-pw]
        + integral[:-ph, :-pw]
    )
    # Track how many pixels in each candidate patch are already covered
    overlap_counts = np.zeros((valid_h, valid_w), dtype=np.int32)

    boxes = []

    pbar = tqdm(total=num_target, desc="Selecting patches", unit="pixel")

    # 4. Greedy loop with overlap awareness
    while True:
        # Score rewards new target pixels and slightly penalizes redundant overlap
        scores = uncovered_counts - (overlap_penalty * overlap_counts)

        # Mask out positions with zero uncovered target pixels
        valid_candidates = uncovered_counts > 0

        pbar.update(num_target - uncovered_counts[valid_candidates].sum())

        if not np.any(valid_candidates):
            break

        if max_overlap_ratio is not None:
            max_allowed_overlap = int(patch_area * max_overlap_ratio)
            non_overlapping = valid_candidates & (
                overlap_counts <= max_allowed_overlap
            )
            # Use non-overlapping candidates if available; otherwise relax cap so no target pixels get stranded
            active_mask = (
                non_overlapping if np.any(non_overlapping) else valid_candidates
            )
        else:
            active_mask = valid_candidates

        masked_scores = np.where(active_mask, scores, -np.inf)
        best_idx = int(np.argmax(masked_scores))
        best_y, best_x = divmod(best_idx, valid_w)

        y_max, x_max = best_y + ph, best_x + pw
        boxes.append((best_y, best_x, y_max, x_max))

        # Local update window
        upd_y0 = max(0, best_y - ph + 1)
        upd_x0 = max(0, best_x - pw + 1)
        upd_y1 = min(valid_h, y_max)
        upd_x1 = min(valid_w, x_max)

        uncovered[best_y:y_max, best_x:x_max] = 0
        covered_canvas[best_y:y_max, best_x:x_max] = 1

        # Update uncovered target counts locally
        sub_unc = uncovered[upd_y0 : upd_y1 + ph - 1, upd_x0 : upd_x1 + pw - 1]
        int_unc = (
            np.pad(sub_unc, ((1, 0), (1, 0)), mode="constant")
            .cumsum(axis=0)
            .cumsum(axis=1)
        )
        uncovered_counts[upd_y0:upd_y1, upd_x0:upd_x1] = (
            int_unc[ph:, pw:]
            - int_unc[:-ph, pw:]
            - int_unc[ph:, :-pw]
            + int_unc[:-ph, :-pw]
        )

        # Update overlap counts locally
        sub_cov = covered_canvas[
            upd_y0 : upd_y1 + ph - 1, upd_x0 : upd_x1 + pw - 1
        ]
        int_cov = (
            np.pad(sub_cov, ((1, 0), (1, 0)), mode="constant")
            .cumsum(axis=0)
            .cumsum(axis=1)
        )
        overlap_counts[upd_y0:upd_y1, upd_x0:upd_x1] = (
            int_cov[ph:, pw:]
            - int_cov[:-ph, pw:]
            - int_cov[ph:, :-pw]
            + int_cov[:-ph, :-pw]
        )

    covered_area_pct = float(
        (covered_canvas.astype(bool).sum() / total_pixels) * 100.0
    )

    if return_boxes:
        return covered_area_pct, boxes
    return covered_area_pct

import time

def visualize_patch_comparison(
    image: np.ndarray,
    top_percent: float,
    patch_size: int,
    border_margin: int = 16,
    downsample: int = 8,
    run_adaptive: bool = True,
):
    clean_img = np.nan_to_num(image, nan=0.0)
    if clean_img.ndim == 3:
        clean_img = clean_img.mean(axis=-1)

    grid_cov, grid_boxes = compute_fixed_grid_patch_coverage(
        clean_img, top_percent, patch_size, return_boxes=True
    )

    if run_adaptive:
        adap_cov, adap_boxes = compute_adaptive_patch_coverage(
            clean_img,
            top_percent,
            patch_size,
            border_margin=border_margin,
            return_boxes=True,
        )
        second_panel = (
            adap_boxes,
            f"Adaptive (margin={border_margin}): {len(adap_boxes)} patches ({adap_cov:.2f}% area)",
            "lime",
            True,  # show red target overlay
        )
    else:
        second_panel = (
            [],
            "Base Image (Uncertainty Map)",
            None,
            False,  # clean base image without red overlay
        )

    configs = [
        (
            grid_boxes,
            f"Fixed Grid: {len(grid_boxes)} patches ({grid_cov:.2f}% area)",
            "cyan",
            True,
        ),
        second_panel,
    ]

    # Compute target mask using the exact same > 0.0 guard as the patch functions
    if top_percent <= 0:
        target_mask = np.zeros_like(clean_img, dtype=bool)
    else:
        num_target = int(np.ceil(clean_img.size * (top_percent / 100.0)))
        thresh = np.partition(clean_img.ravel(), -num_target)[-num_target]
        target_mask = (
            (clean_img > 0.0) if thresh <= 0.0 else (clean_img >= thresh)
        )

    # Downsample for fast, safe plotting on large images
    H, W = clean_img.shape
    if downsample > 1:
        dh, dw = H // downsample, W // downsample
        img_disp = (
            clean_img[: dh * downsample, : dw * downsample]
            .reshape(dh, downsample, dw, downsample)
            .max(axis=(1, 3))
        )
        mask_disp = (
            target_mask[: dh * downsample, : dw * downsample]
            .reshape(dh, downsample, dw, downsample)
            .any(axis=(1, 3))
        )
    else:
        img_disp = clean_img
        mask_disp = target_mask

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, (boxes, title, color, show_overlay) in zip(axes, configs):
        ax.imshow(img_disp, cmap="gray", vmin=0, vmax=1)

        if show_overlay:
            overlay = np.zeros((*img_disp.shape, 4))
            overlay[mask_disp] = [1.0, 0.2, 0.2, 0.65]
            ax.imshow(overlay)

        for y_min, x_min, y_max, x_max in boxes:
            rect = patches.Rectangle(
                (x_min / downsample, y_min / downsample),
                (x_max - x_min) / downsample,
                (y_max - y_min) / downsample,
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


