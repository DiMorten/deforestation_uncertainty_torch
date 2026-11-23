import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Sequence, Dict, Literal, List, Tuple, Mapping, Any
from numpy.lib.stride_tricks import sliding_window_view

AggMethod = Literal["mean", "max", "mean_sq", "p90", "mass_top10"]
Box = Tuple[int, int, int, int]  # (y_min, x_min, y_max, x_max)


def _fast_f1(preds_flat: np.ndarray, gt_flat: np.ndarray, num_classes: int, is_binary: bool) -> float:
    """Fast confusion-matrix-free F1 calculation via bincount."""
    encoded = gt_flat.astype(np.int64) * num_classes + preds_flat.astype(np.int64)
    cm = np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    denom = 2 * tp + fp + fn
    f1_per_class = np.where(denom > 0, (2 * tp) / denom, 0.0)

    return float(f1_per_class[1]) if is_binary else float(np.mean(f1_per_class))


def _infer_classes(preds: np.ndarray, gt: np.ndarray, num_classes: int | None) -> Tuple[int, bool]:
    if num_classes is None:
        num_classes = int(max(gt.max(), preds.max())) + 1
    return num_classes, (num_classes == 2)


def evaluate_hybrid_f1(
    preds: np.ndarray,
    gt: np.ndarray,
    selection: np.ndarray | Sequence[Box],
    num_classes: int | None = None,
) -> Tuple[float, float]:
    """
    Substitutes `gt` values into `preds` at the selected region and computes full-image F1.

    Args:
        preds: (H, W) integer array of predicted class labels.
        gt: (H, W) integer array of ground truth class labels.
        selection: Either a (H, W) boolean mask of pixels to fix, OR a list of
                   (y_min, x_min, y_max, x_max) bounding boxes.
        num_classes: Number of classes (inferred if None).

    Returns:
        Tuple of (annotated_area_pct, full_image_f1).
    """
    assert preds.shape == gt.shape, "preds and gt shapes must match."
    n_cls, is_binary = _infer_classes(preds, gt, num_classes)

    if isinstance(selection, np.ndarray) and selection.dtype == bool:
        pixel_mask = selection.ravel()
    else:
        mask_2d = np.zeros(preds.shape, dtype=bool)
        for y_min, x_min, y_max, x_max in selection:
            mask_2d[y_min:y_max, x_min:x_max] = True
        pixel_mask = mask_2d.ravel()

    area_pct = float(pixel_mask.mean() * 100.0)
    hybrid_preds = np.where(pixel_mask, gt.ravel(), preds.ravel())
    f1 = _fast_f1(hybrid_preds, gt.ravel(), n_cls, is_binary)
    return area_pct, f1


def _rank_grid_patches(
    uncertainty: np.ndarray,
    patch_size: int,
    agg_method: AggMethod,
) -> Tuple[np.ndarray, int, int]:
    """
    Pads `uncertainty` into a fixed grid, scores each patch, and returns
    (ranked_flat_indices, n_rows, n_cols).
    """
    H, W = uncertainty.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    u_pad = np.pad(uncertainty, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)

    n_rows = u_pad.shape[0] // patch_size
    n_cols = u_pad.shape[1] // patch_size
    total_patches = n_rows * n_cols

    patches_arr = (
        u_pad.reshape(n_rows, patch_size, n_cols, patch_size)
        .swapaxes(1, 2)
        .reshape(total_patches, patch_size * patch_size)
    )

    if agg_method == "mean":
        scores = patches_arr.mean(axis=1)
    elif agg_method == "max":
        scores = patches_arr.max(axis=1)
    elif agg_method == "mean_sq":
        scores = np.mean(patches_arr ** 2, axis=1)
    elif agg_method == "p90":
        scores = np.percentile(patches_arr, 90, axis=1)
    elif agg_method == "mass_top10":
        global_p90 = np.quantile(uncertainty, 0.90)
        scores = np.maximum(patches_arr - global_p90, 0.0).sum(axis=1)
    else:
        raise ValueError(f"Unknown agg_method: {agg_method}")

    ranked_indices = np.argsort(scores)[::-1]
    return ranked_indices, n_rows, n_cols


def _indices_to_boxes(
    selected_indices: np.ndarray,
    n_cols: int,
    patch_size: int,
    H: int,
    W: int,
) -> List[Box]:
    """Converts flat grid indices into clipped (y_min, x_min, y_max, x_max) tuples."""
    rows, cols = np.divmod(selected_indices, n_cols)
    return [
        (
            int(r * patch_size),
            int(c * patch_size),
            min(int((r + 1) * patch_size), H),
            min(int((c + 1) * patch_size), W),
        )
        for r, c in zip(rows, cols)
    ]



def evaluate_pixel_oracle(
    uncertainty: np.ndarray,
    preds: np.ndarray,
    gt: np.ndarray,
    percentages: Sequence[float] = (0.0, 1.0, 3.0, 5.0, 10.0, 20.0),
    num_classes: int | None = None,
) -> Dict[str, np.ndarray]:
    """
    Colleague's baseline: replaces the top n% most uncertain pixels with ground truth
    and computes full-image F1.
    """
    assert uncertainty.shape == preds.shape == gt.shape, "Shapes must match."
    n_cls, _ = _infer_classes(preds, gt, num_classes)

    pcts = np.sort(np.asarray(percentages, dtype=np.float64))
    u_flat = uncertainty.ravel()

    quantiles = np.clip(100.0 - pcts, 0.0, 100.0)
    thresholds = np.quantile(u_flat, quantiles / 100.0)

    areas, f1s = [], []
    for pct, thresh in zip(pcts, thresholds):
        if pct <= 0.0:
            mask = np.zeros(uncertainty.shape, dtype=bool)
        elif pct >= 100.0:
            mask = np.ones(uncertainty.shape, dtype=bool)
        else:
            mask = uncertainty >= thresh

        area_pct, f1 = evaluate_hybrid_f1(preds, gt, mask, num_classes=n_cls)
        areas.append(area_pct)
        f1s.append(f1)

    return {
        "area_pct": np.array(areas),
        "f1_scores": np.array(f1s),
    }


def evaluate_patch_grid(
    uncertainty: np.ndarray,
    preds: np.ndarray,
    gt: np.ndarray,
    percentages: Sequence[float] = (0.0, 1.0, 3.0, 5.0, 10.0, 20.0),
    patch_size: int = 128,
    agg_method: AggMethod = "mean_sq",
    num_classes: int | None = None,
    return_boxes: bool = False,
) -> Dict[str, np.ndarray | Dict[float, List[Box]]]:
    """
    Divides the image into a fixed (patch_size x patch_size) grid, ranks patches
    by `agg_method`, replaces top n% patches with GT, and computes full-image F1.

    If `return_boxes=True`, includes 'boxes_by_pct': {pct: [(y_min, x_min, y_max, x_max), ...]}.
    """
    assert uncertainty.shape == preds.shape == gt.shape, "Shapes must match."
    H, W = uncertainty.shape
    n_cls, _ = _infer_classes(preds, gt, num_classes)

    ranked_indices, n_rows, n_cols = _rank_grid_patches(uncertainty, patch_size, agg_method)
    total_patches = n_rows * n_cols

    areas, f1s = [], []
    boxes_by_pct: Dict[float, List[Box]] = {}

    for pct in sorted(percentages):
        k_patches = int(np.clip(np.round((pct / 100.0) * total_patches), 0, total_patches))
        boxes = _indices_to_boxes(ranked_indices[:k_patches], n_cols, patch_size, H, W)

        area_pct, f1 = evaluate_hybrid_f1(preds, gt, boxes, num_classes=n_cls)
        areas.append(area_pct)
        f1s.append(f1)

        if return_boxes:
            boxes_by_pct[float(pct)] = boxes

    result: Dict[str, np.ndarray | Dict[float, List[Box]]] = {
        "area_pct": np.array(areas),
        "f1_scores": np.array(f1s),
    }
    if return_boxes:
        result["boxes_by_pct"] = boxes_by_pct
    return result


def evaluate_patch_nms(
    uncertainty: np.ndarray,
    preds: np.ndarray,
    gt: np.ndarray,
    percentages: Sequence[float] = (0.0, 1.0, 3.0, 5.0, 10.0, 20.0),
    patch_size: int = 128,
    stride: int = 32,
    agg_method: AggMethod = "mean_sq",
    num_classes: int | None = None,
    return_boxes: bool = False,
) -> Dict[str, np.ndarray | Dict[float, List[Box]]]:
    """
    Selects (patch_size x patch_size) patches using Adaptive Zero-Out NMS:
    picks the highest-scoring window, zeroes out its uncertainty values so
    overlapping candidates lose credit for already-covered pixels, and updates
    only the affected local neighborhood of candidate scores.

    Args:
        uncertainty: (H, W) float array of pixel uncertainties.
        preds: (H, W) integer array of predicted class labels.
        gt: (H, W) integer array of ground truth class labels.
        percentages: Target unique annotated area percentages (0 to 100).
        patch_size: Side length of square patches.
        stride: Step size between candidate window top-left corners.
        agg_method: Patch uncertainty aggregation rule ('mean', 'max', 'mean_sq', 'p90', 'mass_top10').
        num_classes: Number of classes (inferred if None).
        return_boxes: If True, includes 'boxes_by_pct' in the returned dict.

    Returns:
        Dict with 'area_pct', 'f1_scores', and optionally 'boxes_by_pct'.
    """
    assert uncertainty.shape == preds.shape == gt.shape, "Shapes must match."
    H, W = uncertainty.shape
    total_pixels = H * W
    n_cls, _ = _infer_classes(preds, gt, num_classes)

    # Pad right/bottom so candidate windows cover the entire image cleanly
    pad_h = (stride - (H - patch_size) % stride) % stride if H > patch_size else max(0, patch_size - H)
    pad_w = (stride - (W - patch_size) % stride) % stride if W > patch_size else max(0, patch_size - W)
    u_work = np.pad(uncertainty, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0).astype(np.float32, copy=True)

    # Precompute threshold if using mass_top10
    global_p90 = float(np.quantile(uncertainty, 0.90)) if agg_method == "mass_top10" else 0.0

    def _score_windows(windows: np.ndarray) -> np.ndarray:
        """Scores a (..., patch_size, patch_size) array of windows."""
        if agg_method == "mean":
            return windows.mean(axis=(-2, -1))
        elif agg_method == "max":
            return windows.max(axis=(-2, -1))
        elif agg_method == "mean_sq":
            return np.mean(windows ** 2, axis=(-2, -1))
        elif agg_method == "p90":
            return np.percentile(windows, 90, axis=(-2, -1))
        elif agg_method == "mass_top10":
            return np.maximum(windows - global_p90, 0.0).sum(axis=(-2, -1))
        else:
            raise ValueError(f"Unknown agg_method: {agg_method}")

    # Zero-copy strided view of all candidate windows: shape (n_rows, n_cols, patch_size, patch_size)
    all_windows = sliding_window_view(u_work, (patch_size, patch_size))[::stride, ::stride]
    n_rows, n_cols = all_windows.shape[:2]

    # Initial score grid (tiny: e.g. ~530 x 310 for stride=32)
    scores = _score_windows(all_windows).astype(np.float64)

    # Sort target percentages and handle 0% immediately if requested
    sorted_pcts = sorted(float(p) for p in percentages)
    areas: List[float] = []
    f1s: List[float] = []
    boxes_by_pct: Dict[float, List[Box]] = {}

    next_target_idx = 0
    while next_target_idx < len(sorted_pcts) and sorted_pcts[next_target_idx] <= 0.0:
        pct = sorted_pcts[next_target_idx]
        area_pct, f1 = evaluate_hybrid_f1(preds, gt, [], num_classes=n_cls)
        areas.append(area_pct)
        f1s.append(f1)
        if return_boxes:
            boxes_by_pct[pct] = []
        next_target_idx += 1

    covered_mask = np.zeros((H, W), dtype=bool)
    covered_pixels = 0
    selected_boxes: List[Box] = []
    max_candidates = n_rows * n_cols

    for _ in range(max_candidates):
        if next_target_idx >= len(sorted_pcts):
            break

        # Pick highest-scoring candidate window
        flat_idx = int(np.argmax(scores))
        best_score = scores.ravel()[flat_idx]
        r, c = divmod(flat_idx, n_cols)

        # Prevent re-selecting an already-picked exact grid cell if remaining uncertainty is 0
        if best_score <= 0.0 and covered_pixels >= total_pixels:
            break
        scores[r, c] = -1.0  # Mark exact (r, c) position as consumed

        y_min = r * stride
        x_min = c * stride
        y_max = min(y_min + patch_size, H)
        x_max = min(x_min + patch_size, W)

        # Count newly covered unique pixels
        newly_covered = int((~covered_mask[y_min:y_max, x_min:x_max]).sum())
        if newly_covered == 0:
            continue

        covered_mask[y_min:y_max, x_min:x_max] = True
        covered_pixels += newly_covered
        selected_boxes.append((y_min, x_min, y_max, x_max))

        # Zero out uncertainty inside the selected box so overlapping neighbors lose credit
        u_work[y_min:y_max, x_min:x_max] = 0.0

        # Recompute scores ONLY for candidate windows overlapping [y_min, y_max) x [x_min, x_max)
        r_start = max(0, (y_min - patch_size) // stride + 1)
        r_end = min(n_rows, (y_max - 1) // stride + 1)
        c_start = max(0, (x_min - patch_size) // stride + 1)
        c_end = min(n_cols, (x_max - 1) // stride + 1)

        if r_start < r_end and c_start < c_end:
            local_windows = all_windows[r_start:r_end, c_start:c_end]
            local_scores = _score_windows(local_windows)
            # Preserve consumed (-1.0) markers so exact positions aren't re-picked
            consumed_mask = scores[r_start:r_end, c_start:c_end] < 0
            scores[r_start:r_end, c_start:c_end] = np.where(consumed_mask, -1.0, local_scores)

        # Check if we reached one or more target percentages
        current_pct = (covered_pixels / total_pixels) * 100.0
        while next_target_idx < len(sorted_pcts) and current_pct >= sorted_pcts[next_target_idx]:
            target_pct = sorted_pcts[next_target_idx]
            area_pct, f1 = evaluate_hybrid_f1(preds, gt, covered_mask, num_classes=n_cls)
            areas.append(area_pct)
            f1s.append(f1)
            if return_boxes:
                boxes_by_pct[target_pct] = list(selected_boxes)
            next_target_idx += 1

    # Handle any remaining targets (e.g. 100%) if loop ended early
    while next_target_idx < len(sorted_pcts):
        target_pct = sorted_pcts[next_target_idx]
        area_pct, f1 = evaluate_hybrid_f1(preds, gt, covered_mask, num_classes=n_cls)
        areas.append(area_pct)
        f1s.append(f1)
        if return_boxes:
            boxes_by_pct[target_pct] = list(selected_boxes)
        next_target_idx += 1

    result: Dict[str, np.ndarray | Dict[float, List[Box]]] = {
        "area_pct": np.array(areas),
        "f1_scores": np.array(f1s),
    }
    if return_boxes:
        result["boxes_by_pct"] = boxes_by_pct
    return result


def compare_patch_selections(
    uncertainty: np.ndarray,
    results: Mapping[str, Dict[str, Any]],
    target_pct: float | Sequence[float] = 3.0,
    downsample: int = 8,
    ncols: int | None = None,
    colors: Sequence[str] | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Renders selected patches side-by-side on the downsampled uncertainty map.
    Strictly handles plotting; expects pre-computed results with `return_boxes=True`.

    Args:
        uncertainty: (H, W) float array.
        results: Dict mapping {label: eval_result_dict}, where each eval_result_dict
                 was generated with `return_boxes=True`.
        target_pct: A single percentage (e.g., 3.0) to compare across all methods in `results`,
                    OR a sequence of percentages (e.g., [1.0, 3.0, 5.0]) to plot progression
                    (if `results` has 1 method, plots all target_pcts for it; if `results` has N
                    methods and N target_pcts, pairs them 1-to-1).
        downsample: Stride factor for display image.
        ncols: Number of subplot columns. Defaults to total number of panels.
        colors: Optional sequence of box edge colors. Defaults to high-contrast bright colors.
        save_path: Optional file path to save the figure.
        show: Whether to call plt.show().
    """
    H, W = uncertainty.shape
    default_colors = ["cyan", "lime", "yellow", "magenta", "orange", "deepskyblue"]
    palette = colors if colors is not None else default_colors

    # Normalize (label, result_dict, pct) pairs to plot
    items = list(results.items())
    if isinstance(target_pct, (int, float)):
        plot_specs = [(label, res, float(target_pct)) for label, res in items]
    else:
        pct_list = [float(p) for p in target_pct]
        if len(items) == 1:
            # 1 method across multiple budgets (e.g., mean @ 1%, 3%, 5%)
            label, res = items[0]
            plot_specs = [(label, res, p) for p in pct_list]
        elif len(items) == len(pct_list):
            # 1-to-1 pairing of methods and budgets
            plot_specs = [(label, res, p) for (label, res), p in zip(items, pct_list)]
        else:
            raise ValueError(
                f"Cannot align {len(items)} methods with {len(pct_list)} target_pct values. "
                "Pass a single float, 1 method with multiple pcts, or matching lengths."
            )

    configs = []
    for idx, (label, res, pct) in enumerate(plot_specs):
        if "boxes_by_pct" not in res:
            raise KeyError(
                f"Result for '{label}' is missing 'boxes_by_pct'. "
                "Re-run evaluation with `return_boxes=True`."
            )

        boxes_dict: Dict[float, List[Box]] = res["boxes_by_pct"]
        available_pcts = np.array(list(boxes_dict.keys()), dtype=np.float64)

        # Match the closest evaluated target percentage
        closest_idx = int(np.argmin(np.abs(available_pcts - pct)))
        matched_key = list(boxes_dict.keys())[closest_idx]
        boxes = boxes_dict[matched_key]

        # Retrieve matching actual area and F1 score
        eval_idx = int(np.argmin(np.abs(np.asarray(res["area_pct"]) - matched_key)))
        actual_area = float(res["area_pct"][eval_idx])
        f1_pct = float(res["f1_scores"][eval_idx]) * 100.0

        color = palette[idx % len(palette)]
        title = (
            f"{label} @ {matched_key:.1f}% (real: {actual_area:.2f}%)\n"
            f"{len(boxes)} patches | F1: {f1_pct:.2f}%"
        )
        configs.append((boxes, title, color))

    if not configs:
        raise ValueError(
            "No valid results with 'boxes_by_pct' were passed to `compare_patch_selections`. "
            "Make sure `results` is non-empty and generated with `return_boxes=True`."
        )

    # Downsample background once
    img_disp = uncertainty[::downsample, ::downsample]
    u_min, u_max = float(img_disp.min()), float(img_disp.max())
    if u_max > u_min:
        img_disp = (img_disp - u_min) / (u_max - u_min)

    # Aspect-matched subplot grid
    n_plots = len(configs)
    ncols = ncols or n_plots
    nrows = int(np.ceil(n_plots / ncols))

    aspect_ratio = H / W
    subplot_width = 5.5
    subplot_height = subplot_width * aspect_ratio + 0.6

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(subplot_width * ncols, subplot_height * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, (boxes, title, color) in zip(axes_flat, configs):
        ax.imshow(img_disp, cmap="gray", vmin=0, vmax=1, aspect="equal")
        for y_min, x_min, y_max, x_max in boxes:
            rect = patches.Rectangle(
                (x_min / downsample, y_min / downsample),
                (x_max - x_min) / downsample,
                (y_max - y_min) / downsample,
                linewidth=1.1,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")

    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.90, wspace=0.03, hspace=0.20)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_f1_curves(
    results: Dict[str, np.ndarray] | Mapping[str, Dict[str, np.ndarray]],
    title: str = "Annotated Area vs. Full-Image F1 Score",
    gain_threshold: float | None = 0.5,
    scatter_symbol: List[str] | None = None,
    colors: List[str] | None = None,
    oracle_key: str | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plots one or multiple Area (%) vs. F1 Score (%) curves.

    Args:
        results: Either a single results dict {'area_pct': ..., 'f1_scores': ...}
                 OR a dict of {label: results_dict} to plot multiple curves.
        title: Plot title.
        gain_threshold: Marginal F1 gain (% per 1% area) below which a vertical
                        'diminishing returns' line is drawn for the primary/oracle curve.
        oracle_key: Name of the reference curve in `results` (styled as a dashed baseline
                    and used for the diminishing-returns knee). Defaults to the first entry.
        save_path: Optional file path to save the figure.
    """
    # Normalize input: wrap a single results dict into a {label: dict} mapping
    if "area_pct" in results and "f1_scores" in results:
        curves: Dict[str, Dict[str, np.ndarray]] = {
            "Pixel-Level Oracle (Top n% Uncertain)": results  # type: ignore[dict-item]
        }
    else:
        curves = dict(results)  # type: ignore[arg-type]

    if oracle_key is None:
        oracle_key = next(iter(curves))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    n_non_oracle = sum(1 for k in curves if not (k == oracle_key and len(curves) > 1))
    if n_non_oracle <= 10:
        cmap = plt.get_cmap("tab10")
        palette = [cmap(i) for i in range(n_non_oracle)]
    elif n_non_oracle <= 20:
        cmap = plt.get_cmap("tab20")
        palette = [cmap(i) for i in range(n_non_oracle)]
    else:
        # Evenly spaced hues around the color wheel for >20 curves
        cmap = plt.get_cmap("hsv")
        palette = [cmap(i / n_non_oracle) for i in range(n_non_oracle)]

    color_iter = iter(palette)

    all_areas = []
    all_f1s = []
    knee_x = None

    for idx, (label, res) in enumerate(curves.items()):
        areas = np.asarray(res["area_pct"], dtype=np.float64)
        f1 = np.asarray(res["f1_scores"], dtype=np.float64)
        if f1.max() <= 1.0:
            f1 = f1 * 100.0

        all_areas.append(areas)
        all_f1s.append(f1)

        is_oracle = (label == oracle_key) and (len(curves) > 1)
        if is_oracle:
            color = "#111111"
        elif colors is not None:
            color = colors[idx]
        else:
            color =  next(color_iter)
        marker = scatter_symbol[idx] if scatter_symbol is not None else "o"
        print(color)
        ax.plot(
            areas,
            f1,
            color=color,
            linestyle="--" if is_oracle else "-",
            marker=marker if len(areas) <= 15 else None,
            linewidth=2.3 if is_oracle else 2.0,
            markersize=5,
            label=label,
            zorder=4 if is_oracle else 3,
        )

        # Compute diminishing returns knee on the reference/first curve
        if label == oracle_key and gain_threshold is not None and len(areas) > 1:
            marginal_gain = np.diff(f1) / np.diff(areas)
            below_thresh = np.where(marginal_gain < gain_threshold)[0]
            if len(below_thresh) > 0:
                knee_idx = below_thresh[0]
                knee_x = areas[knee_idx]
                knee_y = f1[knee_idx]
                ax.axvline(
                    knee_x,
                    color="#d62728",
                    linestyle=":",
                    alpha=0.8,
                    label=f"Diminishing returns (<{gain_threshold}% F1 / 1% area) at {knee_x:.1f}%",
                )
                ax.scatter([knee_x], [knee_y], color="#d62728", s=80, zorder=5)

        # Annotate key points (3% mark and endpoints when plotting multiple curves to avoid clutter)
        idx_3pct = int(np.argmin(np.abs(areas - 3.0)))
        if np.abs(areas[idx_3pct] - 3.0) > 0.5:
            idx_3pct = -1  # Skip if curve doesn't cover the 3% region at all

        # Find the single index closest to the knee point (if present)
        idx_knee = int(np.argmin(np.abs(areas - knee_x))) if knee_x is not None else -1

        for i, (x, y) in enumerate(zip(areas, f1)):
            is_3pct = (i == idx_3pct)
            is_endpoint = (i == len(areas) - 1) or (i == 0 and len(curves) == 1)
            is_knee = (i == idx_knee)
            break
            if len(curves) == 1 and len(areas) < 15:
                should_label = True
            else:
                should_label = is_3pct or is_endpoint or is_knee

            if should_label:
                offset_y = 8 if idx % 2 == 0 else -13
                ax.annotate(
                    f"{y:.2f}%",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, offset_y),
                    ha="center",
                    fontsize=8.5,
                    fontweight="bold" if is_3pct else "normal",
                    color=color,
                )
                ax.scatter([x], [y], color=color, s=65, zorder=5)

    # Reference vertical line at 3%
    ax.axvline(
        3.0,
        color="#555555",
        linestyle="-.",
        alpha=0.65,
        label="3% do artigo original",
    )

    # Formatting
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel("Audited Image Area (%)", fontsize=11)
    ax.set_ylabel("Full-Image F1 Score (%)", fontsize=11)

    concat_areas = np.concatenate(all_areas)
    concat_f1s = np.concatenate(all_f1s)
    y_margin = (concat_f1s.max() - concat_f1s.min()) * 0.15
    ax.set_ylim(max(0, concat_f1s.min() - y_margin), min(100.5, concat_f1s.max() + y_margin))
    ax.set_xlim(-0.3, concat_areas.max() + 0.8)

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()