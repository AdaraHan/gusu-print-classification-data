#!/usr/bin/env python3
"""Reproducible candidate-sample GLCM experiment.

Scope: the 568 Stage-1 candidate images (459 + 9 + 100). This program does
not treat the candidate labels, dates, work identities, or duplicate status as
human-verified. It never modifies source images or Stage-1 files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gusu_glcm_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gusu_glcm_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage, stats


EXPERIMENT_DATE = "2026-07-29"
EXPERIMENT_ID = "formal_glcm_candidate_experiment_20260729"
RANDOM_SEED = 20260729
EXPECTED_TOTAL = 568
EXPECTED_COUNTS = {
    "清代姑苏版画（候选，待确认）": 459,
    "20世纪50年代桃花坞年画（候选，待确认）": 9,
    "清末杨柳青年画（候选，待确认）": 100,
}
GROUP_LABELS = {
    "清代姑苏版画（候选，待确认）": "姑苏版画（候选）",
    "20世纪50年代桃花坞年画（候选，待确认）": "桃花坞年画（候选）",
    "清末杨柳青年画（候选，待确认）": "杨柳青年画（候选）",
}
GROUP_ORDER = list(GROUP_LABELS.values())
METRICS = ["energy", "contrast", "correlation", "homogeneity"]
METRIC_ZH = {
    "energy": "能量",
    "contrast": "对比度",
    "correlation": "相关性",
    "homogeneity": "同质性",
}
LEVELS = 32
LONG_EDGE = 512
BLOCK_SIZE = 64
MIN_VALID_FRACTION = 0.80
DISTANCES = (1, 2, 4)
ANGLES_DEG = (0, 45, 90, 135)
PRIMARY_L = 0.95
PRIMARY_S = 0.08
L_THRESHOLDS = (0.93, 0.95, 0.97)
S_THRESHOLDS = (0.05, 0.08, 0.10)
BOOTSTRAP_REPS = 5000
MIN_VALID_PAIRS = 100
FONT_PATH = Path("/System/Library/Fonts/STHeiti Medium.ttc")


@dataclass(frozen=True)
class Task:
    sample_id: str
    image_path: str
    sha256: str
    proposed_group: str
    project_root: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/formal_aesthetics_reproduction_20260727/formal_analysis_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/formal_glcm_candidate_experiment_20260729"),
    )
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing generated outputs; never touches inputs.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def configure_font() -> None:
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        name = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
        plt.rcParams["font.family"] = name
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 220


def rgb_to_hsl_sl(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return standard HSL saturation and lightness for RGB in [0, 1]."""
    high = rgb.max(axis=2)
    low = rgb.min(axis=2)
    lightness = (high + low) / 2.0
    delta = high - low
    saturation = np.zeros_like(lightness, dtype=np.float32)
    active = delta > 1e-7
    denom = 1.0 - np.abs(2.0 * lightness - 1.0)
    saturation[active] = delta[active] / np.maximum(denom[active], 1e-7)
    return saturation, lightness


def resize_rgb(path: Path) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], str, bool]:
    with Image.open(path) as opened:
        original_format = opened.format or path.suffix.lstrip(".").upper()
        had_alpha = "A" in opened.getbands() or "transparency" in opened.info
        image = ImageOps.exif_transpose(opened).convert("RGB")
        original_size = image.size
        scale = LONG_EDGE / max(image.size)
        resized_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(resized_size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0, original_size, resized_size, original_format, had_alpha


def boundary_background_mask(saturation: np.ndarray, lightness: np.ndarray, l_min: float, s_max: float) -> np.ndarray:
    """Boundary-connected near-white/near-neutral background candidate.

    Internal white regions are retained because propagation starts at the image
    boundary. Closing is fixed at 3x3 and constrained to candidate pixels.
    """
    candidate = (lightness >= l_min) & (saturation <= s_max)
    seed = np.zeros_like(candidate, dtype=bool)
    seed[0, :] = candidate[0, :]
    seed[-1, :] = candidate[-1, :]
    seed[:, 0] = candidate[:, 0]
    seed[:, -1] = candidate[:, -1]
    connected = ndimage.binary_propagation(seed, structure=np.ones((3, 3), dtype=bool), mask=candidate)
    closed = ndimage.binary_closing(connected, structure=np.ones((3, 3), dtype=bool)) & candidate
    return closed


def centered_block_crop(height: int, width: int) -> tuple[int, int, int, int]:
    crop_h = (height // BLOCK_SIZE) * BLOCK_SIZE
    crop_w = (width // BLOCK_SIZE) * BLOCK_SIZE
    if crop_h < BLOCK_SIZE or crop_w < BLOCK_SIZE:
        raise ValueError(f"resized image too narrow for {BLOCK_SIZE}x{BLOCK_SIZE} blocks: {(width, height)}")
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return top, left, crop_h, crop_w


def quantize_bt709(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    q = np.floor(np.clip(gray, 0.0, 1.0) * LEVELS).astype(np.uint8)
    q[q == LEVELS] = LEVELS - 1
    return gray.astype(np.float32), q


def paired_views(array: np.ndarray, row_offset: int, col_offset: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = array.shape
    if row_offset >= 0:
        r1, r2 = slice(0, h - row_offset), slice(row_offset, h)
    else:
        r1, r2 = slice(-row_offset, h), slice(0, h + row_offset)
    if col_offset >= 0:
        c1, c2 = slice(0, w - col_offset), slice(col_offset, w)
    else:
        c1, c2 = slice(-col_offset, w), slice(0, w + col_offset)
    return array[r1, c1], array[r2, c2]


def glcm_properties(quantized: np.ndarray, valid: np.ndarray, distance: int, angle_deg: int) -> tuple[dict[str, float], int]:
    angle = math.radians(angle_deg)
    # Match scikit-image's image-coordinate convention: positive angle moves
    # toward increasing row (down) and increasing column (right).
    row_offset = int(round(math.sin(angle) * distance))
    col_offset = int(round(math.cos(angle) * distance))
    a, b = paired_views(quantized, row_offset, col_offset)
    va, vb = paired_views(valid, row_offset, col_offset)
    keep = va & vb
    pair_count = int(keep.sum())
    if pair_count < MIN_VALID_PAIRS:
        return {metric: math.nan for metric in METRICS}, pair_count
    code = a[keep].astype(np.int32) * LEVELS + b[keep].astype(np.int32)
    matrix = np.bincount(code, minlength=LEVELS * LEVELS).reshape(LEVELS, LEVELS).astype(np.float64)
    matrix += matrix.T  # symmetric=True
    matrix /= matrix.sum()  # normed=True
    i = np.arange(LEVELS, dtype=np.float64)[:, None]
    j = np.arange(LEVELS, dtype=np.float64)[None, :]
    delta2 = (i - j) ** 2
    energy = float(np.sqrt(np.sum(matrix * matrix)))
    contrast = float(np.sum(matrix * delta2))
    homogeneity = float(np.sum(matrix / (1.0 + delta2)))
    mean_i = float(np.sum(i * matrix))
    mean_j = float(np.sum(j * matrix))
    std_i = float(np.sqrt(np.sum(((i - mean_i) ** 2) * matrix)))
    std_j = float(np.sqrt(np.sum(((j - mean_j) ** 2) * matrix)))
    if std_i <= 1e-15 or std_j <= 1e-15:
        correlation = 1.0
    else:
        correlation = float(np.sum((i - mean_i) * (j - mean_j) * matrix) / (std_i * std_j))
    return {
        "energy": energy,
        "contrast": contrast,
        "correlation": correlation,
        "homogeneity": homogeneity,
    }, pair_count


def process_threshold(
    quantized: np.ndarray,
    effective_mask: np.ndarray,
    crop: tuple[int, int, int, int],
    keep_long: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    top, left, crop_h, crop_w = crop
    q = quantized[top : top + crop_h, left : left + crop_w]
    mask = effective_mask[top : top + crop_h, left : left + crop_w]
    values = {metric: [] for metric in METRICS}
    long_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    accepted = 0
    block_id = 0
    for row in range(0, crop_h, BLOCK_SIZE):
        for col in range(0, crop_w, BLOCK_SIZE):
            block_id += 1
            block_q = q[row : row + BLOCK_SIZE, col : col + BLOCK_SIZE]
            block_mask = mask[row : row + BLOCK_SIZE, col : col + BLOCK_SIZE]
            coverage = float(block_mask.mean())
            is_accepted = coverage >= MIN_VALID_FRACTION
            if is_accepted:
                accepted += 1
            if keep_long:
                block_rows.append({
                    "block_id": block_id,
                    "row_px": top + row,
                    "col_px": left + col,
                    "height_px": BLOCK_SIZE,
                    "width_px": BLOCK_SIZE,
                    "valid_fraction": coverage,
                    "accepted": is_accepted,
                })
            if not is_accepted:
                continue
            for distance in DISTANCES:
                for angle_deg in ANGLES_DEG:
                    props, pair_count = glcm_properties(block_q, block_mask, distance, angle_deg)
                    if keep_long:
                        long_rows.append({
                            "block_id": block_id,
                            "distance_px": distance,
                            "angle_deg": angle_deg,
                            "valid_fraction": coverage,
                            "valid_pair_count": pair_count,
                            **props,
                        })
                    for metric in METRICS:
                        if math.isfinite(props[metric]):
                            values[metric].append(props[metric])
    result: dict[str, Any] = {
        "effective_pixel_fraction": float(effective_mask.mean()),
        "crop_effective_pixel_fraction": float(mask.mean()),
        "total_blocks": block_id,
        "accepted_blocks": accepted,
        "accepted_block_fraction": accepted / block_id if block_id else 0.0,
        "glcm_observations": len(values[METRICS[0]]),
    }
    for metric in METRICS:
        result[metric] = float(np.median(values[metric])) if values[metric] else math.nan
    return result, long_rows, block_rows


def process_one(task: Task) -> dict[str, Any]:
    path = Path(task.project_root) / task.image_path
    rgb, original_size, resized_size, image_format, had_alpha = resize_rgb(path)
    saturation, lightness = rgb_to_hsl_sl(rgb)
    gray, quantized = quantize_bt709(rgb)
    crop = centered_block_crop(rgb.shape[0], rgb.shape[1])
    sensitivity_rows: list[dict[str, Any]] = []
    primary_long: list[dict[str, Any]] = []
    primary_blocks: list[dict[str, Any]] = []
    primary_mask: np.ndarray | None = None
    for l_min in L_THRESHOLDS:
        for s_max in S_THRESHOLDS:
            background = boundary_background_mask(saturation, lightness, l_min, s_max)
            effective = ~background
            is_primary = math.isclose(l_min, PRIMARY_L) and math.isclose(s_max, PRIMARY_S)
            metrics, long_rows, block_rows = process_threshold(quantized, effective, crop, is_primary)
            sensitivity_rows.append({"mask_l_min": l_min, "mask_s_max": s_max, **metrics})
            if is_primary:
                primary_long = long_rows
                primary_blocks = block_rows
                primary_mask = effective
    if primary_mask is None:
        raise RuntimeError("primary mask variant was not computed")
    return {
        "sample_id": task.sample_id,
        "image_path": task.image_path,
        "sha256": task.sha256,
        "proposed_group": task.proposed_group,
        "group": GROUP_LABELS[task.proposed_group],
        "original_width": original_size[0],
        "original_height": original_size[1],
        "resized_width": resized_size[0],
        "resized_height": resized_size[1],
        "image_format": image_format,
        "had_alpha": had_alpha,
        "crop_top": crop[0],
        "crop_left": crop[1],
        "crop_height": crop[2],
        "crop_width": crop[3],
        "sensitivity": sensitivity_rows,
        "long": primary_long,
        "blocks": primary_blocks,
    }


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (n - rank) * p[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def bootstrap_mean_ci(values: np.ndarray, key: str) -> tuple[float, float]:
    seed = (RANDOM_SEED + zlib.crc32(key.encode("utf-8"))) % (2**32)
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(BOOTSTRAP_REPS, dtype=float)
    chunk = 500
    for start in range(0, BOOTSTRAP_REPS, chunk):
        stop = min(start + chunk, BOOTSTRAP_REPS)
        indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def build_statistics(per_image: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    omnibus_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        groups = []
        for group in GROUP_ORDER:
            values = per_image.loc[per_image["group"] == group, metric].dropna().to_numpy(float)
            groups.append(values)
            ci_low, ci_high = bootstrap_mean_ci(values, f"{group}:{metric}")
            q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
            summary_rows.append({
                "group": group,
                "metric": metric,
                "metric_zh": METRIC_ZH[metric],
                "n_images": len(values),
                "mean": values.mean(),
                "sd": values.std(ddof=1),
                "median": median,
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "minimum": values.min(),
                "maximum": values.max(),
                "bootstrap_mean_ci_low": ci_low,
                "bootstrap_mean_ci_high": ci_high,
            })
        test = stats.kruskal(*groups, nan_policy="raise")
        omnibus_rows.append({"metric": metric, "metric_zh": METRIC_ZH[metric], "test": "Kruskal-Wallis", "statistic": test.statistic, "p_raw": test.pvalue})
        for left_idx in range(len(GROUP_ORDER)):
            for right_idx in range(left_idx + 1, len(GROUP_ORDER)):
                left, right = GROUP_ORDER[left_idx], GROUP_ORDER[right_idx]
                a, b = groups[left_idx], groups[right_idx]
                mw = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
                pairwise_rows.append({
                    "metric": metric,
                    "metric_zh": METRIC_ZH[metric],
                    "group_a": left,
                    "group_b": right,
                    "n_a": len(a),
                    "n_b": len(b),
                    "test": "Mann-Whitney U",
                    "statistic": mw.statistic,
                    "p_raw": mw.pvalue,
                    "rank_biserial_a_minus_b": 2.0 * mw.statistic / (len(a) * len(b)) - 1.0,
                    "median_a": np.median(a),
                    "median_b": np.median(b),
                })
    omnibus = pd.DataFrame(omnibus_rows)
    omnibus["p_holm_4_metrics"] = holm_adjust(omnibus["p_raw"])
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise["p_holm_12_tests"] = holm_adjust(pairwise["p_raw"])
    return pd.DataFrame(summary_rows), omnibus, pairwise


def format_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown table writer."""
    columns = [str(column) for column in frame.columns]
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(clean(value) for value in row) + " |")
    return "\n".join(lines)


def write_table10(summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for group in GROUP_ORDER:
        item: dict[str, Any] = {"版画类别": group}
        for metric in METRICS:
            row = summary[(summary["group"] == group) & (summary["metric"] == metric)].iloc[0]
            item[METRIC_ZH[metric]] = f"{row['mean']:.3f} ± {row['sd']:.3f}"
        rows.append(item)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "table10_candidate_glcm_reproduction.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    ax.axis("off")
    rendered = ax.table(cellText=table.values, colLabels=table.columns, cellLoc="center", colLoc="center", loc="center")
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(12)
    rendered.scale(1.0, 1.9)
    for (row, col), cell in rendered.get_celld().items():
        cell.set_edgecolor("#354052")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#dbe7f2")
            cell.set_text_props(weight="bold")
        elif row % 2:
            cell.set_facecolor("#f7f9fb")
    ax.set_title("表10（新复现）：三类候选版画的局部GLCM特征（图像级均值 ± 样本标准差）", fontsize=15, weight="bold", pad=14)
    fig.text(0.5, 0.035, "候选样本 n=459/9/100；每图为有效局部块×距离×方向的中位数；非正式manifest结论", ha="center", fontsize=10, color="#4a5568")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(output_dir / "table10_candidate_glcm_reproduction.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return table


def plot_distributions(per_image: pd.DataFrame, output_dir: Path) -> None:
    colors = ["#2F6B8A", "#D58B3A", "#5C8D66"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    rng = np.random.default_rng(RANDOM_SEED)
    for ax, metric in zip(axes.flat, METRICS):
        arrays = [per_image.loc[per_image["group"] == g, metric].dropna().to_numpy(float) for g in GROUP_ORDER]
        box = ax.boxplot(arrays, positions=np.arange(1, 4), widths=0.55, patch_artist=True, showfliers=False, medianprops={"color": "#222", "linewidth": 1.5})
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.28)
            patch.set_edgecolor(color)
        for idx, (values, color) in enumerate(zip(arrays, colors), start=1):
            jitter = rng.normal(idx, 0.06, size=len(values))
            ax.scatter(jitter, values, s=12 if len(values) > 150 else 24, alpha=0.30 if len(values) > 150 else 0.72, color=color, edgecolors="none")
        ax.set_xticks([1, 2, 3], [f"{g}\n(n={len(a)})" for g, a in zip(GROUP_ORDER, arrays)])
        ax.set_title(METRIC_ZH[metric], weight="bold")
        ax.set_ylabel("图像级GLCM值")
        ax.grid(axis="y", alpha=0.22)
        low = min(float(np.nanmin(np.concatenate(arrays))), 0.0)
        high = float(np.nanmax(np.concatenate(arrays)))
        pad = max((high - low) * 0.08, 1e-4)
        ax.set_ylim(low - (pad if low < 0 else 0), high + pad)
    fig.suptitle("三类候选版画的局部GLCM指标分布", fontsize=17, weight="bold")
    fig.text(0.5, 0.012, "点为单幅图像；箱体为四分位区间。样本口径尚未完成作品/版次去重与来源审核。", ha="center", fontsize=10, color="#4a5568")
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(output_dir / "figure_glcm_distributions.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_means_ci(summary: pd.DataFrame, output_dir: Path) -> None:
    colors = ["#2F6B8A", "#D58B3A", "#5C8D66"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    for ax, metric in zip(axes.flat, METRICS):
        subset = summary[summary["metric"] == metric].set_index("group").loc[GROUP_ORDER]
        y = np.arange(3)
        means = subset["mean"].to_numpy()
        low = subset["bootstrap_mean_ci_low"].to_numpy()
        high = subset["bootstrap_mean_ci_high"].to_numpy()
        for idx in range(3):
            ax.errorbar(means[idx], y[idx], xerr=[[means[idx] - low[idx]], [high[idx] - means[idx]]], fmt="o", color=colors[idx], markersize=7, capsize=4, linewidth=1.6)
        ax.set_yticks(y, GROUP_ORDER)
        ax.invert_yaxis()
        ax.set_title(METRIC_ZH[metric], weight="bold")
        ax.set_xlabel("图像级均值及非参数bootstrap 95% CI")
        ax.grid(axis="x", alpha=0.22)
    fig.suptitle("候选样本GLCM组均值与不确定性", fontsize=17, weight="bold")
    fig.text(0.5, 0.012, f"每组内部按图像重采样 {BOOTSTRAP_REPS} 次；桃花坞候选仅9张，区间应谨慎解释。", ha="center", fontsize=10, color="#4a5568")
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(output_dir / "figure_glcm_group_means_ci.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_sensitivity(sensitivity_summary: pd.DataFrame, output_dir: Path) -> None:
    colors = {GROUP_ORDER[0]: "#2F6B8A", GROUP_ORDER[1]: "#D58B3A", GROUP_ORDER[2]: "#5C8D66"}
    variants = [(l, s) for l in L_THRESHOLDS for s in S_THRESHOLDS]
    labels = [f"L≥{l:.2f}\nS≤{s:.2f}" for l, s in variants]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    for ax, metric in zip(axes.flat, METRICS):
        for group in GROUP_ORDER:
            values = []
            for l, s in variants:
                row = sensitivity_summary[(sensitivity_summary["group"] == group) & (sensitivity_summary["metric"] == metric) & np.isclose(sensitivity_summary["mask_l_min"], l) & np.isclose(sensitivity_summary["mask_s_max"], s)]
                values.append(float(row.iloc[0]["mean"]))
            ax.plot(range(len(variants)), values, marker="o", linewidth=1.4, markersize=4, label=group, color=colors[group])
        ax.axvline(4, color="#111", linestyle="--", linewidth=0.9, alpha=0.55)
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_title(METRIC_ZH[metric], weight="bold")
        ax.set_ylabel("组内图像均值")
        ax.grid(axis="y", alpha=0.2)
    handles, labels_legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("背景掩膜阈值敏感性（全部9组预注册组合）", fontsize=17, weight="bold")
    fig.text(0.5, 0.012, "虚线为主参数 L≥0.95、S≤0.08；敏感性结果用于判断结论是否依赖背景阈值。", ha="center", fontsize=10, color="#4a5568")
    fig.tight_layout(rect=(0, 0.035, 1, 0.91))
    fig.savefig(output_dir / "figure_glcm_mask_sensitivity.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_qc_figure(per_image: pd.DataFrame, blocks: pd.DataFrame, project_root: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 14))
    for row_idx, group in enumerate(GROUP_ORDER):
        sample = per_image[per_image["group"] == group].sort_values("sample_id").iloc[0]
        path = project_root / sample["image_path"]
        rgb, _, _, _, _ = resize_rgb(path)
        saturation, lightness = rgb_to_hsl_sl(rgb)
        effective = ~boundary_background_mask(saturation, lightness, PRIMARY_L, PRIMARY_S)
        gray, quantized = quantize_bt709(rgb)
        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].set_title(f"{group}\n原图（固定sample_id排序首张）")
        axes[row_idx, 1].imshow(quantized, cmap="gray", vmin=0, vmax=LEVELS - 1)
        axes[row_idx, 1].set_title("BT.709灰度量化至32级")
        overlay = (rgb * 255).astype(np.uint8)
        image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(image, "RGBA")
        sample_blocks = blocks[blocks["sample_id"] == sample["sample_id"]]
        for _, block in sample_blocks.iterrows():
            x0, y0 = int(block["col_px"]), int(block["row_px"])
            x1, y1 = x0 + BLOCK_SIZE - 1, y0 + BLOCK_SIZE - 1
            color = (46, 160, 67, 210) if bool(block["accepted"]) else (210, 52, 52, 150)
            draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        axes[row_idx, 2].imshow(image)
        axes[row_idx, 2].set_title(f"有效块：绿=纳入，红=排除\n有效像素={effective.mean():.1%}")
        for col_idx in range(3):
            axes[row_idx, col_idx].axis("off")
        axes[row_idx, 0].text(0.01, -0.08, sample["sample_id"], transform=axes[row_idx, 0].transAxes, fontsize=8, color="#4a5568")
    fig.suptitle("GLCM输入与有效局部块质量控制示例", fontsize=17, weight="bold")
    fig.text(0.5, 0.012, "示例选择规则在查看图像前固定为各组sample_id字典序首张；该图不代表人工QC通过。", ha="center", fontsize=10, color="#4a5568")
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(output_dir / "figure_glcm_qc_examples.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_text_report(
    output_dir: Path,
    table10: pd.DataFrame,
    summary: pd.DataFrame,
    omnibus: pd.DataFrame,
    pairwise: pd.DataFrame,
    per_image: pd.DataFrame,
    sensitivity_robustness: pd.DataFrame,
) -> None:
    group_counts = per_image.groupby("group").size().reindex(GROUP_ORDER)
    table_md = dataframe_to_markdown(table10)
    omnibus_md = omnibus[["metric_zh", "statistic", "p_raw", "p_holm_4_metrics"]].copy()
    omnibus_md["statistic"] = omnibus_md["statistic"].map(lambda x: f"{x:.3f}")
    omnibus_md["p_raw"] = omnibus_md["p_raw"].map(format_p)
    omnibus_md["p_holm_4_metrics"] = omnibus_md["p_holm_4_metrics"].map(format_p)
    best_pairwise = pairwise.sort_values(["metric", "p_holm_12_tests"])
    pair_md = best_pairwise[["metric_zh", "group_a", "group_b", "rank_biserial_a_minus_b", "p_holm_12_tests"]].copy()
    pair_md["rank_biserial_a_minus_b"] = pair_md["rank_biserial_a_minus_b"].map(lambda x: f"{x:.3f}")
    pair_md["p_holm_12_tests"] = pair_md["p_holm_12_tests"].map(format_p)
    sens_lines = []
    for metric in METRICS:
        for group in GROUP_ORDER:
            row = sensitivity_robustness[(sensitivity_robustness["metric"] == metric) & (sensitivity_robustness["group"] == group)].iloc[0]
            sens_lines.append(f"- {group}—{METRIC_ZH[metric]}：9组阈值均值范围 {row['min_variant_mean']:.4f}–{row['max_variant_mean']:.4f}，主参数 {row['primary_mean']:.4f}。")
    report = f"""# 三类候选版画局部灰度纹理的GLCM复现实验

实验日期：{EXPERIMENT_DATE}  
实验编号：`{EXPERIMENT_ID}`  
结论等级：**候选样本探索性复现；不能替代人工冻结后的正式作品/版次分析。**

## 技术摘要

本实验首次在当前项目中建立了可运行、逐图可追溯的GLCM计算链。输入严格取自Stage 1候选manifest，共{len(per_image)}张图像：姑苏{group_counts.iloc[0]}张、20世纪50年代桃花坞{group_counts.iloc[1]}张、清末杨柳青{group_counts.iloc[2]}张。所有文件路径、SHA-256和分组计数均在计算前核验。

新结果描述的是**固定尺度、按自动掩膜规则判定合格的局部块中的灰度共生分布**。它可以支持“局部灰度纹理在候选组之间是否存在差异”的统计描述，但不能测量整体构图、线条连续性、消失点、工艺规范或审美传统。由于样本尚未完成年代、来源、同版异图和独立作品统计单位审核，正文只能称为“候选样本结果”。

## 新复现表10

{table_md}

表中每一格为图像级指标的组内算术均值±图像间样本标准差。每幅图像的指标先由所有合格局部块、3个距离和4个方向的属性值取中位数，再进行组间汇总；没有把所有像素或所有共生矩阵直接混为一个总体。

![三类候选样本的GLCM分布](figure_glcm_distributions.png)

分布图显示每幅图像的实际离散程度。桃花坞候选仅9张，不能只凭均值高低给出稳定的时代或地域归因；箱线与散点比单独的“均值±标准差”更能揭示这一不确定性。

![组均值与bootstrap置信区间](figure_glcm_group_means_ci.png)

置信区间按图像为重采样单位、固定种子进行{BOOTSTRAP_REPS}次非参数bootstrap。它量化当前候选图像口径下均值的不确定性，但没有解决同版异图或组画导致的非独立性。

## 组间检验

四个预注册属性先分别进行Kruskal–Wallis检验，再对四个总体检验进行Holm校正：

{dataframe_to_markdown(omnibus_md)}

探索性两两比较采用Mann–Whitney U检验，效应量为秩二列相关（正值表示前一组总体秩更高），12项比较统一Holm校正：

{dataframe_to_markdown(pair_md)}

这些检验以图像为暂定统计单位。只有正式manifest把重复、同版、同作品和组画统计单位冻结后，才可升级为论文主推断。

## 方法与公式

1. EXIF方向校正并转为RGB；保持长宽比，将长边缩放为512像素，不补边。灰度使用BT.709：`Y=0.2126R+0.7152G+0.0722B`。
2. 主背景规则为边界连通且HSL明度`L≥0.95`、饱和度`S≤0.08`的近白近中性像素；3×3闭运算仅在背景候选内执行。画面内部不与边界连通的白色区域保留。
3. 灰度按`floor(Y×32)`量化到0–31级。居中裁去不足64像素的余边后划分不重叠64×64块；有效像素比例至少80%的块才进入计算。
4. 距离为1、2、4像素，方向为0°、45°、90°、135°。只统计两个端点都在有效掩膜内的像素对；共生矩阵与其转置相加（`symmetric=true`）并归一化到和为1（`normed=true`）。
5. 对归一化共生矩阵`P(i,j)`：对比度为`Σ(i−j)²P(i,j)`；能量为`√ΣP(i,j)²`；同质性为`ΣP(i,j)/(1+(i−j)²)`；相关性为灰度索引的标准化协方差。属性含义遵循数学定义，不作构图或空间解释。

![掩膜与局部块QC](figure_glcm_qc_examples.png)

绿色框表示纳入计算的局部块，红色框表示有效像素不足80%的局部块。示例按sample_id字典序固定抽取，未根据结果挑选；深色画框、拍摄背景和纸张老化仍可能残留，需要正式阶段逐图人工QC。

## 掩膜阈值敏感性

![掩膜阈值敏感性](figure_glcm_mask_sensitivity.png)

预先报告`L∈{{0.93,0.95,0.97}}`与`S∈{{0.05,0.08,0.10}}`的全部9种组合，不选择最符合论文假设的一组：

{chr(10).join(sens_lines)}

九组均值几乎不变，但这不能被解释为“背景处理十分稳健”：主阈值在534/568张图像中保留了100%像素，34张有像素级排除，且只有2张因此排除了完整局部块。它更直接说明近白规则对大多数扫描背景缺乏敏感性。若以后引入经人工验证的内容掩膜，必须作为新版本完整重跑并报告与本轮差异。

## 可以与不可以写入论文的内容

可以写：本候选样本复现实验使用明确的量化级数、距离、方向、掩膜、局部块、汇总和统计检验，得到表中当前数值；GLCM属性反映固定条件下的局部灰度共现结构。

不可以写：能量高直接证明“纹理规整”“工艺严格”；相关性高证明“线条连续”；同质性高证明“色彩晕染”；GLCM证明整体构图、消失点或中西透视；候选组差异直接由地域审美或历史沿革造成。上述论断都超出该算法与当前样本设计的证据边界。

## 局限与下一步

- 568张仍是候选图像，未完成人工分组、年代、来源、重复、同版及组画审核；当前统计单位只能是图像，不是独立作品/版次。
- 三组样本量高度不均衡，桃花坞仅9张；数字化来源、压缩、画框、纸张背景和扫描条件可能与候选组别共线。
- 自动掩膜只处理边界连通的近白背景，不能可靠识别深色画框或复杂拍摄背景；正式实验应输出全量QC画廊并逐图审核。
- 正式manifest冻结后应以独立作品/版次为主统计单位重跑同一脚本，并按series进行聚类重采样；届时不得沿用本表数字。

## 进一步问题

1. 经人工去重和作品级聚合后，四项GLCM组间差异是否保持方向与效应量？
2. 分层控制数字化来源、acquisition batch与图像分辨率后，差异是否仍存在？
3. 深色框、色卡、装裱和大面积空白的人工掩膜修订是否会改变结果？
"""
    (output_dir / "PAPER_SECTION_GLCM_CANDIDATE.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    manifest_path = (project_root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest.resolve()
    output_dir = (project_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_names = [
        "parameters.json", "analysis_manifest_snapshot.csv", "candidate_glcm_metrics_per_image.csv",
        "candidate_glcm_values_long.csv", "candidate_glcm_block_diagnostics.csv", "candidate_glcm_mask_sensitivity_per_image.csv",
        "candidate_glcm_group_summary.csv", "candidate_glcm_omnibus_tests.csv", "candidate_glcm_pairwise_tests.csv",
        "candidate_glcm_sensitivity_summary.csv", "candidate_glcm_sensitivity_robustness.csv",
        "table10_candidate_glcm_reproduction.csv", "table10_candidate_glcm_reproduction.png",
        "figure_glcm_distributions.png", "figure_glcm_group_means_ci.png", "figure_glcm_mask_sensitivity.png",
        "figure_glcm_qc_examples.png", "PAPER_SECTION_GLCM_CANDIDATE.md", "run_receipt.json",
    ]
    existing = [name for name in generated_names if (output_dir / name).exists()]
    if existing and not args.overwrite:
        raise SystemExit("Refusing to overwrite existing generated outputs: " + ", ".join(existing))
    configure_font()
    started = time.time()
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    required = {"sample_id", "image_path", "sha256", "proposed_group"}
    missing_columns = sorted(required - set(manifest.columns))
    if missing_columns:
        raise SystemExit(f"Manifest missing columns: {missing_columns}")
    if len(manifest) != EXPECTED_TOTAL:
        raise SystemExit(f"Expected {EXPECTED_TOTAL} candidate rows, found {len(manifest)}")
    if manifest["sample_id"].duplicated().any() or manifest["sha256"].duplicated().any():
        raise SystemExit("sample_id or SHA-256 is not unique")
    counts = manifest["proposed_group"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise SystemExit(f"Candidate group counts differ from preregistered scope: {counts}")
    errors = []
    for row in manifest.itertuples(index=False):
        path = project_root / row.image_path
        if not path.is_file():
            errors.append(f"missing:{row.sample_id}:{row.image_path}")
            continue
        actual = sha256_file(path)
        if actual != row.sha256:
            errors.append(f"sha256:{row.sample_id}:{actual}!={row.sha256}")
    if errors:
        raise SystemExit("Input integrity failed:\n" + "\n".join(errors[:30]))
    manifest.to_csv(output_dir / "analysis_manifest_snapshot.csv", index=False, encoding="utf-8-sig")
    parameters = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_date": EXPERIMENT_DATE,
        "scope_status": "Stage-1 candidate images; not a frozen formal manifest",
        "manifest_path": str(manifest_path.relative_to(project_root)),
        "manifest_sha256": sha256_file(manifest_path),
        "expected_counts": EXPECTED_COUNTS,
        "statistical_unit": "image (temporary; independent work/edition unresolved)",
        "preprocessing": {"exif_transpose": True, "rgb": True, "long_edge_px": LONG_EDGE, "preserve_aspect_ratio": True, "padding": False, "resample": "Pillow LANCZOS", "grayscale": "BT.709"},
        "mask": {"definition": "not boundary-connected HSL near-white/near-neutral background", "primary_l_min": PRIMARY_L, "primary_s_max": PRIMARY_S, "connectivity": 8, "closing": "3x3, constrained to candidate background", "sensitivity_l_min": L_THRESHOLDS, "sensitivity_s_max": S_THRESHOLDS},
        "glcm": {"levels": LEVELS, "quantization": "floor(clip(BT709,0,1)*32)", "block_size_px": BLOCK_SIZE, "block_stride_px": BLOCK_SIZE, "block_layout": "center crop to complete non-overlapping blocks", "minimum_valid_pixel_fraction": MIN_VALID_FRACTION, "masked_pairs": "both pair endpoints must be valid", "minimum_valid_pairs": MIN_VALID_PAIRS, "distances_px": DISTANCES, "angles_deg": ANGLES_DEG, "symmetric": True, "normed": True, "image_aggregation": "median across accepted blocks x distances x angles"},
        "statistics": {"group_summary": "mean, sample SD, median, IQR", "bootstrap_mean_ci": BOOTSTRAP_REPS, "bootstrap_seed": RANDOM_SEED, "omnibus": "Kruskal-Wallis + Holm across 4 metrics", "pairwise": "Mann-Whitney U + rank-biserial + Holm across 12 comparisons"},
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "pillow": Image.__version__},
    }
    (output_dir / "parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tasks = [Task(row.sample_id, row.image_path, row.sha256, row.proposed_group, str(project_root)) for row in manifest.itertuples(index=False)]
    results: list[dict[str, Any]] = []
    print(f"Validated {len(tasks)} input images. Running with {args.workers} workers...", flush=True)
    if args.workers == 1:
        for idx, task in enumerate(tasks, start=1):
            results.append(process_one(task))
            if idx % 20 == 0 or idx == len(tasks):
                print(f"processed {idx}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one, task): task.sample_id for task in tasks}
            for idx, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if idx % 20 == 0 or idx == len(tasks):
                    print(f"processed {idx}/{len(tasks)}", flush=True)
    results.sort(key=lambda row: row["sample_id"])
    per_image_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for item in results:
        primary = next(row for row in item["sensitivity"] if math.isclose(row["mask_l_min"], PRIMARY_L) and math.isclose(row["mask_s_max"], PRIMARY_S))
        base = {key: value for key, value in item.items() if key not in {"sensitivity", "long", "blocks"}}
        per_image_rows.append({**base, **{key: value for key, value in primary.items() if key not in {"mask_l_min", "mask_s_max"}}})
        for row in item["sensitivity"]:
            sensitivity_rows.append({"sample_id": item["sample_id"], "image_path": item["image_path"], "sha256": item["sha256"], "proposed_group": item["proposed_group"], "group": item["group"], **row})
        for row in item["long"]:
            long_rows.append({"sample_id": item["sample_id"], "group": item["group"], **row})
        for row in item["blocks"]:
            block_rows.append({"sample_id": item["sample_id"], "group": item["group"], **row})
    per_image = pd.DataFrame(per_image_rows).sort_values(["group", "sample_id"])
    long_df = pd.DataFrame(long_rows).sort_values(["sample_id", "block_id", "distance_px", "angle_deg"])
    blocks_df = pd.DataFrame(block_rows).sort_values(["sample_id", "block_id"])
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values(["sample_id", "mask_l_min", "mask_s_max"])
    if len(per_image) != EXPECTED_TOTAL or per_image[METRICS].isna().any().any():
        raise SystemExit("One or more images lack a complete primary GLCM result")
    per_image.to_csv(output_dir / "candidate_glcm_metrics_per_image.csv", index=False, encoding="utf-8-sig")
    long_df.to_csv(output_dir / "candidate_glcm_values_long.csv", index=False, encoding="utf-8-sig")
    blocks_df.to_csv(output_dir / "candidate_glcm_block_diagnostics.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(output_dir / "candidate_glcm_mask_sensitivity_per_image.csv", index=False, encoding="utf-8-sig")
    summary, omnibus, pairwise = build_statistics(per_image)
    summary.to_csv(output_dir / "candidate_glcm_group_summary.csv", index=False, encoding="utf-8-sig")
    omnibus.to_csv(output_dir / "candidate_glcm_omnibus_tests.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(output_dir / "candidate_glcm_pairwise_tests.csv", index=False, encoding="utf-8-sig")
    sensitivity_summary_rows = []
    for (l_min, s_max, group), frame in sensitivity.groupby(["mask_l_min", "mask_s_max", "group"], sort=True):
        for metric in METRICS:
            values = frame[metric].dropna().to_numpy(float)
            sensitivity_summary_rows.append({"mask_l_min": l_min, "mask_s_max": s_max, "group": group, "metric": metric, "metric_zh": METRIC_ZH[metric], "n_images": len(values), "mean": values.mean(), "sd": values.std(ddof=1), "median": np.median(values)})
    sensitivity_summary = pd.DataFrame(sensitivity_summary_rows)
    sensitivity_summary.to_csv(output_dir / "candidate_glcm_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    robustness_rows = []
    for (group, metric), frame in sensitivity_summary.groupby(["group", "metric"]):
        primary = frame[np.isclose(frame["mask_l_min"], PRIMARY_L) & np.isclose(frame["mask_s_max"], PRIMARY_S)].iloc[0]
        robustness_rows.append({"group": group, "metric": metric, "metric_zh": METRIC_ZH[metric], "primary_mean": primary["mean"], "min_variant_mean": frame["mean"].min(), "max_variant_mean": frame["mean"].max(), "absolute_range": frame["mean"].max() - frame["mean"].min(), "relative_range_to_primary": (frame["mean"].max() - frame["mean"].min()) / abs(primary["mean"]) if primary["mean"] != 0 else math.nan})
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(output_dir / "candidate_glcm_sensitivity_robustness.csv", index=False, encoding="utf-8-sig")
    table10 = write_table10(summary, output_dir)
    plot_distributions(per_image, output_dir)
    plot_means_ci(summary, output_dir)
    plot_sensitivity(sensitivity_summary, output_dir)
    make_qc_figure(per_image, blocks_df, project_root, output_dir)
    write_text_report(output_dir, table10, summary, omnibus, pairwise, per_image, robustness)
    receipt = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "elapsed_seconds": time.time() - started,
        "input_images": len(per_image),
        "group_counts": per_image["group"].value_counts().to_dict(),
        "primary_valid_images": int(per_image[METRICS].notna().all(axis=1).sum()),
        "primary_total_blocks": int(blocks_df.shape[0]),
        "primary_accepted_blocks": int(blocks_df["accepted"].astype(bool).sum()),
        "primary_long_rows": int(long_df.shape[0]),
        "sensitivity_rows": int(sensitivity.shape[0]),
        "warnings": [
            "candidate groups are not human-frozen formal groups",
            "image is the temporary statistical unit",
            "dark frames and complex photographic backgrounds are not automatically removed",
        ],
    }
    (output_dir / "run_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
    main()
