#!/usr/bin/env python3
"""Exploratory line-feature reproduction on the 459+9+100 candidate images.

The analysis is deliberately labeled candidate/exploratory. It does not turn
unreviewed folder labels, series members, or aHash pairs into a frozen formal
manifest. Old manuscript values are never read or used as targets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gusu_line_candidate_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gusu_line_candidate_cache")

import cv2
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.font_manager import FontProperties
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / (
    "reports/formal_aesthetics_reproduction_20260727/"
    "formal_analysis_manifest.csv"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

GROUP_MAP = {
    "清代姑苏版画（候选，待确认）": ("gusu_qing_candidate", "清代姑苏版画（候选）", 1),
    "20世纪50年代桃花坞年画（候选，待确认）": (
        "taohuawu_1950s_candidate",
        "20世纪50年代桃花坞年画（候选）",
        2,
    ),
    "清末杨柳青年画（候选，待确认）": (
        "yangliuqing_late_qing_candidate",
        "清末杨柳青年画（候选）",
        3,
    ),
}
EXPECTED_COUNTS = {
    "gusu_qing_candidate": 459,
    "taohuawu_1950s_candidate": 9,
    "yangliuqing_late_qing_candidate": 100,
}
GROUP_ORDER = list(EXPECTED_COUNTS)
GROUP_SHORT = {
    "gusu_qing_candidate": "姑苏",
    "taohuawu_1950s_candidate": "桃花坞",
    "yangliuqing_late_qing_candidate": "杨柳青",
}

LONG_EDGE = 512
SOBEL_THEORETICAL_MAX = math.sqrt(2.0) * 4.0 * 255.0
SOBEL_THRESHOLDS = (0.08, 0.10, 0.12)
PRIMARY_SOBEL_THRESHOLD = 0.10
CANNY_THRESHOLDS = ((40, 120), (50, 150), (60, 180))
PRIMARY_CANNY = (50, 150)
BORDER_EXCLUSIONS = (0.00, 0.05, 0.10)
PRIMARY_BORDER_EXCLUSION = 0.00
GAUSSIAN_KERNEL = 5
GAUSSIAN_SIGMA = 1.0
DIRECTION_BINS = 9
BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 20260727

OUTPUT_FILENAMES = (
    "analysis_manifest_snapshot.csv",
    "candidate_line_metrics_per_image.csv",
    "candidate_line_sensitivity_per_image.csv",
    "candidate_line_metrics_summary.csv",
    "candidate_line_sensitivity_summary.csv",
    "candidate_line_sensitivity_omnibus_tests.csv",
    "candidate_line_sensitivity_pairwise_tests.csv",
    "candidate_line_omnibus_tests.csv",
    "candidate_line_pairwise_tests.csv",
    "candidate_source_prefix_summary.csv",
    "candidate_line_parameters.json",
    "table9_candidate_exploratory.csv",
    "table9_candidate_exploratory.png",
    "figure_line_metric_distributions.png",
    "figure_edge_detection_candidate_qc.png",
    "FULL_PAPER_SECTION.md",
    "VALIDATION_REPORT.md",
    "chart_map.md",
    "candidate_line_analysis.ipynb",
    "artifact.json",
)


def read_image_rgb(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot_read_image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_long_edge(image: np.ndarray, long_edge: int = LONG_EDGE) -> np.ndarray:
    height, width = image.shape[:2]
    scale = long_edge / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def rgb_to_bt709_gray(image_rgb: np.ndarray) -> np.ndarray:
    rgb = image_rgb.astype(np.float32)
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def valid_mask_for(shape: tuple[int, int], border_fraction: float) -> np.ndarray:
    height, width = shape
    margin_y = max(1, int(round(height * border_fraction)))
    margin_x = max(1, int(round(width * border_fraction)))
    mask = np.zeros((height, width), dtype=bool)
    if height > 2 * margin_y and width > 2 * margin_x:
        mask[margin_y : height - margin_y, margin_x : width - margin_x] = True
    return mask


def direction_entropy(
    orientation_deg: np.ndarray,
    magnitude: np.ndarray,
    magnitude_normalized: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float,
) -> tuple[float, int]:
    selected = valid_mask & (magnitude_normalized >= threshold)
    count = int(selected.sum())
    if count == 0:
        return float("nan"), 0
    hist, _ = np.histogram(
        orientation_deg[selected],
        bins=np.linspace(0.0, 180.0, DIRECTION_BINS + 1),
        weights=magnitude[selected],
    )
    total = float(hist.sum())
    if total <= 0:
        return float("nan"), count
    probabilities = hist / total
    nonzero = probabilities > 0
    raw_entropy = -float(np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))
    return raw_entropy / math.log(DIRECTION_BINS), count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    required = {"sample_id", "image_path", "sha256", "proposed_group", "source_prefix"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manifest_missing_columns: {sorted(missing)}")
    if len(frame) != 568 or frame["sample_id"].nunique() != 568:
        raise ValueError(
            f"candidate_manifest_expected_568_unique_rows: rows={len(frame)}, "
            f"unique={frame['sample_id'].nunique()}"
        )
    unknown_groups = sorted(set(frame["proposed_group"]) - set(GROUP_MAP))
    if unknown_groups:
        raise ValueError(f"unexpected_candidate_groups: {unknown_groups}")
    mapped = frame["proposed_group"].map(GROUP_MAP)
    frame["group_id"] = mapped.map(lambda value: value[0])
    frame["group_label"] = mapped.map(lambda value: value[1])
    frame["group_order"] = mapped.map(lambda value: value[2])
    counts = frame.groupby("group_id").size().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"candidate_group_counts_mismatch: {counts}")
    frame["resolved_image_path"] = frame["image_path"].map(
        lambda value: str((PROJECT_ROOT / value).resolve()) if not Path(value).is_absolute() else value
    )
    return frame


def calculate_candidate_metrics(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    primary_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    qc_products: list[dict[str, Any]] = []

    qc_sample_ids = {
        group_id: manifest.loc[manifest["group_id"] == group_id, "sample_id"].sort_values().iloc[0]
        for group_id in GROUP_ORDER
    }

    for index, row in manifest.iterrows():
        image_path = Path(row["resolved_image_path"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        observed_sha = sha256_file(image_path)
        if observed_sha.lower() != row["sha256"].lower():
            raise ValueError(f"sha256_mismatch: {row['sample_id']} {image_path}")

        original = read_image_rgb(image_path)
        resized = resize_long_edge(original)
        gray = rgb_to_bt709_gray(resized)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)
        magnitude = cv2.magnitude(gx, gy)
        magnitude_normalized = np.clip(magnitude / SOBEL_THEORETICAL_MAX, 0.0, 1.0)
        orientation = np.mod(np.degrees(np.arctan2(gy, gx)), 180.0)

        blurred = cv2.GaussianBlur(
            gray,
            (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL),
            sigmaX=GAUSSIAN_SIGMA,
            sigmaY=GAUSSIAN_SIGMA,
            borderType=cv2.BORDER_REFLECT101,
        )
        canny_maps: dict[tuple[int, int], np.ndarray] = {}
        for low, high in CANNY_THRESHOLDS:
            canny_maps[(low, high)] = cv2.Canny(
                blurred,
                threshold1=low,
                threshold2=high,
                apertureSize=3,
                L2gradient=True,
            ) > 0

        for border_fraction in BORDER_EXCLUSIONS:
            valid_mask = valid_mask_for(gray.shape, border_fraction)
            valid_count = int(valid_mask.sum())
            sensitivity: dict[str, Any] = {
                "sample_id": row["sample_id"],
                "group_id": row["group_id"],
                "group_label": row["group_label"],
                "source_prefix": row["source_prefix"],
                "border_exclusion_fraction": border_fraction,
                "valid_pixel_count": valid_count,
            }
            for threshold in SOBEL_THRESHOLDS:
                entropy, selected_count = direction_entropy(
                    orientation,
                    magnitude,
                    magnitude_normalized,
                    valid_mask,
                    threshold,
                )
                key = f"t{int(round(threshold * 100)):03d}"
                sensitivity[f"direction_entropy_{key}"] = entropy
                sensitivity[f"strong_gradient_ratio_{key}"] = selected_count / valid_count
            for low, high in CANNY_THRESHOLDS:
                key = f"l{low:03d}_h{high:03d}"
                sensitivity[f"canny_edge_density_{key}"] = float(
                    np.logical_and(canny_maps[(low, high)], valid_mask).sum() / valid_count
                )
            sensitivity_rows.append(sensitivity)

            if math.isclose(border_fraction, PRIMARY_BORDER_EXCLUSION):
                primary_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "image_path": row["image_path"],
                        "sha256": row["sha256"],
                        "proposed_group": row["proposed_group"],
                        "group_id": row["group_id"],
                        "group_label": row["group_label"],
                        "group_order": int(row["group_order"]),
                        "source_prefix": row["source_prefix"],
                        "series_id_candidate": row.get("series_id", ""),
                        "work_id_candidate": row.get("work_id", ""),
                        "original_width": int(original.shape[1]),
                        "original_height": int(original.shape[0]),
                        "analysis_width": int(resized.shape[1]),
                        "analysis_height": int(resized.shape[0]),
                        "valid_pixel_count": valid_count,
                        "strong_gradient_pixel_count": int(
                            round(sensitivity["strong_gradient_ratio_t010"] * valid_count)
                        ),
                        "canny_edge_pixel_count": int(
                            round(sensitivity["canny_edge_density_l050_h150"] * valid_count)
                        ),
                        "normalized_direction_entropy": sensitivity["direction_entropy_t010"],
                        "sobel_strong_gradient_ratio": sensitivity["strong_gradient_ratio_t010"],
                        "canny_edge_density": sensitivity["canny_edge_density_l050_h150"],
                        "analysis_scope": "candidate_exploratory_not_formal_manifest",
                    }
                )

        if row["sample_id"] == qc_sample_ids[row["group_id"]]:
            qc_products.append(
                {
                    "sample_id": row["sample_id"],
                    "group_id": row["group_id"],
                    "group_label": row["group_label"],
                    "image_path": row["image_path"],
                    "rgb": resized,
                    "gray": gray,
                    "magnitude_normalized": magnitude_normalized,
                    "canny": canny_maps[PRIMARY_CANNY],
                }
            )

        if (index + 1) % 100 == 0 or index + 1 == len(manifest):
            print(f"processed {index + 1}/{len(manifest)}", flush=True)

    primary = pd.DataFrame(primary_rows).sort_values(["group_order", "sample_id"]).reset_index(drop=True)
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values(
        ["group_id", "sample_id", "border_exclusion_fraction"]
    ).reset_index(drop=True)
    return primary, sensitivity, qc_products


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    sampled = rng.choice(values, size=(BOOTSTRAP_ITERATIONS, len(values)), replace=True)
    boot_means = sampled.mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


def summarize_metric_rows(frame: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    for group_id in GROUP_ORDER:
        subset = frame[frame["group_id"] == group_id]
        for metric in metrics:
            values = subset[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            ci_low, ci_high = bootstrap_mean_ci(values, rng)
            rows.append(
                {
                    "group_id": group_id,
                    "group_label": subset["group_label"].iloc[0],
                    "group_order": int(subset["group_order"].iloc[0]) if "group_order" in subset else GROUP_ORDER.index(group_id) + 1,
                    "metric": metric,
                    "n": int(len(values)),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "mean_ci95_low": ci_low,
                    "mean_ci95_high": ci_high,
                    "median": float(np.median(values)),
                    "q1": float(np.percentile(values, 25)),
                    "q3": float(np.percentile(values, 75)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (count - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def inferential_tests(frame: pd.DataFrame, metrics: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    omnibus_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    for metric in metrics:
        arrays = [
            frame.loc[frame["group_id"] == group_id, metric].dropna().to_numpy(dtype=float)
            for group_id in GROUP_ORDER
        ]
        statistic, p_value = stats.kruskal(*arrays)
        omnibus_rows.append(
            {
                "metric": metric,
                "test": "Kruskal-Wallis",
                "statistic_H": float(statistic),
                "degrees_of_freedom": 2,
                "p_value": float(p_value),
                "analysis_unit": "candidate_image",
            }
        )
        metric_pairs: list[dict[str, Any]] = []
        for first_index in range(len(GROUP_ORDER)):
            for second_index in range(first_index + 1, len(GROUP_ORDER)):
                group_a = GROUP_ORDER[first_index]
                group_b = GROUP_ORDER[second_index]
                values_a = arrays[first_index]
                values_b = arrays[second_index]
                result = stats.mannwhitneyu(values_a, values_b, alternative="two-sided", method="asymptotic")
                rank_biserial = 2.0 * float(result.statistic) / (len(values_a) * len(values_b)) - 1.0
                metric_pairs.append(
                    {
                        "metric": metric,
                        "group_a": group_a,
                        "group_a_label": GROUP_SHORT[group_a],
                        "group_b": group_b,
                        "group_b_label": GROUP_SHORT[group_b],
                        "n_a": len(values_a),
                        "n_b": len(values_b),
                        "median_a": float(np.median(values_a)),
                        "median_b": float(np.median(values_b)),
                        "median_difference_a_minus_b": float(np.median(values_a) - np.median(values_b)),
                        "U_statistic": float(result.statistic),
                        "p_value_raw": float(result.pvalue),
                        "rank_biserial_a_higher_positive": rank_biserial,
                    }
                )
        adjusted = holm_adjust([row["p_value_raw"] for row in metric_pairs])
        for row, corrected in zip(metric_pairs, adjusted):
            row["p_value_holm_within_metric"] = corrected
            row["significant_holm_0_05"] = corrected < 0.05
            pairwise_rows.append(row)
    omnibus = pd.DataFrame(omnibus_rows)
    omnibus["p_value_holm_across_metrics"] = holm_adjust(omnibus["p_value"].tolist())
    omnibus["significant_holm_0_05"] = omnibus["p_value_holm_across_metrics"] < 0.05
    return omnibus, pd.DataFrame(pairwise_rows)


def sensitivity_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in frame.columns
        if column.startswith("direction_entropy_")
        or column.startswith("strong_gradient_ratio_")
        or column.startswith("canny_edge_density_")
    ]
    rows: list[dict[str, Any]] = []
    for (group_id, group_label, border), subset in frame.groupby(
        ["group_id", "group_label", "border_exclusion_fraction"], sort=False
    ):
        for metric in metric_columns:
            values = subset[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "group_id": group_id,
                    "group_label": group_label,
                    "border_exclusion_fraction": border,
                    "metric": metric,
                    "n": int(np.isfinite(values).sum()),
                    "mean": float(np.nanmean(values)),
                    "sd": float(np.nanstd(values, ddof=1)),
                    "median": float(np.nanmedian(values)),
                }
            )
    return pd.DataFrame(rows)


def sensitivity_inferential_tests(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the same non-parametric tests for each pre-specified sensitivity setting.

    Holm adjustment is applied to the three pairwise contrasts within each single
    parameter setting. These tests diagnose robustness; they are not used to
    select the primary threshold.
    """
    metrics = [
        "direction_entropy_t008",
        "direction_entropy_t010",
        "direction_entropy_t012",
        "canny_edge_density_l040_h120",
        "canny_edge_density_l050_h150",
        "canny_edge_density_l060_h180",
    ]
    omnibus_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    for border in BORDER_EXCLUSIONS:
        border_frame = frame[np.isclose(frame["border_exclusion_fraction"], border)]
        for metric in metrics:
            arrays = [
                border_frame.loc[border_frame["group_id"] == group_id, metric]
                .dropna()
                .to_numpy(dtype=float)
                for group_id in GROUP_ORDER
            ]
            statistic, p_value = stats.kruskal(*arrays)
            family = "direction_entropy" if metric.startswith("direction_entropy") else "canny_edge_density"
            omnibus_rows.append(
                {
                    "metric_family": family,
                    "metric": metric,
                    "border_exclusion_fraction": border,
                    "test": "Kruskal-Wallis",
                    "statistic_H": float(statistic),
                    "degrees_of_freedom": 2,
                    "p_value": float(p_value),
                    "analysis_unit": "candidate_image",
                }
            )
            setting_pairs: list[dict[str, Any]] = []
            for first_index in range(len(GROUP_ORDER)):
                for second_index in range(first_index + 1, len(GROUP_ORDER)):
                    group_a = GROUP_ORDER[first_index]
                    group_b = GROUP_ORDER[second_index]
                    values_a = arrays[first_index]
                    values_b = arrays[second_index]
                    result = stats.mannwhitneyu(
                        values_a, values_b, alternative="two-sided", method="asymptotic"
                    )
                    rank_biserial = 2.0 * float(result.statistic) / (
                        len(values_a) * len(values_b)
                    ) - 1.0
                    setting_pairs.append(
                        {
                            "metric_family": family,
                            "metric": metric,
                            "border_exclusion_fraction": border,
                            "group_a": group_a,
                            "group_a_label": GROUP_SHORT[group_a],
                            "group_b": group_b,
                            "group_b_label": GROUP_SHORT[group_b],
                            "n_a": len(values_a),
                            "n_b": len(values_b),
                            "median_a": float(np.median(values_a)),
                            "median_b": float(np.median(values_b)),
                            "median_difference_a_minus_b": float(
                                np.median(values_a) - np.median(values_b)
                            ),
                            "U_statistic": float(result.statistic),
                            "p_value_raw": float(result.pvalue),
                            "rank_biserial_a_higher_positive": rank_biserial,
                        }
                    )
            adjusted = holm_adjust([row["p_value_raw"] for row in setting_pairs])
            for row, corrected in zip(setting_pairs, adjusted):
                row["p_value_holm_within_setting"] = corrected
                row["significant_holm_0_05"] = corrected < 0.05
                pairwise_rows.append(row)
    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)


def prefix_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["group_id", "group_label", "source_prefix"], as_index=False)
        .agg(
            n=("sample_id", "size"),
            direction_entropy_mean=("normalized_direction_entropy", "mean"),
            direction_entropy_sd=("normalized_direction_entropy", "std"),
            canny_edge_density_mean=("canny_edge_density", "mean"),
            canny_edge_density_sd=("canny_edge_density", "std"),
        )
        .sort_values(["group_id", "n", "source_prefix"], ascending=[True, False, True])
    )


def chinese_font() -> FontProperties | None:
    candidates = (
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for path in candidates:
        if path.exists():
            return FontProperties(fname=str(path))
    return None


def font_kwargs() -> dict[str, Any]:
    font = chinese_font()
    return {"fontproperties": font} if font else {}


def make_distribution_figure(frame: pd.DataFrame, output_path: Path) -> None:
    colors = ["#3767A6", "#C58A20", "#C75D45"]
    rng = np.random.default_rng(RANDOM_SEED)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    specs = [
        ("normalized_direction_entropy", "归一化方向熵", "0–1；越低表示方向分布越集中", (0, 1)),
        ("canny_edge_density", "Canny边缘密度", "有效区域内边缘像素比例", (0, None)),
    ]
    labels = [GROUP_SHORT[group_id] + f"\n(n={EXPECTED_COUNTS[group_id]})" for group_id in GROUP_ORDER]
    for axis, (metric, title, subtitle, limits) in zip(axes, specs):
        arrays = [
            frame.loc[frame["group_id"] == group_id, metric].to_numpy(dtype=float)
            for group_id in GROUP_ORDER
        ]
        box = axis.boxplot(
            arrays,
            positions=np.arange(1, 4),
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#202124", "linewidth": 1.6},
            whiskerprops={"color": "#50545A", "linewidth": 1.0},
            capprops={"color": "#50545A", "linewidth": 1.0},
            boxprops={"linewidth": 1.0, "edgecolor": "#50545A"},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.28)
        for position, values, color in zip(range(1, 4), arrays, colors):
            jitter = rng.uniform(-0.17, 0.17, size=len(values))
            axis.scatter(
                position + jitter,
                values,
                s=11 if len(values) < 150 else 7,
                color=color,
                alpha=0.55 if len(values) < 150 else 0.23,
                edgecolors="none",
                rasterized=True,
            )
        axis.set_xticks([1, 2, 3])
        axis.set_xticklabels(labels, **font_kwargs())
        axis.set_title(
            f"{title}\n{subtitle}",
            loc="left",
            fontsize=13,
            linespacing=1.35,
            pad=10,
            **font_kwargs(),
        )
        axis.grid(axis="y", color="#D8DADF", linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        if limits[1] is None:
            axis.set_ylim(0, max(max(values) for values in arrays) * 1.08)
        else:
            axis.set_ylim(*limits)
        if metric == "canny_edge_density":
            axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
    fig.suptitle("三类候选版画图像的线条操作性指标分布", fontsize=18, **font_kwargs())
    fig.text(
        0.5,
        -0.01,
        "候选样本探索性结果；统计单位为图像，尚未完成作品级去重与来源混杂控制。",
        ha="center",
        color="#5A6068",
        fontsize=10,
        **font_kwargs(),
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_qc_figure(products: list[dict[str, Any]], output_path: Path) -> None:
    products = sorted(products, key=lambda item: GROUP_ORDER.index(item["group_id"]))
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 12.5), constrained_layout=True)
    for row_index, product in enumerate(products):
        rgb = product["rgb"]
        magnitude = product["magnitude_normalized"]
        canny = product["canny"]
        overlay = rgb.astype(np.float32)
        red = np.array([215.0, 40.0, 40.0], dtype=np.float32)
        overlay[canny] = 0.25 * overlay[canny] + 0.75 * red
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        panels = [
            (rgb, "原图（等比例缩放）", None, None, None),
            (magnitude, "Sobel梯度幅值", "magma", 0, 1),
            (overlay, "Canny边缘叠加", None, None, None),
        ]
        for column_index, (data, title, cmap, vmin, vmax) in enumerate(panels):
            axis = axes[row_index, column_index]
            axis.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.axis("off")
            axis.set_title(title, fontsize=11, **font_kwargs())
        axes[row_index, 0].text(
            -0.06,
            0.5,
            f"{GROUP_SHORT[product['group_id']]}\n{product['sample_id']}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=10,
            **font_kwargs(),
        )
    fig.suptitle("候选样本边缘检测质控示例", fontsize=17, **font_kwargs())
    fig.text(
        0.5,
        -0.005,
        "每组按sample_id排序取首张，不按结果挑选；图中博物馆边框、标尺或留白可能进入计算。",
        ha="center",
        color="#5A6068",
        fontsize=9.5,
        **font_kwargs(),
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_table_outputs(summary: pd.DataFrame, csv_path: Path, png_path: Path) -> pd.DataFrame:
    records = []
    for group_id in GROUP_ORDER:
        entropy = summary[(summary["group_id"] == group_id) & (summary["metric"] == "normalized_direction_entropy")].iloc[0]
        edge = summary[(summary["group_id"] == group_id) & (summary["metric"] == "canny_edge_density")].iloc[0]
        records.append(
            {
                "candidate_group": GROUP_SHORT[group_id],
                "n_images": int(entropy["n"]),
                "normalized_direction_entropy_mean": entropy["mean"],
                "normalized_direction_entropy_sd": entropy["sd"],
                "normalized_direction_entropy_mean_sd": f"{entropy['mean']:.3f} ± {entropy['sd']:.3f}",
                "canny_edge_density_mean": edge["mean"],
                "canny_edge_density_sd": edge["sd"],
                "canny_edge_density_mean_sd_percent": f"{edge['mean'] * 100:.2f}% ± {edge['sd'] * 100:.2f}%",
            }
        )
    table = pd.DataFrame(records)
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig, axis = plt.subplots(figsize=(10.5, 3.3))
    axis.axis("off")
    display_rows = [
        [row["candidate_group"], str(row["n_images"]), row["normalized_direction_entropy_mean_sd"], row["canny_edge_density_mean_sd_percent"]]
        for _, row in table.iterrows()
    ]
    mpl_table = axis.table(
        cellText=display_rows,
        colLabels=["候选组", "图像数", "归一化方向熵（0–1）", "Canny边缘密度"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.23, 0.13, 0.31, 0.28],
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(12)
    font = chinese_font()
    for (row_index, _), cell in mpl_table.get_celld().items():
        cell.set_height(0.20 if row_index == 0 else 0.18)
        cell.set_edgecolor("#555A61")
        if font:
            cell.get_text().set_fontproperties(font)
        if row_index == 0:
            cell.set_facecolor("#E6EEF7")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#FAFBFC")
    axis.set_title(
        "表9  基于Sobel方向统计与Canny边缘检测的候选样本探索性结果",
        fontsize=15,
        pad=18,
        **font_kwargs(),
    )
    fig.text(
        0.5,
        0.04,
        "数值为图像级均值 ± 标准差；候选标签、作品关系与数字化来源尚未完成正式审核。",
        ha="center",
        color="#5A6068",
        fontsize=9.5,
        **font_kwargs(),
    )
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return table


def p_text(value: float) -> str:
    if value < 0.001:
        return "p < 0.001"
    return f"p = {value:.3f}"


def pair_sentence(pairwise: pd.DataFrame, metric: str, group_a: str, group_b: str) -> str:
    row = pairwise[
        (pairwise["metric"] == metric)
        & (pairwise["group_a"] == group_a)
        & (pairwise["group_b"] == group_b)
    ].iloc[0]
    direction = "高于" if row["rank_biserial_a_higher_positive"] > 0 else "低于"
    return (
        f"{GROUP_SHORT[group_a]}的分布{direction}{GROUP_SHORT[group_b]}"
        f"（Holm校正{p_text(row['p_value_holm_within_metric'])}，"
        f"秩二列相关 r_rb={row['rank_biserial_a_higher_positive']:.3f}）"
    )


def make_paper_section(
    output_path: Path,
    table: pd.DataFrame,
    omnibus: pd.DataFrame,
    pairwise: pd.DataFrame,
    sensitivity_omnibus: pd.DataFrame,
) -> None:
    table_by_group = table.set_index("candidate_group")
    entropy_omnibus = omnibus[omnibus["metric"] == "normalized_direction_entropy"].iloc[0]
    edge_omnibus = omnibus[omnibus["metric"] == "canny_edge_density"].iloc[0]
    entropy_means = {row["candidate_group"]: row["normalized_direction_entropy_mean"] for _, row in table.iterrows()}
    edge_means = {row["candidate_group"]: row["canny_edge_density_mean"] for _, row in table.iterrows()}
    lowest_entropy = min(entropy_means, key=entropy_means.get)
    lowest_edge = min(edge_means, key=edge_means.get)
    entropy_border_zero = sensitivity_omnibus[
        (sensitivity_omnibus["metric_family"] == "direction_entropy")
        & np.isclose(sensitivity_omnibus["border_exclusion_fraction"], 0.0)
    ]
    entropy_border_five = sensitivity_omnibus[
        (sensitivity_omnibus["metric_family"] == "direction_entropy")
        & np.isclose(sensitivity_omnibus["border_exclusion_fraction"], 0.05)
    ]
    entropy_border_ten = sensitivity_omnibus[
        (sensitivity_omnibus["metric_family"] == "direction_entropy")
        & np.isclose(sensitivity_omnibus["border_exclusion_fraction"], 0.10)
    ]
    edge_sensitivity = sensitivity_omnibus[
        sensitivity_omnibus["metric_family"] == "canny_edge_density"
    ]

    table_lines = [
        "| 候选组 | n（图像） | 归一化方向熵（0–1） | Canny边缘密度 |",
        "|---|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        table_lines.append(
            f"| {row['candidate_group']} | {int(row['n_images'])} | "
            f"{row['normalized_direction_entropy_mean_sd']} | "
            f"{row['canny_edge_density_mean_sd_percent']} |"
        )
    table_markdown = "\n".join(table_lines)

    text = f"""# （二）线条美学：候选样本的梯度方向与边缘分布

> **证据状态说明：**本节使用Stage 1中459张清代姑苏候选图、9张20世纪50年代桃花坞候选图和100张清末杨柳青候选图进行探索性复现。三组标签、年代、独立作品关系、同版关系及数字化来源尚未完成正式人工冻结，故以下结果只能写作“候选样本探索性分析”，不能表述为最终总体结论。

线条是木刻版画形式语言的重要组成部分。本研究将连续的亮度梯度与二值边缘检测分开：Sobel算子用于计算梯度幅值和梯度方向，Canny算法用于生成二值边缘图。两类指标分别描述梯度方向分布和可检测边缘像素比例，不直接测量刻线的工艺质量、断裂率或审美价值。

## 1. 数据与计算方法

分析以每幅候选图像为一个统计观察单位。图像保持原始纵横比等比例缩放至长边512像素，不补边；灰度按BT.709亮度公式计算：

`Y = 0.2126R + 0.7152G + 0.0722B`。

使用3×3 Sobel算子获得水平梯度`Gx`和垂直梯度`Gy`，梯度幅值与无符号方向分别为：

`M = sqrt(Gx² + Gy²)`，`θ = atan2(Gy,Gx) mod 180°`。

梯度幅值按固定理论尺度`sqrt(2)×4×255`归一化；主分析仅保留归一化幅值不低于0.10的像素。将`θ∈[0°,180°)`划分为9个20°区间，以梯度幅值加权得到方向概率`p_k`，归一化方向熵定义为：

`H_dir = -Σ p_k log(p_k) / log(9)`。

该指标范围为0—1；数值越低表示梯度集中于较少方向，数值越高表示方向分布更分散。它不等同于“线条规整度”或“线条连续性”。

Canny边缘检测前采用5×5高斯平滑（`sigma=1.0`），主分析双阈值固定为50和150，`apertureSize=3`，`L2gradient=true`。边缘密度定义为：

`D_edge = #(Canny边缘像素且属于有效区域) / #有效区域像素`。

主分析仅排除缩放图像最外侧1像素；另以排除四周5%和10%区域作为边界敏感性检查。方向熵同时检查Sobel阈值0.08、0.10和0.12；边缘密度同时检查Canny阈值40/120、50/150和60/180。这些主参数在组间结果生成前固定，未根据论文原表或预期方向调节。

考虑三组样本量高度不均衡且桃花坞组仅9张，描述统计同时报告均值、标准差、中位数和四分位距。总体差异采用Kruskal–Wallis检验；两两比较采用双侧Mann–Whitney U检验，并在每项指标内使用Holm法校正三次比较。秩二列相关系数`r_rb`报告效应方向和大小。由于作品级重复尚未解决，统计推断仍可能低估不确定性。

## 2. 探索性结果

**表9  基于Sobel方向统计与Canny边缘检测的三类候选版画图像线条特征**

{table_markdown}

注：数值为图像级均值±标准差。方向熵为9方向幅值加权后除以`log(9)`的归一化熵；边缘密度为Canny二值边缘像素比例。本表不是正式manifest结果。

三组候选图像的归一化方向熵存在总体差异（Kruskal–Wallis `H={entropy_omnibus['statistic_H']:.3f}`，校正后{p_text(entropy_omnibus['p_value_holm_across_metrics'])}）。按均值比较，{lowest_entropy}候选组的方向熵最低，但这一结果只能说明其图像梯度更集中于少数方向，不能直接解释为刻版更精细或线条更规整。两两比较显示，{pair_sentence(pairwise, 'normalized_direction_entropy', 'gusu_qing_candidate', 'taohuawu_1950s_candidate')}；{pair_sentence(pairwise, 'normalized_direction_entropy', 'gusu_qing_candidate', 'yangliuqing_late_qing_candidate')}。

三组Canny边缘密度的总体差异检验结果为`H={edge_omnibus['statistic_H']:.3f}`（校正后{p_text(edge_omnibus['p_value_holm_across_metrics'])}）。按均值比较，{lowest_edge}候选组的边缘像素比例最低。具体而言，{pair_sentence(pairwise, 'canny_edge_density', 'gusu_qing_candidate', 'taohuawu_1950s_candidate')}；{pair_sentence(pairwise, 'canny_edge_density', 'gusu_qing_candidate', 'yangliuqing_late_qing_candidate')}。边缘密度同时受到画面内容、扫描清晰度、纸张污损、博物馆标尺和边框影响，不能直接改写为“构图疏朗”。

### 敏感性分析

方向熵对边界处理敏感。在不额外裁切边界时，三个Sobel阈值的总体检验均达到未校正`p<0.05`（`H={entropy_border_zero['statistic_H'].min():.3f}—{entropy_border_zero['statistic_H'].max():.3f}`，`p={entropy_border_zero['p_value'].min():.3f}—{entropy_border_zero['p_value'].max():.3f}`）；排除四周5%后，三个检验均不显著（`p={entropy_border_five['p_value'].min():.3f}—{entropy_border_five['p_value'].max():.3f}`），排除四周10%后亦均不显著（`p={entropy_border_ten['p_value'].min():.3f}—{entropy_border_ten['p_value'].max():.3f}`）。这说明主分析中的方向熵差异可能部分由画框、标尺、纸张边缘或留白结构驱动，现阶段不宜据此形成“姑苏线条方向更集中”的稳定结论。

Canny边缘密度的总体差异在3组阈值×3种边界设置共9个预设组合中均保持同一方向，且Kruskal–Wallis检验均为`p<0.001`（`H={edge_sensitivity['statistic_H'].min():.3f}—{edge_sensitivity['statistic_H'].max():.3f}`）。这表明候选图像层面的边缘像素比例差异对所检查参数具有稳健性；但参数稳健并不能排除组别与数字化来源共线造成的混杂，因此仍不能把差异直接归因于地域画风或刻版工艺。

![图5 三类候选样本的指标分布](figure_line_metric_distributions.png)

**图5** 三类候选版画图像的归一化方向熵和Canny边缘密度分布。箱体表示四分位区间，中线表示中位数，散点表示逐图结果。图像是当前统计单位，未进行作品级聚合。

![图6 候选样本边缘检测质控](figure_edge_detection_candidate_qc.png)

**图6** 候选样本边缘检测质控示例。每组按稳定`sample_id`排序选取首张，不按指标结果选择。红线为Canny边缘叠加，可见部分来源的边框、标尺和留白也会产生边缘，说明来源混杂必须在最终分析中控制。

## 3. 解释边界与论文表述

本次复现纠正了原表的两个问题。第一，方向熵采用除以`log(9)`的归一化定义，因此结果严格位于0—1；旧稿中1.753、1.803和1.798不满足该定义，且没有找到原始计算代码，故不再沿用。第二，Sobel梯度幅值与Canny边缘检测具有不同输出，边缘密度必须由明确的二值边缘图和分母计算，不能把梯度幅值直接称为边缘密度。

原表中的“线条连续性”列予以删除。Sobel方向熵和Canny边缘密度都不能测量一条刻线是否连续或断裂；若以后恢复该指标，需要建立人工标注金标准、骨架或连通路径算法、标注者一致性及算法效度。

当前证据最多支持“候选图像在梯度方向分布和二值边缘比例上存在/不存在操作性差异”。“刀笔共生”“精工严谨”“疏朗有致”等判断应作为艺术史案例阐释，不能由这两个指标单独证明。

## 4. 必须保留的限制

1. 桃花坞候选组只有9张，估计不稳定且不能代表20世纪50年代桃花坞年画总体。
2. 568张图像尚未完成正式组别、年代、独立作品、同版和重复关系审核；图像数不等于独立作品数。
3. 三组与来源前缀完全或近乎完全共线，数字化设备、压缩、分辨率、画框和扫描背景可能形成伪组间差异。
4. 主分析没有可靠的逐图内容掩膜；方向熵差异在边界裁切后消失，表明边界和来源伪影具有实质影响。
5. 统计检验以候选图像为独立单位，组画和重复数字图可能导致标准误偏小。

因此，本节可用于论文的探索性分析版本，但不能标为“正式三组总体量化实证”。人工manifest冻结后，应按同一参数重新生成逐图结果并以独立作品/版次为统计单位复核。

## 5. 为什么采用这一修订方案

- **保持指标含义清晰：**Sobel描述梯度，Canny生成二值边缘，各自承担不同测量任务。
- **避免结果导向调参：**主阈值在计算组间结果之前已写入参数文件，并同时公开敏感性组合。
- **避免伪精确：**方向熵归一化到0—1，边缘密度保留比例定义，并公开逐图CSV和像素计数。
- **控制多重比较：**总体检验后对每项指标的三次两两比较进行Holm校正，并报告效应量。
- **限制美学外推：**量化指标只支持具体的视觉操作性差异；审美与工艺结论必须结合来源可靠的作品案例和艺术史证据。
"""
    output_path.write_text(text, encoding="utf-8")


def make_parameters(output_path: Path, manifest_path: Path) -> dict[str, Any]:
    parameters = {
        "analysis_status": "candidate_exploratory_not_formal_manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "expected_candidate_counts": EXPECTED_COUNTS,
        "analysis_unit": "candidate_image",
        "image_preprocessing": {
            "resize": {"long_edge_px": LONG_EDGE, "keep_aspect_ratio": True, "padding": False},
            "grayscale": "BT.709 Y=0.2126R+0.7152G+0.0722B",
            "primary_valid_region": "full resized image excluding one-pixel perimeter",
            "mask_limitation": "no validated content/frame/background mask",
        },
        "sobel": {
            "kernel_size": 3,
            "border_type": "BORDER_REFLECT101",
            "magnitude": "sqrt(Gx^2+Gy^2)",
            "normalization": "divide by sqrt(2)*4*255 and clip to [0,1]",
            "primary_threshold": PRIMARY_SOBEL_THRESHOLD,
            "sensitivity_thresholds": list(SOBEL_THRESHOLDS),
        },
        "direction_entropy": {
            "orientation_domain_degrees": [0, 180],
            "bins": DIRECTION_BINS,
            "bin_width_degrees": 20,
            "weight": "Sobel magnitude",
            "formula": "-sum(p_k*ln(p_k))/ln(9)",
            "range": [0, 1],
        },
        "canny": {
            "gaussian_kernel": [GAUSSIAN_KERNEL, GAUSSIAN_KERNEL],
            "gaussian_sigma": GAUSSIAN_SIGMA,
            "primary_thresholds": list(PRIMARY_CANNY),
            "sensitivity_thresholds": [list(value) for value in CANNY_THRESHOLDS],
            "aperture_size": 3,
            "l2gradient": True,
            "edge_density_formula": "count(edge & valid)/count(valid)",
        },
        "border_sensitivity_exclusion_fraction": list(BORDER_EXCLUSIONS),
        "statistics": {
            "descriptive": "mean, sample SD, bootstrap 95% CI, median, IQR, min, max",
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "omnibus": "Kruskal-Wallis",
            "pairwise": "two-sided Mann-Whitney U, asymptotic",
            "effect_size": "rank-biserial correlation; positive means group_a higher",
            "multiple_testing": "Holm within each metric; Holm across two omnibus tests",
            "random_seed": RANDOM_SEED,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "excluded_claims": [
            "line continuity",
            "breakage rate",
            "craft quality",
            "aesthetic superiority",
            "causal attribution to regional style",
        ],
    }
    output_path.write_text(json.dumps(parameters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return parameters


def make_validation_report(
    output_path: Path,
    manifest: pd.DataFrame,
    metrics: pd.DataFrame,
    sensitivity: pd.DataFrame,
    omnibus: pd.DataFrame,
    pairwise: pd.DataFrame,
    sensitivity_omnibus: pd.DataFrame,
    sensitivity_pairwise: pd.DataFrame,
) -> None:
    missing = metrics[["normalized_direction_entropy", "canny_edge_density"]].isna().sum().to_dict()
    entropy_range = (metrics["normalized_direction_entropy"].min(), metrics["normalized_direction_entropy"].max())
    edge_range = (metrics["canny_edge_density"].min(), metrics["canny_edge_density"].max())
    sensitivity_groups = sensitivity.groupby("border_exclusion_fraction").size().to_dict()
    text = f"""# Validation Report

## Overall Assessment: Share with caveats

The computation is reproducible and internally consistent for the explicitly authorized 459+9+100 candidate-image scope. It is not a formal population analysis because group/era/work/duplicate review is unfinished and source is confounded with group.

## Methodology Review

- Input rows: {len(manifest)}; unique sample IDs: {manifest['sample_id'].nunique()}.
- Candidate counts: {metrics.groupby('group_id').size().to_dict()}.
- SHA-256 and file existence were checked before every metric calculation; all rows passed.
- Primary parameters were fixed before group results: Sobel threshold 0.10, 9 direction bins, Canny 50/150, full resized image minus one-pixel perimeter.
- “Line continuity” was not computed because no validated definition exists.

## Calculation Spot-Checks

- Per-image rows: {len(metrics)}; sensitivity rows: {len(sensitivity)} ({sensitivity_groups}).
- Missing primary metrics: {missing}.
- Normalized direction entropy observed range: {entropy_range[0]:.6f} to {entropy_range[1]:.6f}; all finite values lie in [0,1].
- Canny edge density observed range: {edge_range[0]:.6f} to {edge_range[1]:.6f}; all values lie in [0,1].
- Omnibus tests: {len(omnibus)} rows; pairwise tests: {len(pairwise)} rows; Holm-adjusted p-values are present.
- Sensitivity tests: {len(sensitivity_omnibus)} omnibus rows and {len(sensitivity_pairwise)} pairwise rows, covering 3 Sobel thresholds, 3 Canny threshold pairs and 3 border settings.
- Summary counts reconcile to 568 for each primary metric.

## Issues Found

1. **High:** Candidate labels and eras are based partly on folder/file-prefix evidence and have not been signed off by a reviewer.
2. **High:** Group is completely or nearly completely confounded with source prefix; edge metrics may capture digitization or framing differences.
3. **High:** The analysis unit is image, while series, same-work and duplicate relations remain unresolved; inferential uncertainty may be understated.
4. **High:** The 1950s Taohuawu group has n=9 candidate images.
5. **Medium:** No validated content mask excludes frames, scales, paper background or blank margins. Border-exclusion sensitivity reduces but does not eliminate this risk.
6. **High:** Direction-entropy differences are not robust to excluding the outer 5% or 10% of each resized image; no stable substantive direction-entropy claim is warranted.
7. **High:** Canny density differences are parameter-robust in the checked grid, but remain inseparable from source/digitization confounding.

## Visualization Review

Both distribution panels start at zero, use identical group ordering, show sample sizes and display all per-image points. The QC figure uses a deterministic first-sample rule rather than outcome-based example selection.

## Required Caveats for Readers

- Label every table and figure “candidate-sample exploratory analysis”.
- Do not call the results formal, causal, work-level or representative of the three traditions.
- Do not infer line continuity, breakage rate, craft quality or aesthetic superiority.
- Re-run without changing primary parameters after the formal manifest is frozen.
"""
    output_path.write_text(text, encoding="utf-8")


def make_notebook(output_path: Path) -> None:
    def markdown(source: str) -> dict[str, Any]:
        return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.splitlines()]}

    def code(source: str) -> dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.splitlines()],
        }

    notebook = {
        "cells": [
            markdown("## tl;dr\nThis notebook is a rerunnable companion to the executed candidate-sample analysis. Results are exploratory, not a frozen-manifest analysis."),
            markdown("## Context & Methods\n### Key Assumptions\n- 459+9+100 Stage 1 candidate images are used exactly as authorized.\n- Image is the temporary statistical unit.\n- Primary thresholds are fixed in `candidate_line_parameters.json`."),
            code("from pathlib import Path\nimport pandas as pd\nOUTPUT = Path.cwd() / 'reports/formal_line_candidate_experiment_20260727'"),
            markdown("## Data"),
            code("metrics = pd.read_csv(OUTPUT / 'candidate_line_metrics_per_image.csv')\nmetrics.groupby('group_label').size()"),
            markdown("## Results"),
            code("summary = pd.read_csv(OUTPUT / 'candidate_line_metrics_summary.csv')\nsummary[['group_label','metric','n','mean','sd','median','q1','q3']]"),
            code("pairwise = pd.read_csv(OUTPUT / 'candidate_line_pairwise_tests.csv')\npairwise"),
            code("sensitivity_tests = pd.read_csv(OUTPUT / 'candidate_line_sensitivity_omnibus_tests.csv')\nsensitivity_tests"),
            code("from IPython.display import Image, display\ndisplay(Image(filename=str(OUTPUT / 'figure_line_metric_distributions.png')))"),
            markdown("## Takeaways\nUse `FULL_PAPER_SECTION.md` for the evidence-bounded Chinese manuscript text. The notebook intentionally does not infer line continuity or craft quality."),
            markdown("### Re-run command\n`python3 reports/formal_line_candidate_experiment_20260727/run_candidate_line_analysis.py --output-dir reports/formal_line_candidate_experiment_20260727 --overwrite`"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": platform.python_version()},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_chart_map(output_path: Path) -> None:
    output_path.write_text(
        """# Chart map

| Visual | Analytical question | Family/type | Grain | Fields | Supported claim | Palette | Output |
|---|---|---|---|---|---|---|---|
| Candidate metric distributions | How do the candidate-image distributions differ? | Distribution / box plot with jitter | candidate image | group, normalized direction entropy, Canny edge density | operational distribution differences only | blue/gold/orange plus neutrals | figure_line_metric_distributions.png |
| Edge-detection QC | What does each processing stage retain, including source artefacts? | deterministic image QC grid | one deterministic image per candidate group | RGB, Sobel magnitude, Canny overlay | algorithm visibility and masking limitations | grayscale/red overlay | figure_edge_detection_candidate_qc.png |
| Table 9 | What are the exact candidate-group means and SDs? | exact lookup table | candidate-image group summary | n, mean, SD | descriptive candidate-sample comparison | neutral/blue header | table9_candidate_exploratory.png |
""",
        encoding="utf-8",
    )


def make_artifact(
    output_path: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    omnibus: pd.DataFrame,
    pairwise: pd.DataFrame,
    table: pd.DataFrame,
    sensitivity_omnibus: pd.DataFrame,
) -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    compact_metrics = metrics[
        [
            "sample_id",
            "group_id",
            "group_label",
            "group_order",
            "source_prefix",
            "normalized_direction_entropy",
            "canny_edge_density",
            "sobel_strong_gradient_ratio",
            "valid_pixel_count",
        ]
    ].to_dict(orient="records")
    table_rows = []
    for _, row in table.iterrows():
        table_rows.append(
            {
                "candidate_group": row["candidate_group"],
                "n_images": int(row["n_images"]),
                "direction_entropy_mean_sd": row["normalized_direction_entropy_mean_sd"],
                "canny_edge_density_mean_sd": row["canny_edge_density_mean_sd_percent"],
            }
        )
    source_metrics = {
        "id": "src_metrics",
        "label": "候选图像逐图线条指标",
        "path": "candidate_line_metrics_per_image.csv",
        "query": {
            "language": "SQL",
            "engine": "DuckDB over the executed Python pipeline output",
            "sql": (
                "SELECT sample_id, group_id, group_label, group_order, source_prefix, "
                "normalized_direction_entropy, canny_edge_density, sobel_strong_gradient_ratio, "
                "valid_pixel_count FROM read_csv_auto('candidate_line_metrics_per_image.csv')"
            ),
            "description": "The chart snapshot selects reviewed columns from the per-image CSV produced by the reproducible Python/OpenCV pipeline.",
            "executed_at": generated_at,
            "filters": [
                "Exactly 459 Qing Gusu candidates, 9 1950s Taohuawu candidates, and 100 late-Qing Yangliuqing candidates",
                "Image is the exploratory analysis unit",
                "Sobel threshold=0.10; Canny thresholds=50/150; long edge=512px",
            ],
            "metric_definitions": [
                "Normalized direction entropy = -sum(p_k ln p_k)/ln(9), with 9 magnitude-weighted orientation bins",
                "Canny edge density = binary edge pixels divided by valid pixels",
            ],
            "tables_used": ["candidate_line_metrics_per_image.csv"],
        },
    }
    source_manifest = {
        "id": "src_manifest",
        "label": "Stage 1候选样本快照",
        "path": "analysis_manifest_snapshot.csv",
        "query": {
            "language": "SQL",
            "engine": "DuckDB over CSV manifest snapshot",
            "sql": (
                "SELECT * FROM read_csv_auto('analysis_manifest_snapshot.csv') "
                "WHERE proposed_group IN ('清代姑苏版画（候选，待确认）', "
                "'20世纪50年代桃花坞年画（候选，待确认）', '清末杨柳青年画（候选，待确认）')"
            ),
            "description": "Frozen copy of the 568-row candidate selection used for this exploratory run.",
            "executed_at": generated_at,
            "filters": ["Candidate-only scope; human review decisions remain unresolved"],
            "tables_used": ["analysis_manifest_snapshot.csv"],
        },
    }
    entropy_omnibus = omnibus[omnibus["metric"] == "normalized_direction_entropy"].iloc[0]
    edge_omnibus = omnibus[omnibus["metric"] == "canny_edge_density"].iloc[0]
    entropy_border_removed = sensitivity_omnibus[
        (sensitivity_omnibus["metric_family"] == "direction_entropy")
        & (sensitivity_omnibus["border_exclusion_fraction"] > 0)
    ]
    edge_sensitivity = sensitivity_omnibus[
        sensitivity_omnibus["metric_family"] == "canny_edge_density"
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "三类候选版画图像线条特征探索性复现",
            "description": "Sobel归一化方向熵与Canny边缘密度；候选样本，非正式manifest。",
            "generatedAt": generated_at,
            "sources": [source_manifest, source_metrics],
            "charts": [
                {
                    "id": "chart_entropy",
                    "title": "归一化方向熵分布",
                    "subtitle": "候选图像；0–1，越低表示梯度方向越集中",
                    "type": "boxPlot",
                    "intent": "distribution",
                    "question": "How does normalized direction entropy vary across the three candidate groups?",
                    "rationale": "A box plot shows median, spread and outliers under strongly unequal group sizes.",
                    "dataset": "per_image",
                    "sourceId": "src_metrics",
                    "encodings": {
                        "x": {"field": "group_label", "type": "nominal", "label": "候选组"},
                        "y": {"field": "normalized_direction_entropy", "type": "quantitative", "label": "归一化方向熵"},
                        "tooltip": [
                            {"field": "sample_id", "type": "text", "label": "sample_id"},
                            {"field": "source_prefix", "type": "text", "label": "来源前缀"},
                            {"field": "valid_pixel_count", "type": "quantitative", "label": "有效像素"},
                        ],
                    },
                    "layout": "full",
                    "maxRows": 568,
                },
                {
                    "id": "chart_edge_density",
                    "title": "Canny边缘密度分布",
                    "subtitle": "候选图像；有效区域内二值边缘像素比例",
                    "type": "boxPlot",
                    "intent": "distribution",
                    "question": "How does Canny edge density vary across the three candidate groups?",
                    "rationale": "A box plot makes the unequal-size candidate distributions comparable without reducing them to means alone.",
                    "dataset": "per_image",
                    "sourceId": "src_metrics",
                    "encodings": {
                        "x": {"field": "group_label", "type": "nominal", "label": "候选组"},
                        "y": {"field": "canny_edge_density", "type": "quantitative", "format": "percent", "label": "边缘密度"},
                        "tooltip": [
                            {"field": "sample_id", "type": "text", "label": "sample_id"},
                            {"field": "source_prefix", "type": "text", "label": "来源前缀"},
                            {"field": "sobel_strong_gradient_ratio", "type": "quantitative", "format": "percent", "label": "强梯度比例"},
                        ],
                    },
                    "layout": "full",
                    "maxRows": 568,
                },
            ],
            "tables": [
                {
                    "id": "table9",
                    "title": "表9 候选样本线条指标汇总",
                    "subtitle": "图像级均值±标准差；不是正式manifest结果",
                    "dataset": "table9",
                    "sourceId": "src_metrics",
                    "defaultSort": {"field": "candidate_group", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "candidate_group", "label": "候选组", "type": "text"},
                        {"field": "n_images", "label": "n（图像）", "type": "number"},
                        {"field": "direction_entropy_mean_sd", "label": "归一化方向熵（0–1）", "type": "text"},
                        {"field": "canny_edge_density_mean_sd", "label": "Canny边缘密度", "type": "text"},
                    ],
                }
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# 三类候选版画图像线条特征探索性复现", "layout": "full"},
                {
                    "id": "summary",
                    "type": "markdown",
                    "body": (
                        "## 技术摘要\n\n本报告按用户授权使用459+9+100张Stage 1候选图像，重新计算Sobel归一化方向熵和Canny边缘密度，"
                        f"不沿用旧稿数字。方向熵总体检验H={entropy_omnibus['statistic_H']:.3f}（{p_text(entropy_omnibus['p_value_holm_across_metrics'])}）；"
                        f"边缘密度总体检验H={edge_omnibus['statistic_H']:.3f}（{p_text(edge_omnibus['p_value_holm_across_metrics'])}）。"
                        "方向熵差异在排除边界5%或10%后不再显著；Canny密度差异在9个预设敏感性组合中均为p<0.001。"
                        "由于候选标签、作品关系和数字化来源未冻结，结论只能作为探索性证据。"
                    ),
                    "layout": "full",
                    "sourceId": "src_metrics",
                },
                {"id": "finding_entropy", "type": "markdown", "body": "## 梯度方向分布\n\n归一化方向熵只描述梯度方向的集中或分散，不等同于线条规整、连续或工艺水平。", "layout": "full"},
                {"id": "entropy_chart", "type": "chart", "chartId": "chart_entropy", "layout": "full"},
                {"id": "finding_edges", "type": "markdown", "body": "## 二值边缘比例\n\nCanny边缘密度是二值边缘像素比例，可能受画框、标尺、留白、压缩和扫描清晰度影响。", "layout": "full"},
                {"id": "edge_chart", "type": "chart", "chartId": "chart_edge_density", "layout": "full"},
                {"id": "table_heading", "type": "markdown", "body": "## 图像级汇总结果\n\n下表用于精确查阅三组候选图像的均值和标准差。", "layout": "full"},
                {"id": "table_block", "type": "table", "tableId": "table9", "layout": "full"},
                {
                    "id": "sensitivity",
                    "type": "markdown",
                    "body": (
                        "## 敏感性分析\n\n方向熵对边界处理不稳健：排除四周5%或10%后的6个总体检验"
                        f"p值范围为{entropy_border_removed['p_value'].min():.3f}—{entropy_border_removed['p_value'].max():.3f}，均未达到0.05。"
                        "因此不形成稳定的方向熵风格结论。Canny边缘密度在3组阈值×3种边界设置的9个组合中"
                        f"均为p<0.001（H={edge_sensitivity['statistic_H'].min():.3f}—{edge_sensitivity['statistic_H'].max():.3f}），"
                        "但来源混杂仍未消除。"
                    ),
                    "layout": "full",
                    "sourceId": "src_metrics",
                },
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": "## 范围与指标定义\n\n候选池包括清代姑苏459张、20世纪50年代桃花坞9张、清末杨柳青100张。图像是临时统计单位。方向熵采用9个20°方向区间并除以log(9)；Canny边缘密度为边缘像素数除以有效像素数。",
                    "layout": "full",
                    "sourceId": "src_manifest",
                },
                {
                    "id": "methods",
                    "type": "markdown",
                    "body": "## 方法与统计设计\n\n图像等比例缩放至长边512像素、不补边，使用BT.709灰度、Sobel 3×3和Canny 50/150。主Sobel阈值为0.10。总体比较使用Kruskal–Wallis检验，两两比较使用Mann–Whitney U检验和Holm校正，并报告秩二列相关。另检查Sobel阈值、Canny阈值和边界裁除敏感性。",
                    "layout": "full",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": "## 限制、不确定性与稳健性\n\n桃花坞仅9张；组别与来源前缀完全或近乎完全共线；同作品、组画和重复关系未解决；没有经过验证的内容掩膜。因此统计显著性不能解释为地域风格的因果效应，也不能证明线条连续、刻版精细或审美优越。",
                    "layout": "full",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": "## 建议的下一步\n\n冻结人工manifest后保持主参数不变重跑；以独立作品/版次为统计单位；建立内容掩膜QC；按数字化来源和采集批次做分层或敏感性分析。",
                    "layout": "full",
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": "## 后续问题\n\n桃花坞能否补充至少10件独立作品？是否存在共同数字化来源的跨组样本？哪些组画成员应视为同一作品？这些问题决定结果能否升级为正式论文证据。",
                    "layout": "full",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "per_image": compact_metrics,
                "table9": table_rows,
                "summary": summary.to_dict(orient="records"),
                "omnibus": omnibus.to_dict(orient="records"),
                "pairwise": pairwise.to_dict(orient="records"),
                "sensitivity_omnibus": sensitivity_omnibus.to_dict(orient="records"),
            },
            "accessIssues": [],
        },
        "sources": [source_manifest, source_metrics],
    }
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else (PROJECT_ROOT / args.manifest)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "analysis outputs already exist; pass --overwrite to replace only this experiment's outputs:\n"
            + "\n".join(str(path) for path in existing)
        )

    manifest = load_candidate_manifest(manifest_path)
    snapshot_columns = [
        "sample_id",
        "image_path",
        "sha256",
        "proposed_group",
        "era",
        "era_basis",
        "collection",
        "source_prefix",
        "work_id",
        "series_id",
        "same_block_id",
        "duplicate_group_id",
        "review_status",
        "include_in_analysis",
        "group_id",
        "group_label",
        "group_order",
    ]
    manifest[snapshot_columns].to_csv(
        output_dir / "analysis_manifest_snapshot.csv", index=False, encoding="utf-8-sig"
    )
    metrics, sensitivity, qc_products = calculate_candidate_metrics(manifest)
    metrics.to_csv(output_dir / "candidate_line_metrics_per_image.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(
        output_dir / "candidate_line_sensitivity_per_image.csv", index=False, encoding="utf-8-sig"
    )

    primary_metrics = ("normalized_direction_entropy", "canny_edge_density")
    summary = summarize_metric_rows(metrics, primary_metrics)
    omnibus, pairwise = inferential_tests(metrics, primary_metrics)
    sensitivity_summary_frame = sensitivity_summary(sensitivity)
    sensitivity_omnibus, sensitivity_pairwise = sensitivity_inferential_tests(sensitivity)
    prefix = prefix_summary(metrics)
    summary.to_csv(output_dir / "candidate_line_metrics_summary.csv", index=False, encoding="utf-8-sig")
    sensitivity_summary_frame.to_csv(
        output_dir / "candidate_line_sensitivity_summary.csv", index=False, encoding="utf-8-sig"
    )
    sensitivity_omnibus.to_csv(
        output_dir / "candidate_line_sensitivity_omnibus_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensitivity_pairwise.to_csv(
        output_dir / "candidate_line_sensitivity_pairwise_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )
    omnibus.to_csv(output_dir / "candidate_line_omnibus_tests.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(output_dir / "candidate_line_pairwise_tests.csv", index=False, encoding="utf-8-sig")
    prefix.to_csv(output_dir / "candidate_source_prefix_summary.csv", index=False, encoding="utf-8-sig")

    table = make_table_outputs(
        summary,
        output_dir / "table9_candidate_exploratory.csv",
        output_dir / "table9_candidate_exploratory.png",
    )
    make_distribution_figure(metrics, output_dir / "figure_line_metric_distributions.png")
    make_qc_figure(qc_products, output_dir / "figure_edge_detection_candidate_qc.png")
    make_parameters(output_dir / "candidate_line_parameters.json", manifest_path)
    make_paper_section(
        output_dir / "FULL_PAPER_SECTION.md",
        table,
        omnibus,
        pairwise,
        sensitivity_omnibus,
    )
    make_validation_report(
        output_dir / "VALIDATION_REPORT.md",
        manifest,
        metrics,
        sensitivity,
        omnibus,
        pairwise,
        sensitivity_omnibus,
        sensitivity_pairwise,
    )
    make_notebook(output_dir / "candidate_line_analysis.ipynb")
    make_chart_map(output_dir / "chart_map.md")
    make_artifact(
        output_dir / "artifact.json",
        metrics,
        summary,
        omnibus,
        pairwise,
        table,
        sensitivity_omnibus,
    )

    receipt = {
        "status": "completed_candidate_exploratory_not_formal_manifest",
        "rows": len(metrics),
        "group_counts": metrics.groupby("group_id").size().to_dict(),
        "outputs": [str(output_dir / name) for name in OUTPUT_FILENAMES],
    }
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
