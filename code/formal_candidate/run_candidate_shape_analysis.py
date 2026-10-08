#!/usr/bin/env python3
"""Reproducible candidate-level shape and edge-contour experiment.

This script analyzes the existing 459 + 9 + 100 candidate-image manifest.
It does not treat the candidate manifest as a frozen formal sample, does not
infer human-body proportions, and does not use the manuscript's old Table 11
values as computational targets.

The primary contour-complexity metric is the box-counting dimension of a Canny
edge map. It is a whole-image edge-contour descriptor, not a segmented-object
or human-silhouette descriptor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gusu_shape_candidate_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gusu_shape_candidate_cache")

import cv2
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.font_manager import FontProperties
from scipy import stats


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / (
    "reports/formal_aesthetics_reproduction_20260727/formal_analysis_manifest.csv"
)

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
GROUP_SHORT_CN = {
    "gusu_qing_candidate": "姑苏版画",
    "taohuawu_1950s_candidate": "桃花坞版画",
    "yangliuqing_late_qing_candidate": "杨柳青年画",
}
GROUP_SHORT_EN = {
    "gusu_qing_candidate": "Gusu",
    "taohuawu_1950s_candidate": "Taohuawu",
    "yangliuqing_late_qing_candidate": "Yangliuqing",
}

LONG_EDGE = 768
BORDER_EXCLUSION_FRACTIONS = (0.01, 0.03, 0.05)
PRIMARY_BORDER_EXCLUSION = 0.03
CANNY_SETTINGS = ((40, 120), (50, 150), (60, 180))
PRIMARY_CANNY = (50, 150)
GAUSSIAN_KERNEL = 5
GAUSSIAN_SIGMA = 1.0
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
HOG_ORIENTATION_BINS = 9
BOX_SIZES_PX = (4, 8, 16, 32, 64, 128)
MIN_EDGE_PIXELS_FOR_BOX_FIT = 100
MIN_CONTOUR_PERIMETER_PX = 10.0
BOOTSTRAP_ITERATIONS = 10_000
RANDOM_SEED = 20260730

PRIMARY_METRICS = (
    "hog_style_orientation_concentration",
    "sobel_energy_horizontal_symmetry",
    "edge_contour_box_counting_dimension",
    "normalized_total_contour_length",
    "contour_component_density_per_megapixel",
    "length_weighted_contour_irregularity",
    "canny_edge_density",
)

OUTPUT_FILES = (
    "analysis_manifest_snapshot.csv",
    "shape_metrics_per_image.csv",
    "contour_complexity_sensitivity_per_image.csv",
    "shape_metrics_summary.csv",
    "contour_complexity_sensitivity_summary.csv",
    "contour_complexity_sensitivity_correlations.csv",
    "shape_exploratory_omnibus_tests.csv",
    "shape_exploratory_pairwise_tests.csv",
    "table11_candidate_exploratory.csv",
    "table11_old_vs_reproduced_audit.csv",
    "figure_shape_metric_distributions.png",
    "figure_contour_pipeline_qc.png",
    "table11_candidate_exploratory.png",
    "shape_parameters.json",
    "shape_run_receipt.json",
    "requirements_frozen.txt",
    "shape_candidate_analysis.ipynb",
    "README.md",
    "TABLE11_VALIDATION_REPORT.md",
    "FULL_PAPER_SECTION_SHAPE_CANDIDATE.md",
    "chart_map.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image_rgb(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot_read_image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_long_edge(image: np.ndarray, long_edge: int = LONG_EDGE) -> np.ndarray:
    height, width = image.shape[:2]
    scale = long_edge / max(height, width)
    new_width = max(2, int(round(width * scale)))
    new_height = max(2, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def rgb_to_bt709_gray(image_rgb: np.ndarray) -> np.ndarray:
    rgb = image_rgb.astype(np.float32)
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def crop_border(array: np.ndarray, border_fraction: float) -> np.ndarray:
    height, width = array.shape[:2]
    margin_y = max(1, int(round(height * border_fraction)))
    margin_x = max(1, int(round(width * border_fraction)))
    if height <= 2 * margin_y + 8 or width <= 2 * margin_x + 8:
        raise ValueError(f"image_too_small_after_border_crop: {array.shape}")
    return array[margin_y : height - margin_y, margin_x : width - margin_x]


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
    counts = frame.groupby("group_id", sort=False).size().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"candidate_group_counts_mismatch: {counts}")
    frame["resolved_image_path"] = frame["image_path"].map(
        lambda value: str((PROJECT_ROOT / value).resolve())
        if not Path(value).is_absolute()
        else value
    )
    return frame


def sobel_products(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)
    magnitude = cv2.magnitude(gx, gy)
    orientation_deg = np.mod(np.degrees(np.arctan2(gy, gx)), 180.0)
    return magnitude, orientation_deg


def normalize_local_contrast(gray: np.ndarray) -> np.ndarray:
    """Apply the same deterministic local-contrast normalization to every image."""
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE
    )
    return clahe.apply(gray)


def hog_style_orientation_metrics(
    magnitude: np.ndarray, orientation_deg: np.ndarray
) -> tuple[float, float, list[float]]:
    """Aggregate a 9-bin unsigned, magnitude-weighted HOG-style histogram.

    The scalar concentration is max(p_k)/(1/9) = 9*max(p_k). It is not a
    trained HOG detector and does not include block normalization because the
    requested output is one global orientation-distribution summary per image.
    """
    weights = magnitude.astype(np.float64, copy=False).ravel()
    angles = orientation_deg.astype(np.float64, copy=False).ravel()
    positive = weights > 0
    weights = weights[positive]
    angles = angles[positive]
    if weights.size == 0 or float(weights.sum()) <= 0:
        return float("nan"), float("nan"), [float("nan")] * HOG_ORIENTATION_BINS
    bin_width = 180.0 / HOG_ORIENTATION_BINS
    position = angles / bin_width - 0.5
    lower_raw = np.floor(position).astype(int)
    fraction = position - lower_raw
    lower = np.mod(lower_raw, HOG_ORIENTATION_BINS)
    upper = np.mod(lower_raw + 1, HOG_ORIENTATION_BINS)
    histogram = np.bincount(
        lower, weights=weights * (1.0 - fraction), minlength=HOG_ORIENTATION_BINS
    ).astype(np.float64)
    histogram += np.bincount(
        upper, weights=weights * fraction, minlength=HOG_ORIENTATION_BINS
    )
    probabilities = histogram / histogram.sum()
    concentration = HOG_ORIENTATION_BINS * float(probabilities.max())
    nonzero = probabilities > 0
    entropy = -float(np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))
    normalized_entropy = entropy / math.log(HOG_ORIENTATION_BINS)
    return concentration, normalized_entropy, probabilities.tolist()


def horizontal_symmetry_from_energy(magnitude: np.ndarray) -> float:
    """Fixed-axis left-right symmetry of a robustly scaled Sobel energy map."""
    positive = magnitude[magnitude > 0]
    if positive.size == 0:
        return float("nan")
    scale = float(np.percentile(positive, 99.0))
    if scale <= 0:
        return float("nan")
    energy = np.clip(magnitude.astype(np.float64) / scale, 0.0, 1.0)
    half_width = energy.shape[1] // 2
    if half_width < 2:
        return float("nan")
    left = energy[:, :half_width]
    right = np.fliplr(energy[:, -half_width:])
    denominator = float(np.sum(left + right))
    if denominator <= 0:
        return float("nan")
    value = 1.0 - float(np.sum(np.abs(left - right))) / denominator
    return float(np.clip(value, 0.0, 1.0))


def canny_edges(gray: np.ndarray, low: int, high: int) -> np.ndarray:
    blurred = cv2.GaussianBlur(
        gray,
        (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL),
        sigmaX=GAUSSIAN_SIGMA,
        sigmaY=GAUSSIAN_SIGMA,
        borderType=cv2.BORDER_REFLECT101,
    )
    return cv2.Canny(
        blurred,
        threshold1=low,
        threshold2=high,
        apertureSize=3,
        L2gradient=True,
    ) > 0


def box_counting_dimension(edge_map: np.ndarray) -> tuple[float, float, int, list[dict[str, float]]]:
    """Fit log N(s) against log(1/s) for occupied boxes on a binary edge map."""
    height, width = edge_map.shape
    observations: list[dict[str, float]] = []
    for size in BOX_SIZES_PX:
        if size > min(height, width) // 2:
            continue
        padded_height = int(math.ceil(height / size) * size)
        padded_width = int(math.ceil(width / size) * size)
        padded = np.zeros((padded_height, padded_width), dtype=bool)
        padded[:height, :width] = edge_map
        occupied = padded.reshape(
            padded_height // size, size, padded_width // size, size
        ).any(axis=(1, 3))
        count = int(occupied.sum())
        total_boxes = int(occupied.size)
        if 1 < count < total_boxes:
            observations.append(
                {
                    "box_size_px": float(size),
                    "occupied_box_count": float(count),
                    "total_box_count": float(total_boxes),
                }
            )
    if int(edge_map.sum()) < MIN_EDGE_PIXELS_FOR_BOX_FIT or len(observations) < 3:
        return float("nan"), float("nan"), len(observations), observations
    x = np.log([1.0 / row["box_size_px"] for row in observations])
    y = np.log([row["occupied_box_count"] for row in observations])
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    ss_residual = float(np.sum((y - fitted) ** 2))
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else float("nan")
    return float(slope), float(r_squared), len(observations), observations


def contour_metrics(edge_map: np.ndarray) -> dict[str, float | int]:
    edge_u8 = edge_map.astype(np.uint8) * 255
    contours, _ = cv2.findContours(edge_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeters: list[float] = []
    irregularities: list[float] = []
    for contour in contours:
        perimeter = float(cv2.arcLength(contour, closed=True))
        if perimeter < MIN_CONTOUR_PERIMETER_PX:
            continue
        hull = cv2.convexHull(contour)
        hull_perimeter = float(cv2.arcLength(hull, closed=True))
        if hull_perimeter <= 0:
            continue
        perimeters.append(perimeter)
        irregularities.append(perimeter / hull_perimeter)
    valid_area = int(edge_map.size)
    total_perimeter = float(np.sum(perimeters)) if perimeters else 0.0
    weighted_irregularity = (
        float(np.average(irregularities, weights=perimeters)) if perimeters else float("nan")
    )
    dimension, r_squared, scale_count, _ = box_counting_dimension(edge_map)
    return {
        "edge_contour_box_counting_dimension": dimension,
        "box_count_fit_r_squared": r_squared,
        "box_count_scale_count": scale_count,
        "canny_edge_pixel_count": int(edge_map.sum()),
        "canny_edge_density": float(edge_map.mean()),
        "retained_contour_component_count": int(len(perimeters)),
        "total_contour_perimeter_px": total_perimeter,
        "normalized_total_contour_length": total_perimeter / math.sqrt(valid_area),
        "contour_component_density_per_megapixel": len(perimeters) / (valid_area / 1_000_000.0),
        "length_weighted_contour_irregularity": weighted_irregularity,
    }


def compute_metrics(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for index, row in manifest.iterrows():
        image_path = Path(row["resolved_image_path"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        observed_sha = sha256_file(image_path)
        if observed_sha.lower() != row["sha256"].lower():
            raise ValueError(f"sha256_mismatch: {row['sample_id']} {image_path}")
        original = read_image_rgb(image_path)
        resized_rgb = resize_long_edge(original)
        resized_gray = rgb_to_bt709_gray(resized_rgb)
        for border_fraction in BORDER_EXCLUSION_FRACTIONS:
            gray = normalize_local_contrast(crop_border(resized_gray, border_fraction))
            magnitude, orientation = sobel_products(gray)
            hog_concentration, hog_entropy, hog_probabilities = hog_style_orientation_metrics(
                magnitude, orientation
            )
            symmetry = horizontal_symmetry_from_energy(magnitude)
            for low, high in CANNY_SETTINGS:
                edges = canny_edges(gray, low, high)
                contour = contour_metrics(edges)
                sensitivity_row: dict[str, Any] = {
                    "sample_id": row["sample_id"],
                    "group_id": row["group_id"],
                    "group_label": row["group_label"],
                    "group_order": int(row["group_order"]),
                    "source_prefix": row["source_prefix"],
                    "border_exclusion_fraction": border_fraction,
                    "canny_low_threshold": low,
                    "canny_high_threshold": high,
                    "analysis_width": int(gray.shape[1]),
                    "analysis_height": int(gray.shape[0]),
                    **contour,
                }
                sensitivity_rows.append(sensitivity_row)
                if math.isclose(border_fraction, PRIMARY_BORDER_EXCLUSION) and (
                    low,
                    high,
                ) == PRIMARY_CANNY:
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
                            "duplicate_group_id_candidate": row.get("duplicate_group_id", ""),
                            "original_width": int(original.shape[1]),
                            "original_height": int(original.shape[0]),
                            "analysis_width": int(gray.shape[1]),
                            "analysis_height": int(gray.shape[0]),
                            "hog_style_orientation_concentration": hog_concentration,
                            "hog_style_normalized_orientation_entropy": hog_entropy,
                            **{
                                f"hog_orientation_bin_{bin_index + 1:02d}_probability": probability
                                for bin_index, probability in enumerate(hog_probabilities)
                            },
                            "sobel_energy_horizontal_symmetry": symmetry,
                            **contour,
                            "analysis_scope": "candidate_exploratory_not_formal_manifest",
                            "human_body_keypoints_computed": False,
                            "segmented_object_contour_computed": False,
                        }
                    )
        if (index + 1) % 25 == 0 or index + 1 == len(manifest):
            print(f"processed {index + 1}/{len(manifest)}", flush=True)
    primary = pd.DataFrame(primary_rows).sort_values(
        ["group_order", "sample_id"]
    ).reset_index(drop=True)
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values(
        ["group_order", "sample_id", "border_exclusion_fraction", "canny_low_threshold"]
    ).reset_index(drop=True)
    return primary, sensitivity


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    sampled = rng.choice(values, size=(BOOTSTRAP_ITERATIONS, len(values)), replace=True)
    low, high = np.percentile(sampled.mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def summarize(frame: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
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
                    "group_order": int(subset["group_order"].iloc[0]),
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


def summarize_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "edge_contour_box_counting_dimension",
        "normalized_total_contour_length",
        "contour_component_density_per_megapixel",
        "length_weighted_contour_irregularity",
        "canny_edge_density",
    )
    rows: list[dict[str, Any]] = []
    for (group_id, group_label, group_order, border, low, high), subset in frame.groupby(
        [
            "group_id",
            "group_label",
            "group_order",
            "border_exclusion_fraction",
            "canny_low_threshold",
            "canny_high_threshold",
        ],
        sort=False,
    ):
        for metric in metrics:
            values = subset[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "group_id": group_id,
                    "group_label": group_label,
                    "group_order": int(group_order),
                    "border_exclusion_fraction": border,
                    "canny_low_threshold": int(low),
                    "canny_high_threshold": int(high),
                    "metric": metric,
                    "n": int(len(values)),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                }
            )
    return pd.DataFrame(rows)


def sensitivity_correlations(primary: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "edge_contour_box_counting_dimension",
        "normalized_total_contour_length",
        "length_weighted_contour_irregularity",
    )
    primary_indexed = primary.set_index("sample_id")
    rows: list[dict[str, Any]] = []
    for (border, low, high), subset in sensitivity.groupby(
        ["border_exclusion_fraction", "canny_low_threshold", "canny_high_threshold"]
    ):
        aligned = subset.set_index("sample_id")
        for metric in metrics:
            first = primary_indexed[metric]
            second = aligned[metric].reindex(first.index)
            valid = first.notna() & second.notna()
            correlation, p_value = stats.spearmanr(first[valid], second[valid])
            rows.append(
                {
                    "metric": metric,
                    "border_exclusion_fraction": border,
                    "canny_low_threshold": int(low),
                    "canny_high_threshold": int(high),
                    "n": int(valid.sum()),
                    "spearman_rho_vs_primary": float(correlation),
                    "p_value_unadjusted": float(p_value),
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


def exploratory_tests(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = (
        "hog_style_orientation_concentration",
        "sobel_energy_horizontal_symmetry",
        "edge_contour_box_counting_dimension",
    )
    omnibus_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    for metric in metrics:
        arrays = [
            frame.loc[frame.group_id == group_id, metric].dropna().to_numpy(dtype=float)
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
                "analysis_unit": "candidate_image_not_independent_work",
            }
        )
        current_pairs: list[dict[str, Any]] = []
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
                current_pairs.append(
                    {
                        "metric": metric,
                        "group_a": group_a,
                        "group_b": group_b,
                        "n_a": len(values_a),
                        "n_b": len(values_b),
                        "median_a": float(np.median(values_a)),
                        "median_b": float(np.median(values_b)),
                        "U_statistic": float(result.statistic),
                        "p_value_raw": float(result.pvalue),
                        "rank_biserial_a_higher_positive": rank_biserial,
                    }
                )
        adjusted = holm_adjust([row["p_value_raw"] for row in current_pairs])
        for row, adjusted_value in zip(current_pairs, adjusted):
            row["p_value_holm_within_metric"] = adjusted_value
            row["significant_holm_0_05"] = adjusted_value < 0.05
            pairwise_rows.append(row)
    omnibus = pd.DataFrame(omnibus_rows)
    omnibus["p_value_holm_across_metrics"] = holm_adjust(omnibus.p_value.tolist())
    omnibus["significant_holm_0_05"] = omnibus.p_value_holm_across_metrics < 0.05
    return omnibus, pd.DataFrame(pairwise_rows)


def metric_summary_lookup(summary: pd.DataFrame, group_id: str, metric: str) -> pd.Series:
    rows = summary[(summary.group_id == group_id) & (summary.metric == metric)]
    if len(rows) != 1:
        raise ValueError(f"summary_lookup_failed: {group_id} {metric} rows={len(rows)}")
    return rows.iloc[0]


def make_table11(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_id in GROUP_ORDER:
        hog = metric_summary_lookup(summary, group_id, "hog_style_orientation_concentration")
        symmetry = metric_summary_lookup(summary, group_id, "sobel_energy_horizontal_symmetry")
        contour = metric_summary_lookup(summary, group_id, "edge_contour_box_counting_dimension")
        rows.append(
            {
                "candidate_group": GROUP_SHORT_CN[group_id],
                "n_candidate_images": int(hog.n),
                "hog_style_direction_concentration_mean": hog["mean"],
                "hog_style_direction_concentration_sd": hog.sd,
                "hog_style_direction_concentration_mean_sd": f"{hog['mean']:.3f} ± {hog.sd:.3f}",
                "sobel_energy_horizontal_symmetry_mean": symmetry["mean"],
                "sobel_energy_horizontal_symmetry_sd": symmetry.sd,
                "sobel_energy_horizontal_symmetry_mean_sd": f"{symmetry['mean']:.3f} ± {symmetry.sd:.3f}",
                "edge_contour_box_counting_dimension_mean": contour["mean"],
                "edge_contour_box_counting_dimension_sd": contour.sd,
                "edge_contour_box_counting_dimension_mean_sd": f"{contour['mean']:.3f} ± {contour.sd:.3f}",
                "analysis_scope": "candidate_exploratory_not_formal_manifest",
            }
        )
    return pd.DataFrame(rows)


def audit_old_table(table11: pd.DataFrame) -> pd.DataFrame:
    old = {
        "姑苏版画": {"hog_mean": 2.31, "hog_sd": 1.01, "symmetry_mean": 0.864, "symmetry_sd": 0.041},
        "桃花坞版画": {"hog_mean": 1.81, "hog_sd": 0.12, "symmetry_mean": 0.775, "symmetry_sd": 0.026},
        "杨柳青年画": {"hog_mean": 1.87, "hog_sd": 0.23, "symmetry_mean": 0.832, "symmetry_sd": 0.041},
    }
    rows: list[dict[str, Any]] = []
    for _, row in table11.iterrows():
        group = row.candidate_group
        rows.extend(
            [
                {
                    "candidate_group": group,
                    "metric": "HOG方向集中度",
                    "old_manuscript_mean": old[group]["hog_mean"],
                    "old_manuscript_sd": old[group]["hog_sd"],
                    "reproduced_mean": row.hog_style_direction_concentration_mean,
                    "reproduced_sd": row.hog_style_direction_concentration_sd,
                    "numeric_match": False,
                    "audit_status": "not_reproducible_old_formula_and_code_missing",
                    "reason": "旧稿无HOG参数、方向集中度公式或逐图结果；新值来自预注册的9方向幅值加权全局HOG式直方图，不能倒推旧值。",
                },
                {
                    "candidate_group": group,
                    "metric": "水平对称度",
                    "old_manuscript_mean": old[group]["symmetry_mean"],
                    "old_manuscript_sd": old[group]["symmetry_sd"],
                    "reproduced_mean": row.sobel_energy_horizontal_symmetry_mean,
                    "reproduced_sd": row.sobel_energy_horizontal_symmetry_sd,
                    "numeric_match": False,
                    "audit_status": "not_reproducible_old_input_domain_missing",
                    "reason": "旧稿未说明基于原像素、边缘图或显著性图；新值基于固定中轴的Sobel能量图，定义不同，不能把差异解释为程序错误。",
                },
            ]
        )
    return pd.DataFrame(rows)


def configure_font() -> FontProperties:
    candidates = (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            prop = FontProperties(fname=candidate)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return prop
    return FontProperties()


def create_distribution_figure(primary: pd.DataFrame, output_dir: Path) -> None:
    configure_font()
    metrics = [
        ("hog_style_orientation_concentration", "HOG-style orientation concentration", "ratio to uniform peak"),
        ("sobel_energy_horizontal_symmetry", "Horizontal symmetry of Sobel energy", "0–1"),
        ("edge_contour_box_counting_dimension", "Edge-contour box-counting dimension", "D_box"),
    ]
    colors = ["#3568A8", "#C58A24", "#768A3A"]
    rng = np.random.default_rng(RANDOM_SEED)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    for axis, (metric, title, ylabel) in zip(axes, metrics):
        arrays = [
            primary.loc[primary.group_id == group_id, metric].dropna().to_numpy(float)
            for group_id in GROUP_ORDER
        ]
        box = axis.boxplot(
            arrays,
            positions=np.arange(1, 4),
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#20252B", "linewidth": 1.5},
            whiskerprops={"color": "#67727E"},
            capprops={"color": "#67727E"},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
            patch.set_edgecolor(color)
        for position, values, color in zip(np.arange(1, 4), arrays, colors):
            jitter = rng.normal(0, 0.055, size=len(values))
            axis.scatter(
                position + jitter,
                values,
                s=11 if len(values) > 20 else 24,
                alpha=0.28 if len(values) > 20 else 0.75,
                color=color,
                edgecolors="none",
                rasterized=True,
            )
        axis.set_title(title, fontsize=11, color="#20252B")
        axis.set_ylabel(ylabel, fontsize=9)
        axis.set_xticks([1, 2, 3])
        axis.set_xticklabels(
            [f"{GROUP_SHORT_EN[g]}\n(n={EXPECTED_COUNTS[g]})" for g in GROUP_ORDER], fontsize=9
        )
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Candidate-image shape metrics (459 + 9 + 100; descriptive only)",
        fontsize=14,
        color="#20252B",
    )
    fig.savefig(output_dir / "figure_shape_metric_distributions.png", dpi=220, facecolor="white")
    plt.close(fig)


def create_qc_figure(primary: pd.DataFrame, manifest: pd.DataFrame, output_dir: Path) -> None:
    configure_font()
    selected_ids: list[str] = []
    for group_id in GROUP_ORDER:
        subset = primary[primary.group_id == group_id].sort_values(
            "edge_contour_box_counting_dimension"
        )
        selected_ids.append(subset.iloc[len(subset) // 2].sample_id)
    manifest_by_id = manifest.set_index("sample_id")
    primary_by_id = primary.set_index("sample_id")
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 13.2), constrained_layout=True)
    for row_index, sample_id in enumerate(selected_ids):
        manifest_row = manifest_by_id.loc[sample_id]
        metric_row = primary_by_id.loc[sample_id]
        rgb = resize_long_edge(read_image_rgb(Path(manifest_row.resolved_image_path)))
        cropped_rgb = crop_border(rgb, PRIMARY_BORDER_EXCLUSION)
        gray = normalize_local_contrast(rgb_to_bt709_gray(cropped_rgb))
        edges = canny_edges(gray, *PRIMARY_CANNY)
        contours, _ = cv2.findContours(
            edges.astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        overlay = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay, contours, -1, (30, 70, 220), 1, cv2.LINE_AA)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        axes[row_index, 0].imshow(cropped_rgb)
        axes[row_index, 1].imshow(edges, cmap="gray")
        axes[row_index, 2].imshow(overlay)
        axes[row_index, 0].set_title(
            f"{GROUP_SHORT_EN[manifest_row.group_id]} | {sample_id}\n3% inset RGB",
            fontsize=10,
        )
        axes[row_index, 1].set_title(
            f"CLAHE + Canny 50/150 | edge density={metric_row.canny_edge_density:.3f}", fontsize=10
        )
        axes[row_index, 2].set_title(
            f"Retained contours | D_box={metric_row.edge_contour_box_counting_dimension:.3f}",
            fontsize=10,
        )
        for axis in axes[row_index]:
            axis.axis("off")
    fig.suptitle(
        "Edge-contour extraction quality-control examples (median D_box per candidate group)",
        fontsize=14,
        color="#20252B",
    )
    fig.savefig(output_dir / "figure_contour_pipeline_qc.png", dpi=220, facecolor="white")
    plt.close(fig)


def create_table_figure(table11: pd.DataFrame, output_dir: Path) -> None:
    font = configure_font()
    display = table11[
        [
            "candidate_group",
            "hog_style_direction_concentration_mean_sd",
            "sobel_energy_horizontal_symmetry_mean_sd",
            "edge_contour_box_counting_dimension_mean_sd",
        ]
    ].copy()
    display.columns = [
        "版画类别（候选）",
        "HOG式方向集中度",
        "Sobel能量水平对称度",
        "边缘轮廓盒计数维数",
    ]
    fig, axis = plt.subplots(figsize=(12.5, 3.25))
    axis.axis("off")
    table = axis.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.20, 0.25, 0.27, 0.28],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.05)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#4A5560")
        cell.set_linewidth(0.8)
        cell.get_text().set_fontproperties(font)
        if row == 0:
            cell.set_facecolor("#E9EEF5")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#F7F9FB")
    fig.suptitle(
        "表11（候选样本探索性复现）：形状与边缘轮廓特征",
        fontsize=15,
        fontproperties=font,
        y=0.98,
    )
    fig.text(
        0.5,
        0.03,
        "数值为候选图像间均值 ± 样本标准差；n=459/9/100；尚非正式作品级manifest。",
        ha="center",
        fontsize=10,
        fontproperties=font,
        color="#4A5560",
    )
    fig.savefig(output_dir / "table11_candidate_exploratory.png", dpi=220, facecolor="white")
    plt.close(fig)


def markdown_table11(table11: pd.DataFrame) -> str:
    lines = [
        "| 候选组 | n | HOG式方向集中度 | Sobel能量水平对称度 | 边缘轮廓盒计数维数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in table11.iterrows():
        lines.append(
            f"| {row.candidate_group} | {int(row.n_candidate_images)} | "
            f"{row.hog_style_direction_concentration_mean_sd} | "
            f"{row.sobel_energy_horizontal_symmetry_mean_sd} | "
            f"{row.edge_contour_box_counting_dimension_mean_sd} |"
        )
    return "\n".join(lines)


def write_reports(
    output_dir: Path,
    primary: pd.DataFrame,
    summary: pd.DataFrame,
    sensitivity_correlations_frame: pd.DataFrame,
    table11: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    table_md = markdown_table11(table11)
    contour_means = {
        group_id: metric_summary_lookup(
            summary, group_id, "edge_contour_box_counting_dimension"
        )["mean"]
        for group_id in GROUP_ORDER
    }
    hog_means = {
        group_id: metric_summary_lookup(summary, group_id, "hog_style_orientation_concentration")[
            "mean"
        ]
        for group_id in GROUP_ORDER
    }
    symmetry_means = {
        group_id: metric_summary_lookup(summary, group_id, "sobel_energy_horizontal_symmetry")[
            "mean"
        ]
        for group_id in GROUP_ORDER
    }
    highest_contour = max(contour_means, key=contour_means.get)
    highest_hog = max(hog_means, key=hog_means.get)
    highest_symmetry = max(symmetry_means, key=symmetry_means.get)
    primary_corr = sensitivity_correlations_frame[
        (np.isclose(sensitivity_correlations_frame.border_exclusion_fraction, PRIMARY_BORDER_EXCLUSION))
        & (sensitivity_correlations_frame.canny_low_threshold == PRIMARY_CANNY[0])
        & (sensitivity_correlations_frame.canny_high_threshold == PRIMARY_CANNY[1])
    ]
    if not np.allclose(primary_corr.spearman_rho_vs_primary, 1.0):
        raise AssertionError("primary_sensitivity_correlation_not_one")
    min_contour_rho = sensitivity_correlations_frame.loc[
        sensitivity_correlations_frame.metric == "edge_contour_box_counting_dimension",
        "spearman_rho_vs_primary",
    ].min()

    readme = f"""# 全图边缘轮廓复杂度与表11候选复现实验

本目录记录2026-07-30新建的候选样本探索性实验。输入为Stage 1的568张候选图：姑苏459、桃花坞9、杨柳青100。候选组别、年代、独立作品单位和重复关系尚未全部人工冻结，因此本结果不能写成正式作品总体统计。

## 为什么需要补充轮廓复杂度

旧表11的正文同时声称使用HOG、人体关键点、轮廓复杂度和对称度，但旧表只列HOG方向集中度与水平对称度，且项目审计没有找到旧计算代码、参数或逐图结果。若正文保留“轮廓复杂度分析”，表中必须补充定义明确且有逐图证据的指标；否则应删除该方法声称。

本实验补充的是**全图边缘轮廓复杂度**，不是人物或单一物体的分割轮廓。人体关键点仍未计算。

## 固定定义

- 输入统一保持比例、长边缩放至{LONG_EDGE}px，主分析排除四周{PRIMARY_BORDER_EXCLUSION:.0%}。
- 灰度为BT.709；统一CLAHE局部对比度标准化（clip={CLAHE_CLIP_LIMIT}，网格{CLAHE_GRID_SIZE[0]}×{CLAHE_GRID_SIZE[1]}）；高斯核{GAUSSIAN_KERNEL}×{GAUSSIAN_KERNEL}、sigma={GAUSSIAN_SIGMA}；主Canny阈值={PRIMARY_CANNY[0]}/{PRIMARY_CANNY[1]}。
- 轮廓复杂度主指标：Canny边缘图在{BOX_SIZES_PX}像素盒尺度上的盒计数维数`D_box`。越高表示边缘在二维画面中越具空间填充性。
- 辅助指标：归一化轮廓总长度、每百万像素轮廓组件数、长度加权`周长/凸包周长`不规则度及边缘密度。
- HOG式方向集中度：9个0°—180°无向、梯度幅值加权方向箱，`9 × 最大方向概率`；均匀分布为1。该值是全局HOG式方向摘要，不是训练后形状检测器。
- 水平对称度：固定画面中轴上比较Sobel能量图左右镜像，`1-Σ|L-flip(R)|/Σ(L+flip(R))`。

## 候选结果

{table_md}

在这套预先固定的新定义下，HOG式方向集中度均值最高的是{GROUP_SHORT_CN[highest_hog]}，Sobel能量水平对称度均值最高的是{GROUP_SHORT_CN[highest_symmetry]}，边缘轮廓盒计数维数均值最高的是{GROUP_SHORT_CN[highest_contour]}。这些只是候选图像级描述，不能解释为刻版工艺、儒家审美或群体因果差异。

轮廓盒计数维数在预设边界/阈值敏感性组合中，相对主设置的最小逐图Spearman相关为{min_contour_rho:.3f}。完整组均值、逐图值和相关性见相应CSV。

## 表11旧值复核结论

旧稿`2.31±1.01 / 1.81±0.12 / 1.87±0.23`及`0.864±0.041 / 0.775±0.026 / 0.832±0.041`没有旧代码、公式、输入域或逐图结果支持，不能判定为已复现。本实验没有以这些数字为目标调整参数；新指标定义与旧稿缺失定义不可直接视为同一测量。

## 主要文件

- `run_candidate_shape_analysis.py`：完整主脚本。
- `shape_metrics_per_image.csv`：568张逐图主结果。
- `contour_complexity_sensitivity_per_image.csv`：9组边界/阈值设置的逐图轮廓结果。
- `shape_metrics_summary.csv`：均值、样本标准差、中位数、四分位数及bootstrap均值区间。
- `table11_candidate_exploratory.csv/.png`：候选版新表11。
- `table11_old_vs_reproduced_audit.csv`：旧表数字与新定义结果的证据审计。
- `figure_contour_pipeline_qc.png`：各组中位`D_box`样本的原图、边缘图和轮廓叠加。
- `figure_shape_metric_distributions.png`：三项主指标逐图分布。
- `TABLE11_VALIDATION_REPORT.md`：验证与论文使用边界。
- `FULL_PAPER_SECTION_SHAPE_CANDIDATE.md`：当前阶段可使用的审慎文字。

## 重跑

为防止静默覆盖，已有输出存在时脚本会停止。请指定一个新的输出目录：

```bash
PYTHONPYCACHEPREFIX=/tmp/gusu_shape_pycache .venv/bin/python \\
  reports/formal_shape_candidate_experiment_20260730/run_candidate_shape_analysis.py \\
  --output-dir reports/formal_shape_candidate_experiment_YYYYMMDD_rerun
```

`shape_candidate_analysis.ipynb`是读取已生成结果的复核伴随笔记本。当前环境未安装Jupyter/nbformat，故该notebook未执行；主脚本已完整执行，CSV、统计、图像和校验均由主脚本生成并另行复核。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    validation = f"""# 表11与轮廓复杂度复现验证报告

## 总体判定：可作为候选实验报告，不能直接替换为正式作品级结论

### 1. 数据完整性

- 568个唯一`sample_id`，组计数459、9、100。
- 全部路径存在，运行时重新计算SHA-256并与Stage 1 manifest逐图一致。
- 统计单位仍为候选数字图像；同版异图、组画和近重复尚未人工解决。

### 2. 轮廓复杂度

轮廓复杂度已实际计算，主指标为`edge_contour_box_counting_dimension`。它有明确公式、逐图CSV、参数敏感性和边缘叠加图支持。它衡量整幅图的边缘空间填充性，不能直接证明人物“圆融”、轮廓闭合、写实程度或刻版质量。

主设置下全部{len(primary)}张图得到有限`D_box`；盒计数拟合R²最小值为{primary.box_count_fit_r_squared.min():.4f}，中位数为{primary.box_count_fit_r_squared.median():.4f}。预设敏感性设置相对主设置的`D_box`逐图Spearman相关最小值为{min_contour_rho:.4f}。

### 3. 旧表11复核

- HOG旧值：**未复现**。旧项目无HOG方向数、cell/block参数、归一化方式、“方向集中度”公式或逐图结果。
- 水平对称度旧值：**未复现**。旧项目未说明输入是原像素、边缘图还是显著性图，也未说明外框/留白控制。
- 人体关键点比例：**未实现**。本轮没有运行人体检测或关键点模型。
- 轮廓复杂度：**本轮新增实现**，但属于全图边缘轮廓，不是人体或对象轮廓。

### 4. 解释风险

1. 桃花坞只有9张候选图，均值和标准差不稳定。
2. 图像级观测不独立：同版、系列和近重复尚未清理。
3. 数字化来源、压缩、画框、纸张破损、题字与扫描边界均可能影响HOG、对称度和边缘复杂度。
4. 指标差异不能自动推出“精工刻版”“建筑题材规律”“中庸均衡”或儒家理念。
5. 固定中轴对称度不等于视觉重心居中；需另行计算重心偏移才可讨论重心。

### 5. 分享状态

**Share with caveats**：可公开为候选复现实验及方法证据；正式论文表格须等待manifest冻结后按独立作品/版次重算，并完成重复控制和图像质量敏感性分析。
"""
    (output_dir / "TABLE11_VALIDATION_REPORT.md").write_text(validation, encoding="utf-8")

    paper = rf"""## （五）造型与全图轮廓特征：候选样本探索性复现

为避免把梯度描述符直接解释为人物造型，本研究将当前量化范围限定为整幅候选图像的方向结构、水平镜像平衡和边缘轮廓复杂度。输入包括459张姑苏、9张20世纪50年代桃花坞和100张清末杨柳青年画候选图像。三组尚未完成年代、独立作品单位及重复关系的人工冻结，因此以下结果仅用于方法复现和正式实验设计。

所有图像保持纵横比并将长边缩放至{LONG_EDGE}像素，排除四周{PRIMARY_BORDER_EXCLUSION:.0%}后转换为BT.709灰度图，并统一采用clip={CLAHE_CLIP_LIMIT}、{CLAHE_GRID_SIZE[0]}×{CLAHE_GRID_SIZE[1]}网格的CLAHE局部对比度标准化。该步骤用于降低深色、低对比度扫描件无法提取边缘的风险。HOG式方向集中度以9个0°—180°无向梯度箱的幅值加权分布计算：

$$C_{{\mathrm{{HOG}}}}=\frac{{\max_k p_k}}{{1/9}}=9\max_k p_k,$$

其中$p_k$为第$k$个方向箱的梯度幅值占比。$C_{{\mathrm{{HOG}}}}=1$对应九个方向箱均匀分布；数值越高仅表示全图梯度能量越集中于少数方向。

水平对称度基于Sobel梯度能量图，以固定画面中轴将左右两半镜像配准：

$$S_h=1-\frac{{\sum|E_L-\operatorname{{flip}}(E_R)|}}{{\sum(E_L+\operatorname{{flip}}(E_R))+\varepsilon}}.$$

轮廓复杂度采用Canny边缘图的盒计数维数。令$N(s)$为边长$s$的网格中至少包含一个边缘像素的格子数，则：

$$D_{{\mathrm{{box}}}}=\frac{{d\log N(s)}}{{d\log(1/s)}}.$$

本实验在4、8、16、32、64和128像素尺度的有效范围内线性拟合。$D_{{\mathrm{{box}}}}$越高，表示边缘在二维画面中越具空间填充性；它不是人物外轮廓的圆度、闭合度，也不能单独说明人物造型“圆融”。

**表11　候选样本形状与边缘轮廓特征（均值 ± 样本标准差）**

{table_md}

按照本轮固定定义，HOG式方向集中度均值最高的是{GROUP_SHORT_CN[highest_hog]}，水平对称度均值最高的是{GROUP_SHORT_CN[highest_symmetry]}，边缘轮廓盒计数维数均值最高的是{GROUP_SHORT_CN[highest_contour]}。这一描述仅指当前候选数字图像的测量排序。由于桃花坞候选仅9张，且同版异图、系列关系与数字化来源尚未完全控制，不宜据此推导刻版工艺优劣、人物造型范式或“中庸”思想。人物比例和关键点统计尚未实现，正文不得写成已经完成。
"""
    (output_dir / "FULL_PAPER_SECTION_SHAPE_CANDIDATE.md").write_text(paper, encoding="utf-8")

    chart_map = """# Chart map

| Figure | Analytical question | Form | Grain | Supported use |
|---|---|---|---|---|
| figure_shape_metric_distributions.png | 三组候选图的三项主指标分布是否不同 | 三联箱线图＋逐图散点 | 候选数字图像 | 显示分布、离群值和样本量不均衡；不作艺术史因果解释 |
| figure_contour_pipeline_qc.png | Canny边缘与轮廓提取实际捕捉了什么 | 三组中位D_box样例的处理流程 | 单张候选图 | 检查外框、题字、扫描边界和对象轮廓是否进入指标 |
| table11_candidate_exploratory.png | 新定义下三组均值与标准差是多少 | 研究表格 | 候选组汇总 | 精确查阅；须保留候选样本脚注 |
"""
    (output_dir / "chart_map.md").write_text(chart_map, encoding="utf-8")


def create_notebook(output_dir: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## tl;dr\n",
                    "This companion notebook loads and validates the already generated candidate-level shape outputs. The core image computation is in `run_candidate_shape_analysis.py`.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Context & Methods\n",
                    "Population: 459 + 9 + 100 candidate images, not a frozen formal manifest. Primary contour complexity is whole-image Canny edge box-counting dimension. Human keypoints are not computed.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "from IPython.display import display, Image\n",
                    "HERE = Path.cwd()\n",
                    "if not (HERE / 'shape_metrics_per_image.csv').exists():\n",
                    "    HERE = Path('reports/formal_shape_candidate_experiment_20260730')\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Data\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "metrics = pd.read_csv(HERE / 'shape_metrics_per_image.csv', encoding='utf-8-sig')\n",
                    "summary = pd.read_csv(HERE / 'shape_metrics_summary.csv', encoding='utf-8-sig')\n",
                    "assert len(metrics) == 568 and metrics.sample_id.nunique() == 568\n",
                    "assert metrics.groupby('group_id').size().to_dict() == {'gusu_qing_candidate': 459, 'taohuawu_1950s_candidate': 9, 'yangliuqing_late_qing_candidate': 100}\n",
                    "metrics[['sample_id','group_id','edge_contour_box_counting_dimension']].head()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Results\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "table11 = pd.read_csv(HERE / 'table11_candidate_exploratory.csv', encoding='utf-8-sig')\n",
                    "display(table11)\n",
                    "display(Image(filename=str(HERE / 'figure_shape_metric_distributions.png')))\n",
                    "display(Image(filename=str(HERE / 'figure_contour_pipeline_qc.png')))\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Takeaways\n",
                    "Interpret the outputs only as candidate-image descriptions. Do not infer human-body proportions, rounded modeling, carving quality, or Confucian aesthetics from these metrics alone.\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": platform.python_version()},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (output_dir / "shape_candidate_analysis.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def validate_outputs(
    manifest: pd.DataFrame,
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    table11: pd.DataFrame,
) -> dict[str, Any]:
    if len(primary) != 568 or primary.sample_id.nunique() != 568:
        raise AssertionError("primary_output_not_568_unique")
    if len(sensitivity) != 568 * len(BORDER_EXCLUSION_FRACTIONS) * len(CANNY_SETTINGS):
        raise AssertionError("sensitivity_row_count_mismatch")
    if primary.groupby("group_id").size().to_dict() != EXPECTED_COUNTS:
        raise AssertionError("primary_group_count_mismatch")
    if primary[list(PRIMARY_METRICS)].isna().any().any():
        raise AssertionError("primary_metric_missing_values")
    if not primary.hog_style_orientation_concentration.between(1.0, HOG_ORIENTATION_BINS).all():
        raise AssertionError("hog_concentration_out_of_range")
    if not primary.sobel_energy_horizontal_symmetry.between(0.0, 1.0).all():
        raise AssertionError("symmetry_out_of_range")
    if not primary.edge_contour_box_counting_dimension.between(0.0, 2.0).all():
        raise AssertionError("box_dimension_out_of_expected_range")
    if (primary.box_count_fit_r_squared < 0.90).any():
        raise AssertionError("box_count_fit_r_squared_below_0_90")
    bin_columns = [f"hog_orientation_bin_{index:02d}_probability" for index in range(1, 10)]
    if not np.allclose(primary[bin_columns].sum(axis=1), 1.0, atol=1e-9):
        raise AssertionError("hog_probabilities_do_not_sum_to_one")
    if len(table11) != 3:
        raise AssertionError("table11_expected_three_rows")
    expected_sha = manifest.set_index("sample_id").sha256.str.lower()
    observed_sha = primary.set_index("sample_id").sha256.str.lower().reindex(expected_sha.index)
    if not expected_sha.equals(observed_sha):
        raise AssertionError("output_manifest_sha_mismatch")
    return {
        "primary_rows": len(primary),
        "unique_sample_ids": primary.sample_id.nunique(),
        "sensitivity_rows": len(sensitivity),
        "group_counts": primary.groupby("group_id").size().to_dict(),
        "missing_primary_metric_cells": int(primary[list(PRIMARY_METRICS)].isna().sum().sum()),
        "minimum_box_fit_r_squared": float(primary.box_count_fit_r_squared.min()),
        "median_box_fit_r_squared": float(primary.box_count_fit_r_squared.median()),
        "all_hog_probabilities_sum_to_one": True,
        "human_body_keypoints_computed": False,
        "segmented_object_contour_computed": False,
    }


def ensure_new_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            "refusing_to_overwrite_existing_outputs: " + ", ".join(existing)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    ensure_new_outputs(output_dir)
    started = datetime.now(timezone.utc)
    manifest = load_candidate_manifest(args.manifest.resolve())
    primary, sensitivity = compute_metrics(manifest)
    summary = summarize(primary, PRIMARY_METRICS)
    sensitivity_summary = summarize_sensitivity(sensitivity)
    sensitivity_correlation = sensitivity_correlations(primary, sensitivity)
    omnibus, pairwise = exploratory_tests(primary)
    table11 = make_table11(summary)
    audit = audit_old_table(table11)
    validation = validate_outputs(manifest, primary, sensitivity, table11)

    manifest.drop(columns=["resolved_image_path"]).to_csv(
        output_dir / "analysis_manifest_snapshot.csv", index=False, encoding="utf-8-sig"
    )
    primary.to_csv(output_dir / "shape_metrics_per_image.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(
        output_dir / "contour_complexity_sensitivity_per_image.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(output_dir / "shape_metrics_summary.csv", index=False, encoding="utf-8-sig")
    sensitivity_summary.to_csv(
        output_dir / "contour_complexity_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensitivity_correlation.to_csv(
        output_dir / "contour_complexity_sensitivity_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    omnibus.to_csv(
        output_dir / "shape_exploratory_omnibus_tests.csv", index=False, encoding="utf-8-sig"
    )
    pairwise.to_csv(
        output_dir / "shape_exploratory_pairwise_tests.csv", index=False, encoding="utf-8-sig"
    )
    table11.to_csv(
        output_dir / "table11_candidate_exploratory.csv", index=False, encoding="utf-8-sig"
    )
    audit.to_csv(
        output_dir / "table11_old_vs_reproduced_audit.csv", index=False, encoding="utf-8-sig"
    )

    create_distribution_figure(primary, output_dir)
    create_qc_figure(primary, manifest, output_dir)
    create_table_figure(table11, output_dir)

    parameters = {
        "analysis_scope": "candidate_exploratory_not_formal_manifest",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "candidate_counts": EXPECTED_COUNTS,
        "resize": {"preserve_aspect_ratio": True, "long_edge_px": LONG_EDGE, "padding": False},
        "grayscale": "BT.709: 0.2126R + 0.7152G + 0.0722B",
        "local_contrast_normalization": {
            "method": "OpenCV CLAHE",
            "clip_limit": CLAHE_CLIP_LIMIT,
            "tile_grid_size": CLAHE_GRID_SIZE,
            "applied_to_all_images_before_sobel_and_canny": True,
        },
        "primary_border_exclusion_fraction_each_side": PRIMARY_BORDER_EXCLUSION,
        "border_exclusion_sensitivity": BORDER_EXCLUSION_FRACTIONS,
        "gaussian_blur": {"kernel": GAUSSIAN_KERNEL, "sigma": GAUSSIAN_SIGMA},
        "canny": {
            "primary": PRIMARY_CANNY,
            "sensitivity": CANNY_SETTINGS,
            "aperture_size": 3,
            "L2gradient": True,
        },
        "contour_complexity_primary": {
            "name": "edge_contour_box_counting_dimension",
            "definition": "slope of log occupied box count versus log inverse box size",
            "box_sizes_px": BOX_SIZES_PX,
            "minimum_edge_pixels": MIN_EDGE_PIXELS_FOR_BOX_FIT,
            "interpretation": "whole-image edge spatial filling; not object or human silhouette complexity",
        },
        "contour_auxiliary": {
            "retrieval": "cv2.RETR_LIST",
            "chain": "cv2.CHAIN_APPROX_NONE",
            "minimum_perimeter_px": MIN_CONTOUR_PERIMETER_PX,
            "normalized_total_length": "sum contour perimeter / sqrt(valid pixel count)",
            "irregularity": "perimeter/convex-hull-perimeter, length weighted",
        },
        "hog_style_direction_concentration": {
            "unsigned_orientation_range_degrees": [0, 180],
            "bins": HOG_ORIENTATION_BINS,
            "weights": "Sobel gradient magnitude",
            "orientation_bin_interpolation": "linear cyclic",
            "formula": "9 * max(direction-bin probability)",
            "block_normalization": "not used; scalar is a global orientation-distribution summary",
        },
        "horizontal_symmetry": {
            "input": "Sobel gradient magnitude robustly scaled by within-image 99th percentile",
            "axis": "fixed image center",
            "formula": "1-sum(abs(L-flip(R)))/sum(L+flip(R))",
        },
        "statistics": {
            "analysis_unit": "candidate digital image",
            "sd": "sample standard deviation, ddof=1",
            "bootstrap_mean_ci_iterations": BOOTSTRAP_ITERATIONS,
            "exploratory_tests": "Kruskal-Wallis and pairwise Mann-Whitney with Holm correction",
            "formal_inference_allowed": False,
        },
        "random_seed": RANDOM_SEED,
        "old_table_values_used_as_targets": False,
        "human_body_keypoints_computed": False,
        "segmented_object_contour_computed": False,
        "software": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (output_dir / "shape_parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    requirements = (
        f"numpy=={np.__version__}\n"
        f"pandas=={pd.__version__}\n"
        f"scipy=={scipy.__version__}\n"
        f"matplotlib=={matplotlib.__version__}\n"
        "opencv-python==4.13.0.92\n"
    )
    (output_dir / "requirements_frozen.txt").write_text(requirements, encoding="utf-8")
    create_notebook(output_dir)
    write_reports(output_dir, primary, summary, sensitivity_correlation, table11, audit)

    finished = datetime.now(timezone.utc)
    receipt = {
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "command": " ".join(sys.argv),
        "output_dir": str(output_dir),
        "validation": validation,
        "output_sha256": {
            name: sha256_file(output_dir / name)
            for name in OUTPUT_FILES
            if name != "shape_run_receipt.json" and (output_dir / name).exists()
        },
    }
    (output_dir / "shape_run_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt["validation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
