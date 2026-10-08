#!/usr/bin/env python3
"""Reproducible automatic vanishing-point candidate screening.

This pipeline detects long line segments with OpenCV LSD, proposes up to two
vanishing points through deterministic pair-intersection RANSAC, refines each
candidate by weighted least squares, and exports overlays for human review.
Automatic candidates are not treated as verified perspective evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gusu_perspective_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gusu_perspective_cache")

import cv2
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
OVERLAY_DIR = OUTPUT_DIR / "vanishing_point_overlays"
MANIFEST_PATH = PROJECT_ROOT / "reports/formal_aesthetics_reproduction_20260727/formal_analysis_manifest.csv"

GROUP_MAP = {
    "清代姑苏版画（候选，待确认）": ("gusu_qing_candidate", "姑苏版画（候选）", 1),
    "20世纪50年代桃花坞年画（候选，待确认）": ("taohuawu_1950s_candidate", "桃花坞年画（候选）", 2),
    "清末杨柳青年画（候选，待确认）": ("yangliuqing_late_qing_candidate", "杨柳青年画（候选）", 3),
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

LONG_EDGE = 1024
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
MIN_LENGTH_DIAGONAL_FRACTION = 0.04
MIN_LENGTH_PX = 24.0
BORDER_LINE_EXCLUSION_FRACTION = 0.02
MAX_LONG_LINES = 250
MIN_PAIR_ANGLE_DEGREES = 4.0
MAX_RANSAC_CANDIDATES = 1200
ANGULAR_SUPPORT_TOLERANCE_DEGREES = 3.0
MIN_SUPPORT_LINES = 5
MIN_ACTIVE_SUPPORT_LENGTH_RATIO = 0.15
MIN_TOTAL_SUPPORT_LENGTH_RATIO = 0.08
MAX_MEDIAN_RESIDUAL_DEGREES = 2.0
MAX_VP_CENTER_DISTANCE_DIAGONALS = 3.0
MAX_VANISHING_POINTS = 2
RANDOM_SEED = 20260729

OUTPUT_FILES = [
    "analysis_manifest_snapshot.csv",
    "vanishing_point_metrics_per_image.csv",
    "vanishing_points_long.csv",
    "vanishing_point_supporting_lines.csv",
    "vanishing_point_group_summary.csv",
    "vanishing_point_review_template.csv",
    "vanishing_point_review_gallery.html",
    "figure_vanishing_point_candidate_counts.png",
    "figure_vanishing_point_candidate_examples.png",
    "synthetic_vanishing_point_validation.png",
    "synthetic_vanishing_point_validation.json",
    "vanishing_point_parameters.json",
    "VANISHING_POINT_VALIDATION_REPORT.md",
    "PAPER_SECTION_PERSPECTIVE_CANDIDATE.md",
    "vanishing_point_candidate_analysis.ipynb",
    "vanishing_point_run_receipt.json",
]


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


def write_image_rgb(path: Path, image: np.ndarray, quality: int = 88) -> None:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if path.suffix.lower() == ".png":
        extension = ".png"
        parameters = [cv2.IMWRITE_PNG_COMPRESSION, 6]
    else:
        extension = ".jpg"
        parameters = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, encoded = cv2.imencode(extension, bgr, parameters)
    if not ok:
        raise ValueError(f"cannot_encode_image: {path}")
    encoded.tofile(path)


def resize_long_edge(image: np.ndarray, long_edge: int = LONG_EDGE) -> np.ndarray:
    height, width = image.shape[:2]
    scale = long_edge / max(height, width)
    new_width = max(32, int(round(width * scale)))
    new_height = max(32, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def load_manifest() -> pd.DataFrame:
    frame = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig", dtype=str).fillna("")
    required = {"sample_id", "image_path", "sha256", "proposed_group", "source_prefix"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manifest_missing_columns: {sorted(missing)}")
    if len(frame) != 568 or frame["sample_id"].nunique() != 568:
        raise ValueError(f"manifest_expected_568_unique: rows={len(frame)} unique={frame['sample_id'].nunique()}")
    unknown = sorted(set(frame["proposed_group"]) - set(GROUP_MAP))
    if unknown:
        raise ValueError(f"unexpected_groups: {unknown}")
    mapped = frame["proposed_group"].map(GROUP_MAP)
    frame["group_id"] = mapped.map(lambda value: value[0])
    frame["group_label"] = mapped.map(lambda value: value[1])
    frame["group_order"] = mapped.map(lambda value: value[2])
    if frame.groupby("group_id").size().to_dict() != EXPECTED_COUNTS:
        raise ValueError("candidate_group_counts_mismatch")
    frame["resolved_image_path"] = frame["image_path"].map(
        lambda value: str((PROJECT_ROOT / value).resolve()) if not Path(value).is_absolute() else value
    )
    return frame.sort_values(["group_order", "sample_id"]).reset_index(drop=True)


def grayscale_for_lsd(image_rgb: np.ndarray) -> np.ndarray:
    rgb = image_rgb.astype(np.float32)
    gray = np.clip(np.rint(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(gray)


def is_same_border_line(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> bool:
    margin_x = width * BORDER_LINE_EXCLUSION_FRACTION
    margin_y = height * BORDER_LINE_EXCLUSION_FRACTION
    return (
        (x1 <= margin_x and x2 <= margin_x)
        or (x1 >= width - margin_x and x2 >= width - margin_x)
        or (y1 <= margin_y and y2 <= margin_y)
        or (y1 >= height - margin_y and y2 >= height - margin_y)
    )


def detect_long_lines(gray: np.ndarray) -> tuple[int, np.ndarray]:
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    result = detector.detect(gray)
    raw = result[0]
    raw_count = 0 if raw is None else len(raw)
    if raw is None:
        return 0, np.empty((0, 11), dtype=np.float64)
    height, width = gray.shape
    diagonal = math.hypot(width, height)
    minimum_length = max(MIN_LENGTH_PX, MIN_LENGTH_DIAGONAL_FRACTION * diagonal)
    rows = []
    for item in raw[:, 0, :]:
        x1, y1, x2, y2 = map(float, item)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < minimum_length or is_same_border_line(x1, y1, x2, y2, width, height):
            continue
        a, b, c = y1 - y2, x2 - x1, x1 * y2 - x2 * y1
        line_norm = math.hypot(a, b)
        if line_norm <= 0:
            continue
        a, b, c = a / line_norm, b / line_norm, c / line_norm
        rows.append([x1, y1, x2, y2, length, dx / length, dy / length, a, b, c, 0.5 * (x1 + x2), 0.5 * (y1 + y2)])
    if not rows:
        return raw_count, np.empty((0, 12), dtype=np.float64)
    lines = np.array(rows, dtype=np.float64)
    lines = lines[np.argsort(-lines[:, 4])]
    return raw_count, lines[:MAX_LONG_LINES]


def intersections_from_pairs(lines: np.ndarray, width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    if len(lines) < 2:
        return np.empty((0, 2), dtype=np.float64)
    first, second = np.triu_indices(len(lines), k=1)
    direction_dot = np.abs(lines[first, 5] * lines[second, 5] + lines[first, 6] * lines[second, 6])
    angle = np.degrees(np.arccos(np.clip(direction_dot, 0.0, 1.0)))
    mask = angle >= MIN_PAIR_ANGLE_DEGREES
    first, second = first[mask], second[mask]
    if len(first) == 0:
        return np.empty((0, 2), dtype=np.float64)
    determinant = lines[first, 7] * lines[second, 8] - lines[second, 7] * lines[first, 8]
    stable = np.abs(determinant) > 1e-8
    first, second, determinant = first[stable], second[stable], determinant[stable]
    x = (lines[first, 8] * lines[second, 9] - lines[second, 8] * lines[first, 9]) / determinant
    y = (lines[second, 7] * lines[first, 9] - lines[first, 7] * lines[second, 9]) / determinant
    points = np.column_stack([x, y])
    center = np.array([width / 2.0, height / 2.0])
    diagonal = math.hypot(width, height)
    finite = np.isfinite(points).all(axis=1)
    bounded = np.linalg.norm(points - center, axis=1) <= MAX_VP_CENTER_DISTANCE_DIAGONALS * diagonal
    points = points[finite & bounded]
    if len(points) > MAX_RANSAC_CANDIDATES:
        chosen = rng.choice(len(points), MAX_RANSAC_CANDIDATES, replace=False)
        points = points[chosen]
    return points


def angular_residuals(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Return point-to-segment-direction angular residuals in radians."""
    if len(points) == 0 or len(lines) == 0:
        return np.empty((len(points), len(lines)), dtype=np.float64)
    numerator = np.abs(points[:, 0, None] * lines[None, :, 7] + points[:, 1, None] * lines[None, :, 8] + lines[None, :, 9])
    dx = points[:, 0, None] - lines[None, :, 10]
    dy = points[:, 1, None] - lines[None, :, 11]
    denominator = np.maximum(np.hypot(dx, dy), 1e-6)
    sine = np.clip(numerator / denominator, 0.0, 1.0)
    return np.arcsin(sine)


def refine_point(lines: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
    matrix = lines[:, 7:9]
    target = -lines[:, 9]
    weighted_matrix = matrix * np.sqrt(weights)[:, None]
    weighted_target = target * np.sqrt(weights)
    try:
        point, _, rank, _ = np.linalg.lstsq(weighted_matrix, weighted_target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 2 or not np.isfinite(point).all():
        return None
    return point


def detect_vanishing_points(lines: np.ndarray, width: int, height: int, seed: int) -> list[dict[str, Any]]:
    if len(lines) < MIN_SUPPORT_LINES:
        return []
    total_length = float(lines[:, 4].sum())
    active_indices = np.arange(len(lines))
    results: list[dict[str, Any]] = []
    tolerance = math.radians(ANGULAR_SUPPORT_TOLERANCE_DEGREES)
    rng = np.random.default_rng(seed)

    for rank in range(1, MAX_VANISHING_POINTS + 1):
        active = lines[active_indices]
        if len(active) < MIN_SUPPORT_LINES:
            break
        points = intersections_from_pairs(active, width, height, rng)
        if len(points) == 0:
            break
        residuals = angular_residuals(points, active)
        support = residuals <= tolerance
        weighted_support = support @ active[:, 4]
        best = int(np.argmax(weighted_support))
        initial_support = support[best]
        if int(initial_support.sum()) < MIN_SUPPORT_LINES:
            break
        refined = refine_point(active[initial_support], active[initial_support, 4])
        if refined is None:
            break
        refined_residuals = angular_residuals(refined[None, :], active)[0]
        final_support = refined_residuals <= tolerance
        if int(final_support.sum()) < MIN_SUPPORT_LINES:
            break
        refined_second = refine_point(active[final_support], active[final_support, 4])
        if refined_second is not None:
            refined = refined_second
            refined_residuals = angular_residuals(refined[None, :], active)[0]
            final_support = refined_residuals <= tolerance
        support_indices_active = np.where(final_support)[0]
        support_indices_global = active_indices[support_indices_active]
        support_length = float(active[final_support, 4].sum())
        active_ratio = support_length / float(active[:, 4].sum())
        total_ratio = support_length / total_length
        median_residual = float(np.degrees(np.median(refined_residuals[final_support])))
        center_distance = float(np.linalg.norm(refined - np.array([width / 2.0, height / 2.0])))
        center_distance_diagonals = center_distance / math.hypot(width, height)
        accepted = (
            len(support_indices_global) >= MIN_SUPPORT_LINES
            and active_ratio >= MIN_ACTIVE_SUPPORT_LENGTH_RATIO
            and total_ratio >= MIN_TOTAL_SUPPORT_LENGTH_RATIO
            and median_residual <= MAX_MEDIAN_RESIDUAL_DEGREES
            and center_distance_diagonals <= MAX_VP_CENTER_DISTANCE_DIAGONALS
        )
        if not accepted:
            break
        confidence = active_ratio * min(1.0, len(support_indices_global) / 12.0) * math.exp(-median_residual / 2.0)
        results.append(
            {
                "vp_rank": rank,
                "x": float(refined[0]),
                "y": float(refined[1]),
                "x_normalized": float(refined[0] / width),
                "y_normalized": float(refined[1] / height),
                "inside_image": bool(0 <= refined[0] < width and 0 <= refined[1] < height),
                "support_line_count": int(len(support_indices_global)),
                "support_length_ratio_active": active_ratio,
                "support_length_ratio_all": total_ratio,
                "median_angular_residual_degrees": median_residual,
                "center_distance_diagonals": center_distance_diagonals,
                "automatic_confidence_score_not_probability": confidence,
                "support_indices": support_indices_global,
                "support_residuals_degrees": np.degrees(refined_residuals[final_support]),
            }
        )
        active_indices = np.setdiff1d(active_indices, support_indices_global, assume_unique=False)
    return results


def draw_overlay(image: np.ndarray, lines: np.ndarray, vanishing_points: list[dict[str, Any]]) -> np.ndarray:
    canvas = image.copy()
    for line in lines:
        cv2.line(canvas, (int(round(line[0])), int(round(line[1]))), (int(round(line[2])), int(round(line[3]))), (180, 200, 210), 1, cv2.LINE_AA)
    colors = [(215, 48, 39), (49, 130, 189)]
    for vp_index, vp in enumerate(vanishing_points):
        color = colors[vp_index % len(colors)]
        for line_index in vp["support_indices"]:
            line = lines[int(line_index)]
            midpoint = (int(round(line[10])), int(round(line[11])))
            target = (int(round(vp["x"])), int(round(vp["y"])))
            cv2.line(canvas, midpoint, target, color, 1, cv2.LINE_AA)
            cv2.line(canvas, (int(round(line[0])), int(round(line[1]))), (int(round(line[2])), int(round(line[3]))), color, 2, cv2.LINE_AA)
        if vp["inside_image"]:
            point = (int(round(vp["x"])), int(round(vp["y"])))
            cv2.drawMarker(canvas, point, color, cv2.MARKER_CROSS, 24, 3, cv2.LINE_AA)
        label = f"VP{vp['vp_rank']} x={vp['x_normalized']:.2f} y={vp['y_normalized']:.2f} support={vp['support_line_count']}"
        cv2.putText(canvas, label, (12, 28 + 28 * vp_index), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return canvas


def seed_for(sample_id: str) -> int:
    return (int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16) ^ RANDOM_SEED) & 0xFFFFFFFF


def synthetic_validation() -> dict[str, Any]:
    width, height = 800, 600
    true_vp = np.array([410.0, 180.0])
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    endpoints = [(40, 590), (120, 590), (220, 590), (320, 590), (500, 590), (620, 590), (760, 590), (5, 420), (795, 400)]
    for endpoint in endpoints:
        cv2.line(image, endpoint, tuple(true_vp.astype(int)), (30, 30, 30), 3, cv2.LINE_AA)
    cv2.line(image, (40, 80), (760, 80), (120, 120, 120), 2, cv2.LINE_AA)
    cv2.line(image, (60, 520), (740, 520), (120, 120, 120), 2, cv2.LINE_AA)
    gray = grayscale_for_lsd(image)
    raw_count, lines = detect_long_lines(gray)
    candidates = detect_vanishing_points(lines, width, height, RANDOM_SEED)
    if not candidates:
        raise AssertionError("synthetic_converging_lines_not_detected")
    detected = np.array([candidates[0]["x"], candidates[0]["y"]])
    error = float(np.linalg.norm(detected - true_vp))
    if error > 20.0:
        raise AssertionError(f"synthetic_vp_error_too_large: {error}")
    overlay = draw_overlay(image, lines, candidates)
    write_image_rgb(OUTPUT_DIR / "synthetic_vanishing_point_validation.png", overlay, quality=95)
    result = {
        "true_vp": true_vp.tolist(),
        "detected_vp": detected.tolist(),
        "euclidean_error_px": error,
        "raw_lsd_segments": raw_count,
        "filtered_long_segments": len(lines),
        "detected_candidates": len(candidates),
        "status": "passed",
    }
    (OUTPUT_DIR / "synthetic_vanishing_point_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def process_manifest(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, Any]] = []
    vp_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    OVERLAY_DIR.mkdir(parents=True, exist_ok=False)

    for index, row in manifest.iterrows():
        path = Path(row["resolved_image_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        if sha256_file(path).lower() != row["sha256"].lower():
            raise ValueError(f"sha256_mismatch: {row['sample_id']}")
        original = read_image_rgb(path)
        resized = resize_long_edge(original)
        gray = grayscale_for_lsd(resized)
        raw_count, lines = detect_long_lines(gray)
        candidates = detect_vanishing_points(lines, resized.shape[1], resized.shape[0], seed_for(row["sample_id"]))
        if len(lines) < MIN_SUPPORT_LINES:
            status = "insufficient_long_lines_under_fixed_parameters"
        elif candidates:
            status = "automatic_candidate_detected_needs_manual_review"
        else:
            status = "no_candidate_detected_under_fixed_parameters_not_proof_of_absence"
        overlay_rel = ""
        if candidates:
            overlay_path = OVERLAY_DIR / f"{row['sample_id']}.jpg"
            write_image_rgb(overlay_path, draw_overlay(resized, lines, candidates))
            overlay_rel = str(overlay_path.relative_to(OUTPUT_DIR))
        primary = candidates[0] if candidates else None
        metrics_rows.append(
            {
                "sample_id": row["sample_id"],
                "image_path": row["image_path"],
                "sha256": row["sha256"],
                "group_id": row["group_id"],
                "group_label": row["group_label"],
                "group_order": int(row["group_order"]),
                "source_prefix": row["source_prefix"],
                "analysis_width": resized.shape[1],
                "analysis_height": resized.shape[0],
                "raw_lsd_segment_count": raw_count,
                "filtered_long_segment_count": len(lines),
                "automatic_vp_candidate_count": len(candidates),
                "primary_vp_x_normalized": primary["x_normalized"] if primary else np.nan,
                "primary_vp_y_normalized": primary["y_normalized"] if primary else np.nan,
                "primary_vp_inside_image": primary["inside_image"] if primary else "",
                "primary_support_line_count": primary["support_line_count"] if primary else 0,
                "primary_support_length_ratio_all": primary["support_length_ratio_all"] if primary else 0.0,
                "primary_median_angular_residual_degrees": primary["median_angular_residual_degrees"] if primary else np.nan,
                "primary_automatic_confidence_score_not_probability": primary["automatic_confidence_score_not_probability"] if primary else 0.0,
                "automatic_status": status,
                "overlay_path": overlay_rel,
                "manual_review_status": "pending",
                "analysis_scope": "candidate_screening_not_human_verified",
            }
        )
        for vp in candidates:
            vp_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "group_id": row["group_id"],
                    "group_label": row["group_label"],
                    **{key: value for key, value in vp.items() if key not in {"support_indices", "support_residuals_degrees"}},
                    "manual_validity": "",
                }
            )
            for line_index, residual in zip(vp["support_indices"], vp["support_residuals_degrees"]):
                line = lines[int(line_index)]
                support_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "vp_rank": vp["vp_rank"],
                        "line_index": int(line_index),
                        "x1": line[0], "y1": line[1], "x2": line[2], "y2": line[3],
                        "length_px": line[4],
                        "angular_residual_degrees": float(residual),
                    }
                )
        if (index + 1) % 100 == 0 or index + 1 == len(manifest):
            print(f"Perspective screened {index + 1}/{len(manifest)}", flush=True)
    return pd.DataFrame(metrics_rows), pd.DataFrame(vp_rows), pd.DataFrame(support_rows)


def make_group_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_id in GROUP_ORDER:
        subset = metrics[metrics["group_id"] == group_id]
        detected = subset["automatic_vp_candidate_count"] > 0
        rows.append(
            {
                "group_id": group_id,
                "group_label": subset["group_label"].iloc[0],
                "n_candidate_images": len(subset),
                "n_with_sufficient_long_lines": int((subset["filtered_long_segment_count"] >= MIN_SUPPORT_LINES).sum()),
                "n_automatic_candidate_detected": int(detected.sum()),
                "automatic_candidate_detection_fraction": float(detected.mean()),
                "n_with_one_candidate": int((subset["automatic_vp_candidate_count"] == 1).sum()),
                "n_with_two_candidates": int((subset["automatic_vp_candidate_count"] == 2).sum()),
                "median_filtered_long_segments": float(subset["filtered_long_segment_count"].median()),
                "manual_verified_count": 0,
                "interpretation": "automatic_screening_only_not_prevalence",
            }
        )
    return pd.DataFrame(rows)


def make_figures(metrics: pd.DataFrame, group_summary: pd.DataFrame) -> None:
    labels = ["Gusu", "THW", "YLQ"]
    group_english = dict(zip(GROUP_ORDER, labels))
    detected = [int(group_summary.loc[group_summary.group_id == group, "n_automatic_candidate_detected"].iloc[0]) for group in GROUP_ORDER]
    total = [EXPECTED_COUNTS[group] for group in GROUP_ORDER]
    fig, axis = plt.subplots(figsize=(7.4, 5.2), dpi=180)
    bars = axis.bar(labels, detected, color=["#3A7D44", "#D4A017", "#4062BB"])
    axis.set_ylabel("Images with automatic VP candidate")
    axis.set_title("Automatic vanishing-point screening (manual review pending)")
    for bar, value, denominator in zip(bars, detected, total):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value}/{denominator}", ha="center", va="bottom")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_vanishing_point_candidate_counts.png", bbox_inches="tight")
    plt.close(fig)

    selected = []
    for group_id in GROUP_ORDER:
        subset = metrics[(metrics.group_id == group_id) & (metrics.automatic_vp_candidate_count > 0)].nlargest(
            2, "primary_automatic_confidence_score_not_probability"
        )
        selected.extend(subset.to_dict("records"))
    if selected:
        columns = 2
        rows = math.ceil(len(selected) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(10, 4.5 * rows), dpi=160)
        axes_array = np.atleast_1d(axes).ravel()
        for axis, record in zip(axes_array, selected):
            image = read_image_rgb(OUTPUT_DIR / record["overlay_path"])
            axis.imshow(image)
            axis.set_title(f"{group_english[record['group_id']]} | {record['sample_id']} | unverified", fontsize=8, pad=3)
            axis.axis("off")
        for axis in axes_array[len(selected):]:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "figure_vanishing_point_candidate_examples.png", bbox_inches="tight")
        plt.close(fig)
    else:
        blank = np.full((300, 600, 3), 255, dtype=np.uint8)
        cv2.putText(blank, "No automatic candidates under fixed parameters", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        write_image_rgb(OUTPUT_DIR / "figure_vanishing_point_candidate_examples.png", blank)


def build_review_gallery(metrics: pd.DataFrame) -> None:
    records = []
    for row in metrics.itertuples(index=False):
        overlay_path = "" if pd.isna(row.overlay_path) else str(row.overlay_path)
        if overlay_path:
            display_path = overlay_path
        else:
            display_path = "../../" + str(row.image_path).replace("\\", "/")
        records.append(
            {
                "sample_id": row.sample_id,
                "group": GROUP_SHORT[row.group_id],
                "source_prefix": row.source_prefix,
                "automatic_status": row.automatic_status,
                "automatic_count": int(row.automatic_vp_candidate_count),
                "confidence": float(row.primary_automatic_confidence_score_not_probability),
                "display_path": display_path,
                "source_path": row.image_path,
            }
        )
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>消失点候选人工复核</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:18px;background:#f5f5f2;color:#222}} .notice{{background:#fff3cd;padding:12px;border-left:5px solid #d39e00}} .tools{{position:sticky;top:0;background:#f5f5f2;padding:10px 0;z-index:3}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}} .card{{background:white;border:1px solid #ddd;border-radius:8px;padding:10px}} img{{width:100%;height:310px;object-fit:contain;background:#eee}} label{{display:block;margin-top:6px;font-size:12px}} select,textarea{{width:100%;box-sizing:border-box}} textarea{{height:58px}} .meta{{font-size:12px;word-break:break-all}} button{{padding:8px 12px;margin-right:8px}}</style></head><body>
<h1>消失点候选人工复核画廊</h1><div class="notice">自动检测只是审核候选，不代表作品确有消失点；未检测也不代表消失点不存在。必须结合建筑直线适用性和叠加线人工核验。</div>
<div class="tools"><select id="groupFilter"><option value="">全部组别</option><option>姑苏</option><option>桃花坞</option><option>杨柳青</option></select><select id="statusFilter"><option value="">全部自动状态</option><option value="detected">有自动候选</option><option value="none">无自动候选</option></select><button onclick="render()">筛选</button><button onclick="exportCSV()">导出人工审核CSV</button><span id="counter"></span></div><div id="grid" class="grid"></div>
<script>const DATA={payload}; const KEY='gusu_vp_review_20260729'; let decisions=JSON.parse(localStorage.getItem(KEY)||'{{}}');
function esc(s){{return String(s).replace(/[&<>"']/g,m=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m]))}}
function save(id,field,value){{decisions[id]=decisions[id]||{{}};decisions[id][field]=value;localStorage.setItem(KEY,JSON.stringify(decisions));}}
function options(current,values){{return [''].concat(values).map(v=>`<option ${{v===current?'selected':''}}>${{esc(v)}}</option>`).join('')}}
function render(){{const gf=document.getElementById('groupFilter').value,sf=document.getElementById('statusFilter').value;let rows=DATA.filter(r=>(!gf||r.group===gf)&&(!sf||(sf==='detected'?r.automatic_count>0:r.automatic_count===0)));document.getElementById('counter').textContent=`显示 ${{rows.length}}/568`;document.getElementById('grid').innerHTML=rows.map(r=>{{let d=decisions[r.sample_id]||{{}};return `<div class="card"><img loading="lazy" src="${{esc(r.display_path)}}"><b>${{esc(r.sample_id)}}</b>｜${{esc(r.group)}}<div class="meta">${{esc(r.source_path)}}<br>自动候选数=${{r.automatic_count}}；score=${{r.confidence.toFixed(4)}}<br>${{esc(r.automatic_status)}}</div><label>建筑直线适用性<select onchange="save('${{r.sample_id}}','manual_applicability',this.value)">${{options(d.manual_applicability,['applicable','not_applicable','uncertain'])}}</select></label><label>消失点候选有效性<select onchange="save('${{r.sample_id}}','manual_vp_validity',this.value)">${{options(d.manual_vp_validity,['valid','invalid','partial','uncertain'])}}</select></label><label>人工确认消失点数<select onchange="save('${{r.sample_id}}','manual_vp_count',this.value)">${{options(d.manual_vp_count,['0','1','2','3+','uncertain'])}}</select></label><label>审核人<input style="width:100%" value="${{esc(d.reviewer||'')}}" onchange="save('${{r.sample_id}}','reviewer',this.value)"></label><label>备注<textarea onchange="save('${{r.sample_id}}','review_note',this.value)">${{esc(d.review_note||'')}}</textarea></label></div>`}}).join('')}}
function csv(v){{return '"'+String(v??'').replaceAll('"','""')+'"'}} function exportCSV(){{let h=['sample_id','group','source_path','automatic_status','automatic_vp_candidate_count','manual_applicability','manual_vp_validity','manual_vp_count','reviewer','review_note'];let lines=[h.map(csv).join(',')];for(const r of DATA){{let d=decisions[r.sample_id]||{{}};lines.push([r.sample_id,r.group,r.source_path,r.automatic_status,r.automatic_count,d.manual_applicability||'',d.manual_vp_validity||'',d.manual_vp_count||'',d.reviewer||'',d.review_note||''].map(csv).join(','))}}let blob=new Blob(['\ufeff'+lines.join('\n')],{{type:'text/csv'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='vanishing_point_manual_review.csv';a.click();URL.revokeObjectURL(a.href)}} render();</script></body></html>"""
    # The Python triple-quoted template resolves ``\\n`` before JavaScript sees
    # it; use an explicit character expression so the exported CSV remains
    # syntactically valid without embedding a raw newline in a JS string.
    page = page.replace("lines.join('\n')", "lines.join(String.fromCharCode(10))")
    (OUTPUT_DIR / "vanishing_point_review_gallery.html").write_text(page, encoding="utf-8")


def build_notebook() -> None:
    def markdown(source: str) -> dict[str, Any]:
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

    def code(source: str) -> dict[str, Any]:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}

    notebook = {
        "cells": [
            markdown("# Automatic vanishing-point candidate screening\n\n## tl;dr\nThis notebook verifies deterministic LSD + intersection-RANSAC outputs. Automatic candidates require manual review; non-detection is not proof of absence."),
            markdown("## Context & Methods\n\nLong lines are detected after BT.709 grayscale and CLAHE. Candidate intersections are scored by length-weighted angular support, refined by weighted least squares, and filtered using fixed support/residual criteria.\n\n### Key Assumptions\nThe full 568-image candidate set is screened, including images for which architectural-line geometry may be inapplicable. The gallery is the required next human-validation step."),
            code("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nHERE=Path.cwd()\nif not (HERE/'vanishing_point_metrics_per_image.csv').exists(): HERE=Path('reports/formal_perspective_candidate_experiment_20260729')\nmetrics=pd.read_csv(HERE/'vanishing_point_metrics_per_image.csv',encoding='utf-8-sig')\nsummary=pd.read_csv(HERE/'vanishing_point_group_summary.csv',encoding='utf-8-sig')\nassert len(metrics)==568 and metrics.sample_id.nunique()==568\nprint({'images':len(metrics),'automatic_candidates':int((metrics.automatic_vp_candidate_count>0).sum()),'manual_verified':0})"),
            markdown("## Data"),
            code("display(summary)"),
            markdown("## Results"),
            code("display(Image(filename=str(HERE/'figure_vanishing_point_candidate_counts.png')))\nmetrics.nlargest(10,'primary_automatic_confidence_score_not_probability')[['sample_id','group_label','automatic_vp_candidate_count','primary_support_line_count','primary_median_angular_residual_degrees']]"),
            markdown("## Takeaways\n\nUse these outputs to select and manually verify geometry-applicable cases. Do not report candidate counts as historical prevalence, and do not infer 'Chinese-Western hybridity' without artwork-level and art-historical evidence."),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": platform.python_version()}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (OUTPUT_DIR / "vanishing_point_candidate_analysis.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    existing = [str(OUTPUT_DIR / name) for name in OUTPUT_FILES if (OUTPUT_DIR / name).exists()]
    if OVERLAY_DIR.exists():
        existing.append(str(OVERLAY_DIR))
    if existing and not args.force:
        raise FileExistsError(f"refuse_to_overwrite_existing_outputs: {existing}")

    synthetic = synthetic_validation()
    manifest = load_manifest()
    metrics, vanishing_points, supporting_lines = process_manifest(manifest)
    group_summary = make_group_summary(metrics)
    review = metrics[["sample_id", "image_path", "group_id", "group_label", "automatic_status", "automatic_vp_candidate_count", "overlay_path"]].copy()
    for column in ["manual_applicability", "manual_vp_validity", "manual_vp_count", "reviewer", "reviewed_at", "review_note"]:
        review[column] = ""

    manifest.to_csv(OUTPUT_DIR / "analysis_manifest_snapshot.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT_DIR / "vanishing_point_metrics_per_image.csv", index=False, encoding="utf-8-sig")
    vanishing_points.to_csv(OUTPUT_DIR / "vanishing_points_long.csv", index=False, encoding="utf-8-sig")
    supporting_lines.to_csv(OUTPUT_DIR / "vanishing_point_supporting_lines.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(OUTPUT_DIR / "vanishing_point_group_summary.csv", index=False, encoding="utf-8-sig")
    review.to_csv(OUTPUT_DIR / "vanishing_point_review_template.csv", index=False, encoding="utf-8-sig")
    make_figures(metrics, group_summary)
    build_review_gallery(metrics)
    build_notebook()

    parameters = {
        "analysis_status": "automatic_candidate_screening_pending_human_review",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_counts": EXPECTED_COUNTS,
        "preprocessing": {"resize_long_edge_px": LONG_EDGE, "keep_aspect": True, "padding": False, "grayscale": "BT.709", "clahe_clip_limit": CLAHE_CLIP_LIMIT, "clahe_tile_grid": CLAHE_TILE_GRID},
        "line_detection": {"algorithm": "OpenCV LineSegmentDetector LSD_REFINE_STD", "minimum_length": f"max({MIN_LENGTH_PX}px,{MIN_LENGTH_DIAGONAL_FRACTION}*image_diagonal)", "same_border_exclusion_fraction": BORDER_LINE_EXCLUSION_FRACTION, "maximum_lines": MAX_LONG_LINES},
        "candidate_estimation": {"pair_minimum_angle_degrees": MIN_PAIR_ANGLE_DEGREES, "maximum_ransac_pair_intersections": MAX_RANSAC_CANDIDATES, "angular_support_tolerance_degrees": ANGULAR_SUPPORT_TOLERANCE_DEGREES, "refinement": "length-weighted least squares on normalized line equations", "maximum_candidates_per_image": MAX_VANISHING_POINTS, "random_seed": RANDOM_SEED},
        "acceptance": {"minimum_support_lines": MIN_SUPPORT_LINES, "minimum_active_support_length_ratio": MIN_ACTIVE_SUPPORT_LENGTH_RATIO, "minimum_total_support_length_ratio": MIN_TOTAL_SUPPORT_LENGTH_RATIO, "maximum_median_residual_degrees": MAX_MEDIAN_RESIDUAL_DEGREES, "maximum_center_distance_diagonals": MAX_VP_CENTER_DISTANCE_DIAGONALS},
        "confidence_warning": "heuristic ranking score, not calibrated probability",
        "synthetic_validation": synthetic,
        "manual_review_required": ["architectural-line applicability", "whether overlay lines share a meaningful geometric vanishing point", "exclude frames/text/scanning artifacts", "artwork-level interpretation"],
        "forbidden_inferences_before_review": ["absence of perspective from non-detection", "prevalence of 1-2 vanishing points", "Chinese-Western hybridity", "historical causality"],
        "software": {"python": sys.version, "platform": platform.platform(), "opencv": cv2.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    (OUTPUT_DIR / "vanishing_point_parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows = "\n".join(
        f"| {GROUP_SHORT[row.group_id]} | {int(row.n_candidate_images)} | {int(row.n_with_sufficient_long_lines)} | {int(row.n_automatic_candidate_detected)} | {int(row.n_with_one_candidate)} | {int(row.n_with_two_candidates)} |"
        for row in group_summary.itertuples(index=False)
    )
    validation = f"""# 消失点候选检测验证报告

## 总体判定

已建立可重跑的LSD长线段检测、交点RANSAC、加权最小二乘细化和叠加图导出流程。该流程对568张候选图进行自动筛查，但所有结果均标记为“待人工复核”，不能把自动候选直接视为作品真实消失点。

## 合成图算法检查

预设消失点为({synthetic['true_vp'][0]:.1f},{synthetic['true_vp'][1]:.1f})，检测点为({synthetic['detected_vp'][0]:.2f},{synthetic['detected_vp'][1]:.2f})，误差{synthetic['euclidean_error_px']:.3f}像素，合成验证通过。该检查证明实现能够恢复理想汇聚线，不代表其对历史版画天然有效。

## 自动筛查结果

| 候选组 | 图像数 | 长线段数量足够 | 有自动候选 | 1个候选 | 2个候选 |
|---|---:|---:|---:|---:|---:|
{rows}

共{int((metrics['automatic_vp_candidate_count'] > 0).sum())}张图产生至少一个自动候选，输出{len(vanishing_points)}个候选点和{len(supporting_lines)}条支持线；人工确认数目前为0。

## 解释限制

LSD会同时响应建筑轮廓、画框、题跋、器物边缘、扫描裁切和纸张破损。固定参数下没有候选不表示作品没有透视结构；有候选也可能只是非建筑线条的偶然交汇。正式论文只能在人工确认“建筑直线适用”且叠加图几何合理的子集上报告消失点，并需保存审核人和审核时间。“中西混生”仍须结合具体作品和艺术史来源，不能由消失点坐标单独推出。
"""
    (OUTPUT_DIR / "VANISHING_POINT_VALIDATION_REPORT.md").write_text(validation, encoding="utf-8")

    paper = f"""## 空间透视的自动候选检测与人工复核框架

本研究先对三组共568张候选图进行自动筛查。图像保持纵横比并将长边缩放至{LONG_EDGE}像素，转换为BT.709灰度后使用CLAHE增强局部对比度；随后采用OpenCV线段检测器提取长线段，排除紧贴图像外边界的线段。对线段两两交点进行确定性RANSAC候选搜索，以线段长度加权的角度残差计算支持度，再以标准化直线方程进行加权最小二乘细化。每图最多保留两个满足固定支持线数、支持长度比例和残差阈值的候选点。

自动筛查结果如下：

| 候选组 | 图像数 | 有自动候选 | 1个候选 | 2个候选 |
|---|---:|---:|---:|---:|
{chr(10).join(f'| {GROUP_SHORT[r.group_id]} | {int(r.n_candidate_images)} | {int(r.n_automatic_candidate_detected)} | {int(r.n_with_one_candidate)} | {int(r.n_with_two_candidates)} |' for r in group_summary.itertuples(index=False))}

以上只是算法候选清单，尚不能写成“三组作品中存在消失点的比例”，更不能写成“姑苏版画大多具有1—2个消失点”。下一步必须利用逐图叠加图，由研究者判断作品是否适合建筑透视分析、支持线是否来自同一空间结构，以及候选点是否有效。只有通过人工核验的作品才能进入空间透视统计；中西空间机制的解释还需结合代表作品细读和可核验的艺术史文献。
"""
    (OUTPUT_DIR / "PAPER_SECTION_PERSPECTIVE_CANDIDATE.md").write_text(paper, encoding="utf-8")

    receipt = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "manifest_rows": len(manifest),
        "unique_sample_ids": int(manifest["sample_id"].nunique()),
        "automatic_detected_images": int((metrics["automatic_vp_candidate_count"] > 0).sum()),
        "automatic_vanishing_point_candidates": len(vanishing_points),
        "supporting_line_rows": len(supporting_lines),
        "manual_verified_images": 0,
        "synthetic_error_px": synthetic["euclidean_error_px"],
        "status": "passed_pending_human_review",
    }
    (OUTPUT_DIR / "vanishing_point_run_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
