#!/usr/bin/env python3
"""Add a reproducible Sobel axial direction-concentration metric to Table 9.

This is a candidate/exploratory image-level analysis on the existing
459 + 9 + 100 manifest.  It does not estimate line continuity or breakage.
The primary metric is the magnitude-weighted axial mean resultant length:

    R = sqrt[(sum w cos(2 theta))^2 + (sum w sin(2 theta))^2] / sum w

where theta is the unsigned Sobel gradient orientation and pixels are retained
only when the normalized Sobel magnitude is at least 0.10 in the valid region.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
LINE_SCRIPT = HERE / "run_candidate_line_analysis.py"
TARGET_TABLE = HERE / "table9_candidate_exploratory.csv"
BACKUP_TABLE = HERE / "table9_candidate_exploratory_before_direction_concentration_20260729.csv"
PER_IMAGE_OUTPUT = HERE / "candidate_direction_concentration_per_image.csv"
SENSITIVITY_OUTPUT = HERE / "candidate_direction_concentration_sensitivity_per_image.csv"
SUMMARY_OUTPUT = HERE / "candidate_direction_concentration_summary.csv"
OMNIBUS_OUTPUT = HERE / "candidate_direction_concentration_omnibus_tests.csv"
PAIRWISE_OUTPUT = HERE / "candidate_direction_concentration_pairwise_tests.csv"
SENSITIVITY_OMNIBUS_OUTPUT = HERE / "candidate_direction_concentration_sensitivity_omnibus_tests.csv"
PARAMETERS_OUTPUT = HERE / "direction_concentration_parameters.json"
VALIDATION_OUTPUT = HERE / "DIRECTION_CONCENTRATION_VALIDATION.md"
RECEIPT_OUTPUT = HERE / "direction_concentration_run_receipt.json"

METRIC = "sobel_axial_gradient_direction_concentration"
NEW_COLUMNS = [
    f"{METRIC}_mean",
    f"{METRIC}_sd",
    f"{METRIC}_mean_sd",
]
EXPECTED_TABLE_COLUMNS = [
    "candidate_group",
    "n_images",
    "normalized_direction_entropy_mean",
    "normalized_direction_entropy_sd",
    "normalized_direction_entropy_mean_sd",
    "canny_edge_density_mean",
    "canny_edge_density_sd",
    "canny_edge_density_mean_sd_percent",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_line_module() -> Any:
    spec = importlib.util.spec_from_file_location("candidate_line_analysis", LINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot_import_existing_line_script: {LINE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def axial_direction_concentration(
    orientation_rad: np.ndarray,
    magnitude: np.ndarray,
    magnitude_normalized: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float,
) -> tuple[float, int, float]:
    """Return magnitude-weighted axial mean resultant length in [0, 1]."""
    selected = valid_mask & (magnitude_normalized >= threshold)
    selected_count = int(selected.sum())
    if selected_count == 0:
        return float("nan"), 0, 0.0
    theta = orientation_rad[selected].astype(np.float64, copy=False)
    weights = magnitude[selected].astype(np.float64, copy=False)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return float("nan"), selected_count, weight_sum
    cosine = float(np.sum(weights * np.cos(2.0 * theta)))
    sine = float(np.sum(weights * np.sin(2.0 * theta)))
    value = math.hypot(cosine, sine) / weight_sum
    if value < -1e-12 or value > 1.0 + 1e-12:
        raise ValueError(f"axial_concentration_out_of_range: {value}")
    return float(np.clip(value, 0.0, 1.0)), selected_count, weight_sum


def compute_metrics(line: Any, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []

    for index, row in manifest.iterrows():
        image_path = Path(row["resolved_image_path"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        observed_sha = line.sha256_file(image_path)
        if observed_sha.lower() != row["sha256"].lower():
            raise ValueError(f"sha256_mismatch: {row['sample_id']} {image_path}")

        original = line.read_image_rgb(image_path)
        resized = line.resize_long_edge(original)
        gray = line.rgb_to_bt709_gray(resized)
        gx = cv2.Sobel(
            gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101
        )
        gy = cv2.Sobel(
            gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101
        )
        magnitude = cv2.magnitude(gx, gy)
        magnitude_normalized = np.clip(
            magnitude / line.SOBEL_THEORETICAL_MAX, 0.0, 1.0
        )
        orientation_rad = np.mod(np.arctan2(gy, gx), np.pi)

        for border_fraction in line.BORDER_EXCLUSIONS:
            valid_mask = line.valid_mask_for(gray.shape, border_fraction)
            valid_pixel_count = int(valid_mask.sum())
            sensitivity_row: dict[str, Any] = {
                "sample_id": row["sample_id"],
                "group_id": row["group_id"],
                "group_label": row["group_label"],
                "group_order": int(row["group_order"]),
                "source_prefix": row["source_prefix"],
                "border_exclusion_fraction": border_fraction,
                "valid_pixel_count": valid_pixel_count,
            }
            for threshold in line.SOBEL_THRESHOLDS:
                value, selected_count, selected_weight_sum = axial_direction_concentration(
                    orientation_rad,
                    magnitude,
                    magnitude_normalized,
                    valid_mask,
                    threshold,
                )
                key = f"t{int(round(threshold * 100)):03d}"
                sensitivity_row[f"{METRIC}_{key}"] = value
                sensitivity_row[f"selected_gradient_pixel_count_{key}"] = selected_count
                sensitivity_row[f"selected_gradient_weight_sum_{key}"] = selected_weight_sum
            sensitivity_rows.append(sensitivity_row)

            if math.isclose(border_fraction, line.PRIMARY_BORDER_EXCLUSION):
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
                        "analysis_width": int(resized.shape[1]),
                        "analysis_height": int(resized.shape[0]),
                        "valid_pixel_count": valid_pixel_count,
                        "selected_gradient_pixel_count": sensitivity_row[
                            "selected_gradient_pixel_count_t010"
                        ],
                        "selected_gradient_weight_sum": sensitivity_row[
                            "selected_gradient_weight_sum_t010"
                        ],
                        METRIC: sensitivity_row[f"{METRIC}_t010"],
                        "analysis_scope": "candidate_exploratory_not_formal_manifest",
                    }
                )

        if (index + 1) % 100 == 0 or index + 1 == len(manifest):
            print(f"processed {index + 1}/{len(manifest)}", flush=True)

    primary = pd.DataFrame(primary_rows).sort_values(
        ["group_order", "sample_id"]
    ).reset_index(drop=True)
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values(
        ["group_order", "sample_id", "border_exclusion_fraction"]
    ).reset_index(drop=True)
    return primary, sensitivity


def sensitivity_omnibus(line: Any, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for border_fraction in line.BORDER_EXCLUSIONS:
        subset = frame[np.isclose(frame["border_exclusion_fraction"], border_fraction)]
        for threshold in line.SOBEL_THRESHOLDS:
            key = f"t{int(round(threshold * 100)):03d}"
            metric = f"{METRIC}_{key}"
            arrays = [
                subset.loc[subset["group_id"] == group_id, metric]
                .dropna()
                .to_numpy(dtype=float)
                for group_id in line.GROUP_ORDER
            ]
            statistic, p_value = stats.kruskal(*arrays)
            rows.append(
                {
                    "metric": metric,
                    "sobel_normalized_magnitude_threshold": threshold,
                    "border_exclusion_fraction": border_fraction,
                    "test": "Kruskal-Wallis",
                    "statistic_H": float(statistic),
                    "degrees_of_freedom": 2,
                    "p_value": float(p_value),
                    "analysis_unit": "candidate_image",
                }
            )
    return pd.DataFrame(rows)


def validate_target_table(line: Any) -> tuple[list[str], list[dict[str, str]]]:
    with TARGET_TABLE.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if fieldnames != EXPECTED_TABLE_COLUMNS:
        raise ValueError(f"unexpected_table_columns: {fieldnames}")
    if len(rows) != 3:
        raise ValueError(f"table_expected_3_rows: {len(rows)}")
    expected = {
        line.GROUP_SHORT[group_id]: str(line.EXPECTED_COUNTS[group_id])
        for group_id in line.GROUP_ORDER
    }
    observed = {row["candidate_group"]: row["n_images"] for row in rows}
    if observed != expected:
        raise ValueError(f"table_group_counts_mismatch: {observed}")
    return fieldnames, rows


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temp_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    os.replace(temp_path, path)


def merge_target_table(
    line: Any,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    summary: pd.DataFrame,
) -> None:
    summary_by_short: dict[str, tuple[float, float, int]] = {}
    for group_id in line.GROUP_ORDER:
        result = summary[
            (summary["group_id"] == group_id) & (summary["metric"] == METRIC)
        ]
        if len(result) != 1:
            raise ValueError(f"missing_group_summary: {group_id}")
        record = result.iloc[0]
        summary_by_short[line.GROUP_SHORT[group_id]] = (
            float(record["mean"]),
            float(record["sd"]),
            int(record["n"]),
        )

    insert_at = fieldnames.index("normalized_direction_entropy_mean_sd") + 1
    merged_fields = fieldnames[:insert_at] + NEW_COLUMNS + fieldnames[insert_at:]
    merged_rows: list[dict[str, str]] = []
    for row in rows:
        mean, sd, n = summary_by_short[row["candidate_group"]]
        if n != int(row["n_images"]):
            raise ValueError(
                f"summary_count_mismatch: {row['candidate_group']} table={row['n_images']} computed={n}"
            )
        merged = dict(row)
        merged[NEW_COLUMNS[0]] = repr(mean)
        merged[NEW_COLUMNS[1]] = repr(sd)
        merged[NEW_COLUMNS[2]] = f"{mean:.3f} ± {sd:.3f}"
        merged_rows.append(merged)

    temp_path = TARGET_TABLE.with_name(f".{TARGET_TABLE.name}.tmp-{os.getpid()}")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged_rows)
    os.replace(temp_path, TARGET_TABLE)


def main() -> None:
    generated = [
        BACKUP_TABLE,
        PER_IMAGE_OUTPUT,
        SENSITIVITY_OUTPUT,
        SUMMARY_OUTPUT,
        OMNIBUS_OUTPUT,
        PAIRWISE_OUTPUT,
        SENSITIVITY_OMNIBUS_OUTPUT,
        PARAMETERS_OUTPUT,
        VALIDATION_OUTPUT,
        RECEIPT_OUTPUT,
    ]
    existing = [str(path) for path in generated if path.exists()]
    if existing:
        raise FileExistsError(f"refuse_to_overwrite_generated_files: {existing}")
    if not TARGET_TABLE.exists():
        raise FileNotFoundError(TARGET_TABLE)

    line = load_line_module()
    original_hash = sha256_file(TARGET_TABLE)
    fieldnames, original_rows = validate_target_table(line)
    manifest = line.load_candidate_manifest(line.DEFAULT_MANIFEST)
    primary, sensitivity = compute_metrics(line, manifest)

    if len(primary) != 568 or primary["sample_id"].nunique() != 568:
        raise ValueError(
            f"primary_expected_568_unique_rows: rows={len(primary)} unique={primary['sample_id'].nunique()}"
        )
    if primary[METRIC].isna().any():
        raise ValueError("primary_metric_contains_missing_values")
    if not primary[METRIC].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("primary_metric_outside_0_1")
    observed_counts = primary.groupby("group_id").size().to_dict()
    if observed_counts != line.EXPECTED_COUNTS:
        raise ValueError(f"primary_group_counts_mismatch: {observed_counts}")

    summary = line.summarize_metric_rows(primary, [METRIC])
    omnibus, pairwise = line.inferential_tests(primary, [METRIC])
    sensitivity_tests = sensitivity_omnibus(line, sensitivity)

    shutil.copy2(TARGET_TABLE, BACKUP_TABLE)
    write_csv_atomic(PER_IMAGE_OUTPUT, primary)
    write_csv_atomic(SENSITIVITY_OUTPUT, sensitivity)
    write_csv_atomic(SUMMARY_OUTPUT, summary)
    write_csv_atomic(OMNIBUS_OUTPUT, omnibus)
    write_csv_atomic(PAIRWISE_OUTPUT, pairwise)
    write_csv_atomic(SENSITIVITY_OMNIBUS_OUTPUT, sensitivity_tests)

    parameters = {
        "analysis_status": "candidate_exploratory_not_formal_manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(line.DEFAULT_MANIFEST.relative_to(PROJECT_ROOT)),
        "target_table": str(TARGET_TABLE.relative_to(PROJECT_ROOT)),
        "analysis_unit": "candidate_image",
        "candidate_counts": line.EXPECTED_COUNTS,
        "formula": (
            "sqrt((sum_i w_i cos(2 theta_i))^2 + "
            "(sum_i w_i sin(2 theta_i))^2) / sum_i w_i"
        ),
        "orientation": {
            "quantity": "unsigned Sobel gradient orientation",
            "domain_degrees": [0, 180],
            "axial_doubling": "2*theta",
            "note": "gradient orientation is perpendicular to local line orientation, but axial concentration is invariant to the 90-degree rotation",
        },
        "weight": "raw Sobel gradient magnitude sqrt(Gx^2+Gy^2)",
        "primary_pixel_inclusion": {
            "normalized_sobel_magnitude_minimum": line.PRIMARY_SOBEL_THRESHOLD,
            "valid_region": "full resized image excluding one-pixel perimeter",
        },
        "range": [0, 1],
        "interpretation": {
            "higher": "selected gradient energy is more concentrated in fewer axial directions",
            "lower": "selected gradient energy is more dispersed across axial directions",
            "excluded": [
                "line continuity",
                "line breakage rate",
                "craft quality",
                "aesthetic superiority",
                "causal regional-style attribution",
            ],
        },
        "preprocessing_inherited_from": str(LINE_SCRIPT.relative_to(PROJECT_ROOT)),
        "resize_long_edge_px": line.LONG_EDGE,
        "grayscale": "BT.709",
        "sobel_kernel_size": 3,
        "sobel_border_type": "BORDER_REFLECT101",
        "sobel_normalization": "divide by sqrt(2)*4*255 and clip to [0,1]",
        "sensitivity": {
            "normalized_magnitude_thresholds": list(line.SOBEL_THRESHOLDS),
            "border_exclusion_fractions": list(line.BORDER_EXCLUSIONS),
        },
        "statistics": {
            "summary": "image-level mean and sample SD (ddof=1), plus bootstrap mean CI and distribution statistics",
            "omnibus": "Kruskal-Wallis",
            "pairwise": "two-sided Mann-Whitney U with Holm adjustment within metric",
        },
    }
    PARAMETERS_OUTPUT.write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    merge_target_table(line, fieldnames, original_rows, summary)
    merged_hash = sha256_file(TARGET_TABLE)

    # Independent post-merge reconciliation against per-image values.
    merged = pd.read_csv(TARGET_TABLE, encoding="utf-8-sig")
    if len(merged) != 3 or list(merged.columns) != (
        EXPECTED_TABLE_COLUMNS[:5] + NEW_COLUMNS + EXPECTED_TABLE_COLUMNS[5:]
    ):
        raise ValueError("post_merge_table_shape_or_columns_mismatch")
    for row in original_rows:
        merged_row = merged[merged["candidate_group"] == row["candidate_group"]]
        if len(merged_row) != 1:
            raise ValueError(f"post_merge_missing_group: {row['candidate_group']}")
        for column in EXPECTED_TABLE_COLUMNS:
            if str(merged_row.iloc[0][column]) != str(pd.read_csv(BACKUP_TABLE, encoding="utf-8-sig").loc[
                lambda frame: frame["candidate_group"] == row["candidate_group"], column
            ].iloc[0]):
                # Numeric CSV parsing can normalize textual float representations;
                # the stronger byte-level row preservation check follows below.
                if column not in {
                    "normalized_direction_entropy_mean",
                    "normalized_direction_entropy_sd",
                    "canny_edge_density_mean",
                    "canny_edge_density_sd",
                }:
                    raise ValueError(f"original_column_changed: {row['candidate_group']} {column}")

    with BACKUP_TABLE.open("r", encoding="utf-8-sig", newline="") as handle:
        backup_rows = list(csv.DictReader(handle))
    with TARGET_TABLE.open("r", encoding="utf-8-sig", newline="") as handle:
        merged_rows = list(csv.DictReader(handle))
    for before, after in zip(backup_rows, merged_rows):
        for column in EXPECTED_TABLE_COLUMNS:
            if before[column] != after[column]:
                raise ValueError(f"original_text_changed: {before['candidate_group']} {column}")

    summary_display = []
    for group_id in line.GROUP_ORDER:
        record = summary[
            (summary["group_id"] == group_id) & (summary["metric"] == METRIC)
        ].iloc[0]
        summary_display.append(
            f"| {line.GROUP_SHORT[group_id]} | {int(record['n'])} | "
            f"{record['mean']:.6f} | {record['sd']:.6f} | "
            f"{record['mean']:.3f} ± {record['sd']:.3f} |"
        )

    validation = f"""# Sobel轴向梯度方向集中度合并验证

## 结论

已在既有459＋9＋100张候选图像上，沿用原线条实验的缩放、灰度化、Sobel核、梯度归一化、阈值和有效区域，计算Sobel轴向梯度方向集中度，并将组均值、样本标准差及“均值 ± 标准差”合并到`table9_candidate_exploratory.csv`。

该指标不是线条连续性、断裂率或工艺质量指标；合并后的表仍属于候选样本探索性结果，不是人工冻结manifest上的正式作品级推断。

## 指标定义

对每幅图，令 $\\theta_i$ 为未定向Sobel梯度方向（0°—180°），$w_i$ 为原始Sobel梯度幅值。仅纳入有效区域内归一化梯度幅值不低于0.10的像素：

$$
R=\\frac{{\\sqrt{{(\\sum_i w_i\\cos 2\\theta_i)^2+(\\sum_i w_i\\sin 2\\theta_i)^2}}}}{{\\sum_i w_i}}.
$$

$R\\in[0,1]$。数值越高仅表示入选梯度能量越集中于少数轴向方向；数值越低表示方向越分散。梯度方向与局部线条方向相差90°，但轴向集中度对整体90°旋转不变。

## 主分析结果

| 候选组 | 图像数 | 均值 | 样本标准差 | 表中显示值 |
|---|---:|---:|---:|---:|
{chr(10).join(summary_display)}

## 完整性检查

- 候选manifest：568个唯一sample_id；组数为459、9、100。
- 每张图的路径存在，且重新计算的SHA-256与manifest一致。
- 主分析逐图指标无缺失，全部落在[0,1]。
- 统计单位为候选图像；未自动处理同版异图、组画拆分或近重复。
- 合并前表格已备份；原有8列逐字段文本保持不变。
- 主要阈值为0.10；另保存0.08、0.10、0.12与边界排除0%、5%、10%的敏感性逐图结果及Kruskal-Wallis诊断。

## 解释边界

本指标描述整幅候选图像中强梯度的全局方向集中程度。它会受到画面构图、图框、文字、扫描裁切、背景以及同版异图的影响；尚未经过人工方向标注或外部效度验证。因此可以报告为“轴向梯度方向集中度”，不得改写为“线条连续性强”“断裂率低”“刻版更规整”或区域风格的因果证明。
"""
    VALIDATION_OUTPUT.write_text(validation, encoding="utf-8")

    receipt = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "target_table": str(TARGET_TABLE),
        "backup_table": str(BACKUP_TABLE),
        "target_sha256_before": original_hash,
        "backup_sha256": sha256_file(BACKUP_TABLE),
        "target_sha256_after": merged_hash,
        "rows_computed": len(primary),
        "unique_sample_ids": int(primary["sample_id"].nunique()),
        "new_columns": NEW_COLUMNS,
        "validation_status": "passed",
    }
    RECEIPT_OUTPUT.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
