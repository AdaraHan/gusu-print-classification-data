#!/usr/bin/env python3
"""Reproducible Sobel/Canny edge-detection QC pipeline.

This module deliberately produces per-image QC material only.  It does not
aggregate by art-historical group and therefore cannot reproduce or validate
the manuscript's current Table 9 before the formal manifest is frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np


# Headless rendering is required in terminal/CI environments on macOS.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / (
    "reports/formal_aesthetics_reproduction_20260727/"
    "formal_analysis_manifest.csv"
)
SOBEL_THEORETICAL_MAX = math.sqrt(2.0) * 4.0 * 255.0


def read_image_rgb(path: Path) -> np.ndarray:
    """Read Unicode paths reliably and return an RGB uint8 image."""
    encoded = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"无法读取图像: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_long_edge(image: np.ndarray, long_edge: int) -> np.ndarray:
    """Resize proportionally without padding or distortion."""
    height, width = image.shape[:2]
    scale = long_edge / max(height, width)
    if math.isclose(scale, 1.0):
        return image.copy()
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def rgb_to_bt709_gray(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to 8-bit BT.709 luminance."""
    rgb = image_rgb.astype(np.float32)
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def calculate_edges(
    image_rgb: np.ndarray,
    *,
    sobel_threshold: float,
    canny_low: int,
    canny_high: int,
    gaussian_kernel: int,
    gaussian_sigma: float,
) -> dict[str, np.ndarray | float]:
    """Return continuous Sobel responses and two explicit binary edge masks."""
    gray = rgb_to_bt709_gray(image_rgb)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)
    magnitude = cv2.magnitude(gx, gy)
    magnitude_normalized = np.clip(magnitude / SOBEL_THEORETICAL_MAX, 0.0, 1.0)
    sobel_binary = magnitude_normalized >= sobel_threshold

    blurred = cv2.GaussianBlur(
        gray,
        (gaussian_kernel, gaussian_kernel),
        sigmaX=gaussian_sigma,
        sigmaY=gaussian_sigma,
        borderType=cv2.BORDER_REFLECT101,
    )
    canny = cv2.Canny(
        blurred,
        threshold1=canny_low,
        threshold2=canny_high,
        apertureSize=3,
        L2gradient=True,
    )
    canny_binary = canny > 0

    # The formal experiment will replace this with the frozen content mask.
    # A one-pixel perimeter is excluded so convolution boundaries do not enter
    # the QC density denominator.
    valid_mask = np.ones_like(gray, dtype=bool)
    if gray.shape[0] > 2 and gray.shape[1] > 2:
        valid_mask[[0, -1], :] = False
        valid_mask[:, [0, -1]] = False

    valid_count = int(valid_mask.sum())
    sobel_ratio = float(np.logical_and(sobel_binary, valid_mask).sum() / valid_count)
    canny_density = float(np.logical_and(canny_binary, valid_mask).sum() / valid_count)

    return {
        "gray": gray,
        "gx": gx,
        "gy": gy,
        "magnitude": magnitude,
        "magnitude_normalized": magnitude_normalized,
        "sobel_binary": sobel_binary,
        "canny_binary": canny_binary,
        "valid_mask": valid_mask,
        "sobel_strong_gradient_ratio": sobel_ratio,
        "canny_edge_density": canny_density,
    }


def choose_qc_sample(manifest_path: Path) -> tuple[str, Path, dict[str, str]]:
    """Select the first Stage 1 manifest row, without outcome-based cherry-picking."""
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"manifest为空: {manifest_path}")
    row = rows[0]
    image_path = Path(row["image_path"])
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    return row["sample_id"], image_path, row


def save_png(path: Path, array: np.ndarray, *, rgb: bool = False) -> None:
    """Write uint8/boolean arrays to a Unicode path."""
    if array.dtype == bool:
        output = array.astype(np.uint8) * 255
    elif np.issubdtype(array.dtype, np.floating):
        output = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    else:
        output = array
    if rgb:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", output)
    if not ok:
        raise ValueError(f"PNG编码失败: {path}")
    buffer.tofile(path)


def make_overlay(image_rgb: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
    overlay = image_rgb.astype(np.float32)
    edge_color = np.array([220.0, 38.0, 38.0], dtype=np.float32)
    overlay[edge_mask] = 0.25 * overlay[edge_mask] + 0.75 * edge_color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def chinese_font() -> FontProperties | None:
    candidates = (
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for candidate in candidates:
        if candidate.exists():
            return FontProperties(fname=str(candidate))
    return None


def save_panel(
    output_path: Path,
    image_rgb: np.ndarray,
    products: dict[str, np.ndarray | float],
    overlay: np.ndarray,
    sample_id: str,
) -> None:
    font = chinese_font()
    title_kwargs: dict[str, Any] = {"fontproperties": font} if font else {}
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    panels = (
        (image_rgb, "① 原图（等比例缩放）", None, None, None),
        (products["gray"], "② BT.709灰度图", "gray", 0, 255),
        (products["magnitude_normalized"], "③ Sobel梯度幅值（连续量）", "magma", 0, 1),
        (products["sobel_binary"], "④ Sobel阈值后的强梯度像素", "gray", 0, 1),
        (products["canny_binary"], "⑤ Canny二值边缘", "gray", 0, 1),
        (overlay, "⑥ Canny边缘叠加（红色）", None, None, None),
    )
    for axis, (panel, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        axis.imshow(panel, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title, fontsize=13, **title_kwargs)
        axis.axis("off")
    fig.suptitle(
        f"边缘检测流程质控示例｜{sample_id}",
        fontsize=16,
        **title_kwargs,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="单幅输入图像；省略时取Stage 1 manifest首行")
    parser.add_argument("--sample-id", help="输入图像的稳定ID")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--long-edge", type=int, default=512)
    parser.add_argument("--sobel-threshold", type=float, default=0.10)
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--gaussian-kernel", type=int, default=5)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许覆盖本工具此前生成的同名QC文件",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.long_edge <= 0:
        raise ValueError("--long-edge必须为正整数")
    if not 0.0 < args.sobel_threshold < 1.0:
        raise ValueError("--sobel-threshold必须在(0,1)内")
    if not 0 <= args.canny_low < args.canny_high <= 255:
        raise ValueError("Canny阈值必须满足0 ≤ low < high ≤ 255")
    if args.gaussian_kernel <= 0 or args.gaussian_kernel % 2 == 0:
        raise ValueError("--gaussian-kernel必须为正奇数")
    if args.gaussian_sigma < 0:
        raise ValueError("--gaussian-sigma不得为负")


def main() -> None:
    args = parse_args()
    validate_args(args)

    selection_note: dict[str, Any]
    if args.input:
        image_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
        sample_id = args.sample_id or image_path.stem
        selection_note = {"rule": "explicit_cli_input"}
    else:
        manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
        sample_id, image_path, row = choose_qc_sample(manifest_path)
        selection_note = {
            "rule": "first_row_of_stage1_manifest_no_outcome_selection",
            "manifest": str(manifest_path),
            "proposed_group_candidate_only": row.get("proposed_group", ""),
        }

    if not image_path.exists():
        raise FileNotFoundError(image_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        "original": args.output_dir / "qc_01_original.png",
        "gray": args.output_dir / "qc_02_grayscale_bt709.png",
        "sobel": args.output_dir / "qc_03_sobel_magnitude.png",
        "sobel_binary": args.output_dir / "qc_04_sobel_threshold.png",
        "canny": args.output_dir / "qc_05_canny_edges.png",
        "overlay": args.output_dir / "qc_06_canny_overlay.png",
        "panel": args.output_dir / "edge_detection_qc_panel.png",
        "metadata": args.output_dir / "edge_detection_qc_metadata.json",
    }
    existing = [path for path in output_files.values() if path.exists()]
    if existing and not args.overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"目标文件已存在；如确认覆盖本工具输出请加--overwrite:\n{joined}")

    original = read_image_rgb(image_path)
    resized = resize_long_edge(original, args.long_edge)
    products = calculate_edges(
        resized,
        sobel_threshold=args.sobel_threshold,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        gaussian_kernel=args.gaussian_kernel,
        gaussian_sigma=args.gaussian_sigma,
    )
    overlay = make_overlay(resized, products["canny_binary"])

    save_png(output_files["original"], resized, rgb=True)
    save_png(output_files["gray"], products["gray"])
    save_png(output_files["sobel"], products["magnitude_normalized"])
    save_png(output_files["sobel_binary"], products["sobel_binary"])
    save_png(output_files["canny"], products["canny_binary"])
    save_png(output_files["overlay"], overlay, rgb=True)
    save_panel(output_files["panel"], resized, products, overlay, sample_id)

    metadata = {
        "status": "illustrative_qc_only_not_formal_group_result",
        "sample_id": sample_id,
        "source_path": str(image_path),
        "selection": selection_note,
        "original_size_px": {"width": int(original.shape[1]), "height": int(original.shape[0])},
        "analysis_size_px": {"width": int(resized.shape[1]), "height": int(resized.shape[0])},
        "preprocessing": {
            "resize": f"long_edge={args.long_edge}, keep_aspect_ratio, no_padding",
            "grayscale": "BT.709: 0.2126R+0.7152G+0.0722B",
            "valid_mask": "QC only: full resized image minus 1-pixel perimeter",
            "formal_mask_warning": "正式实验须使用冻结的有效内容掩膜，排除补边、画框和背景。",
        },
        "sobel": {
            "kernel": "3x3",
            "magnitude": "sqrt(Gx^2+Gy^2)",
            "normalization": f"fixed theoretical scale sqrt(2)*4*255={SOBEL_THEORETICAL_MAX:.10f}",
            "threshold": args.sobel_threshold,
            "strong_gradient_ratio_qc": products["sobel_strong_gradient_ratio"],
            "interpretation": "连续梯度经固定阈值二值化后的强梯度像素比例；不是Canny边缘密度。",
        },
        "canny": {
            "gaussian_kernel": args.gaussian_kernel,
            "gaussian_sigma": args.gaussian_sigma,
            "low_threshold": args.canny_low,
            "high_threshold": args.canny_high,
            "aperture_size": 3,
            "l2gradient": True,
            "edge_density_qc": products["canny_edge_density"],
            "formula": "count(Canny_edge & valid_mask) / count(valid_mask)",
        },
        "publication_warning": (
            "50/150与0.10是QC默认值，不得为迎合摘要或表9选取；须在独立QC样本上冻结后，"
            "对正式manifest全部图像重算。当前表9数值不能由本示例追认。"
        ),
    }
    output_files["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"outputs": {k: str(v) for k, v in output_files.items()}, **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
