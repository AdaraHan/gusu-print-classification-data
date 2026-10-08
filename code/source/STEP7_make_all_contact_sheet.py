#!/usr/bin/env python3
"""
Generate one contact sheet for the final Gusu classification dataset.

Expected directory layout:

    Step2_input_data_v4_merged/
    ├── gusu/
    └── non_gusu/

The script follows symbolic links, preserves each image's aspect ratio, and
places Gusu and non-Gusu samples in separate labelled sections.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成最终实验数据集的全部图像缩略总览图。"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("Step2_input_data_v4_merged"),
        help="包含 gusu/ 和 non_gusu/ 的数据目录。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/Step2_all_892_contact_sheet.png"),
        help="输出 PNG 路径。",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=28,
        help="每行缩略图数量（默认：28）。",
    )
    parser.add_argument(
        "--thumb_size",
        type=int,
        default=150,
        help="每个正方形缩略图区域的边长，单位为像素（默认：150）。",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=6,
        help="缩略图之间的间距，单位为像素（默认：6）。",
    )
    parser.add_argument(
        "--expected_gusu",
        type=int,
        default=459,
        help="预期姑苏图像数量；设为 -1 可跳过数量校验。",
    )
    parser.add_argument(
        "--expected_non_gusu",
        type=int,
        default=433,
        help="预期非姑苏图像数量；设为 -1 可跳过数量校验。",
    )
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"类别目录不存在：{directory}")

    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda path: str(path.relative_to(directory)).casefold())


def validate_count(label: str, files: list[Path], expected: int) -> None:
    if expected >= 0 and len(files) != expected:
        raise RuntimeError(
            f"{label} 数量不符：实际 {len(files)} 张，预期 {expected} 张。"
            "请确认 --data_dir 指向最终892张实验数据集。"
        )


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def section_height(
    count: int,
    *,
    columns: int,
    thumb_size: int,
    gap: int,
    header_height: int,
) -> int:
    rows = math.ceil(count / columns)
    return header_height + rows * thumb_size + max(0, rows - 1) * gap


def make_thumbnail(path: Path, thumb_size: int) -> Image.Image:
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source)
        source = source.convert("RGB")
        source.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)

        tile = Image.new("RGB", (thumb_size, thumb_size), "white")
        x = (thumb_size - source.width) // 2
        y = (thumb_size - source.height) // 2
        tile.paste(source, (x, y))
        return tile


def paste_section(
    canvas: Image.Image,
    files: Iterable[Path],
    *,
    title: str,
    top: int,
    columns: int,
    thumb_size: int,
    gap: int,
    margin: int,
    header_height: int,
    accent: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> int:
    files = list(files)
    draw = ImageDraw.Draw(canvas)

    draw.rounded_rectangle(
        (
            margin,
            top,
            canvas.width - margin,
            top + header_height - gap,
        ),
        radius=10,
        fill=accent,
    )
    text_box = draw.textbbox((0, 0), title, font=font)
    text_height = text_box[3] - text_box[1]
    draw.text(
        (margin + 18, top + (header_height - gap - text_height) // 2),
        title,
        fill="white",
        font=font,
    )

    grid_top = top + header_height
    failures: list[tuple[Path, str]] = []

    for index, path in enumerate(files):
        row, column = divmod(index, columns)
        x = margin + column * (thumb_size + gap)
        y = grid_top + row * (thumb_size + gap)
        try:
            tile = make_thumbnail(path, thumb_size)
        except Exception as exc:  # report every unreadable image together
            failures.append((path, str(exc)))
            continue
        canvas.paste(tile, (x, y))

    if failures:
        details = "\n".join(f"- {path}: {error}" for path, error in failures)
        raise RuntimeError(f"以下图像无法读取，未生成总览图：\n{details}")

    return top + section_height(
        len(files),
        columns=columns,
        thumb_size=thumb_size,
        gap=gap,
        header_height=header_height,
    )


def main() -> None:
    args = parse_args()
    if args.columns <= 0 or args.thumb_size <= 0 or args.gap < 0:
        raise ValueError("--columns 和 --thumb_size 必须大于0，--gap 不能小于0。")

    data_dir = args.data_dir.expanduser().resolve()
    output = args.output.expanduser()
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()

    gusu_files = image_files(data_dir / "gusu")
    non_gusu_files = image_files(data_dir / "non_gusu")
    validate_count("姑苏图像", gusu_files, args.expected_gusu)
    validate_count("非姑苏图像", non_gusu_files, args.expected_non_gusu)

    margin = 30
    header_height = 64
    section_gap = 36
    width = (
        2 * margin
        + args.columns * args.thumb_size
        + (args.columns - 1) * args.gap
    )
    gusu_height = section_height(
        len(gusu_files),
        columns=args.columns,
        thumb_size=args.thumb_size,
        gap=args.gap,
        header_height=header_height,
    )
    non_gusu_height = section_height(
        len(non_gusu_files),
        columns=args.columns,
        thumb_size=args.thumb_size,
        gap=args.gap,
        header_height=header_height,
    )
    height = 2 * margin + gusu_height + section_gap + non_gusu_height

    canvas = Image.new("RGB", (width, height), (242, 242, 239))
    font = load_font(30)

    next_top = paste_section(
        canvas,
        gusu_files,
        title=f"姑苏图像  Gusu  (n={len(gusu_files)})",
        top=margin,
        columns=args.columns,
        thumb_size=args.thumb_size,
        gap=args.gap,
        margin=margin,
        header_height=header_height,
        accent=(153, 57, 53),
        font=font,
    )
    paste_section(
        canvas,
        non_gusu_files,
        title=f"非姑苏图像  Non-Gusu  (n={len(non_gusu_files)})",
        top=next_top + section_gap,
        columns=args.columns,
        thumb_size=args.thumb_size,
        gap=args.gap,
        margin=margin,
        header_height=header_height,
        accent=(61, 86, 110),
        font=font,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)

    total = len(gusu_files) + len(non_gusu_files)
    print(f"已生成：{output}")
    print(
        f"图像总数：{total}（姑苏 {len(gusu_files)}；"
        f"非姑苏 {len(non_gusu_files)}）"
    )
    print(f"画布尺寸：{width} × {height} px")


if __name__ == "__main__":
    main()
