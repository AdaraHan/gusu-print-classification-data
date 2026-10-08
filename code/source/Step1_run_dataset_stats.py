# dataset_stats.py
"""
数据集质检脚本。数据收集完成后第一步运行，确认质量再训练。

功能（按顺序执行）：
  1. 各子目录图片计数
  2. 损坏文件检测（PIL 无法打开）
  3. 小尺寸检测（任意边 < min_size px）
  4. 非 RGB 检测（L / P / RGBA 等）
  5. 精确重复检测（MD5）
  6. 近似重复检测（自实现 aHash，汉明距离 < phash_threshold）
  7. 纵横比分布（竖版 / 横版 / 近方形）
  8. 终端摘要 + JSON 报告

用法示例：
    python dataset_stats.py \\
        --data_dir /path/to/图片-韩玉凤 \\
        --min_size 200 \\
        --phash_threshold 8 \\
        --out reports/dataset_stats.json
"""
import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image


# ──────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}


def iter_images(root: Path):
    """递归遍历 root，按子目录分组返回 (subdir_name, file_path)。
    subdir_name 取第一级子目录名（相对于 root）。"""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fname in sorted(filenames):
            if Path(fname).suffix.lower() in IMAGE_EXTS:
                fpath = Path(dirpath) / fname
                # 第一级子目录名（作为类别标识）
                rel = fpath.relative_to(root)
                subdir = rel.parts[0] if len(rel.parts) > 1 else "_root_"
                yield subdir, fpath


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ahash(img: Image.Image) -> str:
    """
    自实现 aHash（8×8 灰度均值哈希，64-bit 二进制字符串）。
    无需第三方 imagehash 库。
    """
    small = img.convert("L").resize((8, 8), Image.BILINEAR)
    pixels = list(small.getdata())           # 64 个像素值
    mean   = sum(pixels) / len(pixels)
    bits   = "".join("1" if p >= mean else "0" for p in pixels)
    return bits


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


# ──────────────────────────────────────────────────────────
# 各检测步骤
# ──────────────────────────────────────────────────────────

def step_count(root: Path):
    """统计各子目录的图片数量。"""
    counts = defaultdict(int)
    all_files = []
    for subdir, fpath in iter_images(root):
        counts[subdir] += 1
        all_files.append((subdir, fpath))
    return dict(counts), all_files


def step_corrupted(all_files):
    """检测 PIL 无法打开的文件。"""
    corrupted = []
    for _, fpath in all_files:
        try:
            with Image.open(fpath) as img:
                img.verify()
        except Exception as e:
            corrupted.append({"path": str(fpath), "error": str(e)})
    return corrupted


def step_tiny(all_files, min_size: int):
    """检测任意边 < min_size 的图像。"""
    tiny = []
    for _, fpath in all_files:
        try:
            with Image.open(fpath) as img:
                w, h = img.size
            if w < min_size or h < min_size:
                tiny.append({"path": str(fpath), "width": w, "height": h})
        except Exception:
            pass  # corrupted 已在上一步记录
    return tiny


def step_non_rgb(all_files):
    """检测模式不为 RGB 的图像。"""
    non_rgb = []
    for _, fpath in all_files:
        try:
            with Image.open(fpath) as img:
                mode = img.mode
            if mode != "RGB":
                non_rgb.append({"path": str(fpath), "mode": mode})
        except Exception:
            pass
    return non_rgb


def step_exact_dupes(all_files):
    """MD5 精确重复检测（全局）。"""
    hash_to_paths = defaultdict(list)
    for _, fpath in all_files:
        try:
            h = md5_of_file(fpath)
            hash_to_paths[h].append(str(fpath))
        except Exception:
            pass
    groups = [paths for paths in hash_to_paths.values() if len(paths) > 1]
    return groups


def step_near_dupes(all_files, threshold: int):
    """
    aHash 近似重复检测（仅在同一子目录内比较，避免跨类误报）。
    汉明距离 < threshold 则视为疑似重复对。
    """
    # 按子目录分组
    subdir_to_files = defaultdict(list)
    for subdir, fpath in all_files:
        subdir_to_files[subdir].append(fpath)

    near_pairs = []
    for subdir, fpaths in subdir_to_files.items():
        # 计算 aHash
        hashes = []
        for fp in fpaths:
            try:
                with Image.open(fp) as img:
                    h = ahash(img)
                hashes.append((fp, h))
            except Exception:
                pass

        # O(n²) 比较（同目录内）
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                fp_a, h_a = hashes[i]
                fp_b, h_b = hashes[j]
                dist = hamming(h_a, h_b)
                if dist < threshold:
                    near_pairs.append({
                        "subdir"  : subdir,
                        "file_a"  : str(fp_a),
                        "file_b"  : str(fp_b),
                        "hamming" : dist,
                    })

    # 按汉明距离排序（越小越相似在前）
    near_pairs.sort(key=lambda x: x["hamming"])
    return near_pairs


def step_aspect_ratio(all_files):
    """统计纵横比分布（竖版 / 横版 / 近方形，以及各子目录分布）。"""
    portrait  = 0   # h > w * 1.1
    landscape = 0   # w > h * 1.1
    square    = 0   # 否则（近方形）
    total     = 0

    for _, fpath in all_files:
        try:
            with Image.open(fpath) as img:
                w, h = img.size
        except Exception:
            continue
        total += 1
        if h > w * 1.1:
            portrait += 1
        elif w > h * 1.1:
            landscape += 1
        else:
            square += 1

    return {
        "total"    : total,
        "portrait" : portrait,
        "landscape": landscape,
        "square"   : square,
        "portrait_ratio" : round(portrait  / total, 4) if total else 0,
        "landscape_ratio": round(landscape / total, 4) if total else 0,
        "square_ratio"   : round(square    / total, 4) if total else 0,
    }


# ──────────────────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────────────────

def main(args):
    root = Path(args.data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"data_dir 不存在：{root}")

    print(f"▶ 扫描目录：{root.resolve()}")
    print(f"  min_size={args.min_size}px  phash_threshold={args.phash_threshold}")
    print()

    # Step 1: 计数 & 收集文件列表
    print("[1/7] 统计各子目录图片数量 ...")
    counts, all_files = step_count(root)
    total_images = sum(counts.values())
    for subdir, cnt in sorted(counts.items()):
        print(f"  {subdir:30s}  {cnt:5d} 张")
    print(f"  {'合计':30s}  {total_images:5d} 张")
    print()

    # Step 2: 损坏检测
    print("[2/7] 检测损坏文件 ...")
    corrupted = step_corrupted(all_files)
    print(f"  损坏文件：{len(corrupted)} 个")
    for item in corrupted[:5]:
        print(f"    {item['path']}  ({item['error']})")
    if len(corrupted) > 5:
        print(f"    ... 共 {len(corrupted)} 个（完整列表见 JSON）")
    print()

    # Step 3: 小尺寸检测
    print(f"[3/7] 检测小尺寸图像（任意边 < {args.min_size}px）...")
    tiny = step_tiny(all_files, args.min_size)
    print(f"  小尺寸图像：{len(tiny)} 个")
    for item in tiny[:5]:
        print(f"    {item['path']}  ({item['width']}×{item['height']})")
    if len(tiny) > 5:
        print(f"    ... 共 {len(tiny)} 个")
    print()

    # Step 4: 非 RGB 检测
    print("[4/7] 检测非 RGB 图像 ...")
    non_rgb = step_non_rgb(all_files)
    print(f"  非 RGB 图像：{len(non_rgb)} 个")
    for item in non_rgb[:5]:
        print(f"    {item['path']}  (mode={item['mode']})")
    if len(non_rgb) > 5:
        print(f"    ... 共 {len(non_rgb)} 个")
    print()

    # Step 5: 精确重复
    print("[5/7] 检测精确重复（MD5）...")
    exact_dupes = step_exact_dupes(all_files)
    n_exact_images = sum(len(g) for g in exact_dupes)
    print(f"  重复组数：{len(exact_dupes)}  涉及图片：{n_exact_images} 张")
    for group in exact_dupes[:3]:
        print(f"    {group}")
    if len(exact_dupes) > 3:
        print(f"    ... 共 {len(exact_dupes)} 组")
    print()

    # Step 6: 近似重复
    print(f"[6/7] 检测近似重复（aHash，汉明距离 < {args.phash_threshold}）...")
    near_dupes = step_near_dupes(all_files, args.phash_threshold)
    print(f"  疑似重复对：{len(near_dupes)} 对")
    for pair in near_dupes[:5]:
        print(f"    hamming={pair['hamming']}  {Path(pair['file_a']).name} ↔ {Path(pair['file_b']).name}")
    if len(near_dupes) > 5:
        print(f"    ... 共 {len(near_dupes)} 对")
    print()

    # Step 7: 纵横比分布
    print("[7/7] 统计纵横比分布 ...")
    aspect = step_aspect_ratio(all_files)
    print(f"  竖版（portrait） : {aspect['portrait']:5d}  ({aspect['portrait_ratio']:.1%})")
    print(f"  横版（landscape）: {aspect['landscape']:5d}  ({aspect['landscape_ratio']:.1%})")
    print(f"  近方形（square） : {aspect['square']:5d}  ({aspect['square_ratio']:.1%})")
    print()

    # ── 汇总摘要 ──
    summary = {
        "data_dir"       : str(root.resolve()),
        "timestamp"      : datetime.now().isoformat(timespec="seconds"),
        "total_images"   : total_images,
        "class_counts"   : counts,
        "corrupted_count": len(corrupted),
        "tiny_count"     : len(tiny),
        "non_rgb_count"  : len(non_rgb),
        "exact_dupe_groups": len(exact_dupes),
        "near_dupe_pairs"  : len(near_dupes),
        "aspect_ratio"   : aspect,
    }

    report = {
        "summary"   : summary,
        "corrupted" : corrupted,
        "tiny"      : tiny,
        "non_rgb"   : non_rgb,
        "exact_dupes": exact_dupes,
        "near_dupes" : near_dupes,
    }

    # ── 保存 JSON ──
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"✅ 报告已保存 → {out_path.resolve()}")
    print()
    print("═" * 60)
    print("摘要")
    print("═" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据集质检脚本")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="图片根目录（支持多级子目录）"
    )
    parser.add_argument(
        "--min_size", type=int, default=200,
        help="最小边长阈值（px），小于此值标记为 tiny（默认 200）"
    )
    parser.add_argument(
        "--phash_threshold", type=int, default=8,
        help="aHash 汉明距离阈值，小于此值视为疑似重复（默认 8）"
    )
    parser.add_argument(
        "--out", type=str, default="reports/dataset_stats.json",
        help="JSON 报告输出路径（默认 reports/dataset_stats.json）"
    )
    args = parser.parse_args()
    main(args)
