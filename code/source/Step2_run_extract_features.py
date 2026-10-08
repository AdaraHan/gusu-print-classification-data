# extract_features.py
"""
特征提取脚本：冻结 backbone，批量提取所有图像的特征向量，保存为 .npz。

支持两种模型：
  - effnetv2  : timm EfficientNetV2-S，num_classes=0 → 1280-dim pooled features
  - dinov2    : torch.hub DINOv2，CLS token → 768-dim（vitb14）等

支持两种目录结构：
  - 双类模式（默认）  : data_dir/gusu/ + data_dir/non_gusu/
  - 原始多类模式      : 用 --pos_class 指定正类目录名，其余合并为负类

用法示例：
    # EfficientNetV2
    python extract_features.py \\
        --data_dir data/all_labeled \\
        --model effnetv2 \\
        --out_dir features/effnetv2

    # DINOv2
    python extract_features.py \\
        --data_dir data/all_labeled \\
        --model dinov2 --dinov2_variant vitb14 \\
        --out_dir features/dinov2

    # 原始多类目录 + 指定正类
    python extract_features.py \\
        --data_dir /path/to/图片-韩玉凤 \\
        --pos_class 姑苏 \\
        --model dinov2 \\
        --out_dir features/dinov2
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm


# ──────────────────────────────────────────────────────────
# 与 train_effnet.py 完全一致的预处理组件
# ──────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EnsureRGB:
    """确保所有图像都转成 RGB"""
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return img.convert("RGB")


class AspectResizePad:
    """保持长宽比缩放，并补边到目标尺寸。"""
    def __init__(self, size=224, fill=(245, 245, 240), interpolation=Image.BILINEAR):
        if isinstance(size, int):
            self.target_w = size
            self.target_h = size
        else:
            self.target_w, self.target_h = size
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError("遇到空图像，宽或高为 0。")
        scale = min(self.target_w / w, self.target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), self.interpolation)
        pad_w = self.target_w - new_w
        pad_h = self.target_h - new_h
        left   = pad_w // 2
        right  = pad_w - left
        top    = pad_h // 2
        bottom = pad_h - top
        img = ImageOps.expand(img, border=(left, top, right, bottom), fill=self.fill)
        return img


def build_eval_transform(img_size=224, pad_fill=(245, 245, 240)):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        EnsureRGB(),
        AspectResizePad(size=img_size, fill=pad_fill),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ──────────────────────────────────────────────────────────
# 自定义 Dataset（支持双类 / 多类→二分类）
# ──────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def collect_samples(data_dir: Path, pos_class: str | None):
    """
    返回 (samples, class_to_idx)。
    samples: list of (file_path, label_int)
    class_to_idx: {"gusu": 1, "non_gusu": 0} 或原始双类映射
    """
    subdirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not subdirs:
        raise FileNotFoundError(f"data_dir 中没有找到子目录：{data_dir}")

    subdir_names = [d.name for d in subdirs]

    if pos_class is not None:
        # 原始多类模式：pos_class → 1，其余 → 0
        if pos_class not in subdir_names:
            raise ValueError(
                f"--pos_class '{pos_class}' 在 data_dir 中未找到。\n"
                f"可用子目录：{subdir_names}"
            )
        class_to_idx = {"non_gusu": 0, "gusu": 1}
        label_map = {
            d.name: (1 if d.name == pos_class else 0)
            for d in subdirs
        }
    else:
        # 标准双类模式
        required = {"gusu", "non_gusu"}
        if not required.issubset(set(subdir_names)):
            raise ValueError(
                f"未检测到 gusu/non_gusu 子目录，也未指定 --pos_class。\n"
                f"可用子目录：{subdir_names}\n"
                f"请用 --pos_class <正类目录名> 指定正类。"
            )
        class_to_idx = {"non_gusu": 0, "gusu": 1}
        label_map = {"gusu": 1, "non_gusu": 0}

    samples = []
    for d in subdirs:
        label = label_map.get(d.name, 0)
        for fpath in sorted(d.rglob("*")):
            if fpath.suffix.lower() in IMAGE_EXTS:
                samples.append((fpath, label))

    return samples, class_to_idx


class ImageListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples   # list of (Path, int)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = Image.open(fpath).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, str(fpath)


# ──────────────────────────────────────────────────────────
# 模型构建
# ──────────────────────────────────────────────────────────

def build_effnetv2(device):
    """EfficientNetV2-S，去掉分类头，输出 1280-dim pooled features。"""
    model = timm.create_model("tf_efficientnetv2_s", pretrained=True, num_classes=0)
    model.eval()
    model.to(device)
    return model


def build_dinov2(variant: str, device):
    """
    DINOv2，加载 torch.hub 版本。
    variant: vitb14 / vits14 / vitl14 / vitg14
    输出 CLS token：model.forward_features(x)[:, 0, :]
    """
    model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{variant}")
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def extract_features(model, loader, device, model_type: str):
    """批量提取特征，返回 (X, y, filenames)。"""
    # 尝试导入 tqdm
    try:
        from tqdm import tqdm
        wrap = tqdm
    except ImportError:
        def wrap(x, **kw):
            total = kw.get("total", "?")
            print(f"提取特征中（共约 {total} 批）...")
            return x

    all_feats = []
    all_labels = []
    all_fnames = []

    for batch_imgs, batch_labels, batch_fnames in wrap(loader, total=len(loader), desc="extracting"):
        batch_imgs = batch_imgs.to(device)

        if model_type == "effnetv2":
            feats = model(batch_imgs)                          # (B, 1280)
        else:  # dinov2
            out = model.forward_features(batch_imgs)
            # hub 版本输出形状：(B, seq_len, dim)，取 CLS token
            if out.ndim == 3:
                feats = out[:, 0, :]                           # (B, dim)
            else:
                feats = out                                    # fallback

        all_feats.append(feats.cpu().float().numpy())
        all_labels.extend(batch_labels.numpy().tolist())
        all_fnames.extend(batch_fnames)

    X = np.concatenate(all_feats, axis=0).astype(np.float32)  # (N, D)
    y = np.array(all_labels, dtype=np.int64)
    return X, y, all_fnames


# ──────────────────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────────────────

def main(args):
    device   = get_device()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device : {device}")
    print(f"Model        : {args.model}" + (f" ({args.dinov2_variant})" if args.model == "dinov2" else ""))
    print(f"data_dir     : {data_dir.resolve()}")
    print(f"img_size     : {args.img_size}")
    print()

    # ── 收集样本 ──
    pos_class = args.pos_class if args.pos_class else None
    samples, class_to_idx = collect_samples(data_dir, pos_class)

    label_counts = {}
    for _, lbl in samples:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print(f"总样本数 : {len(samples)}")
    for cls, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        print(f"  {cls}（{idx}）: {label_counts.get(idx, 0)} 张")
    print()

    # ── 预处理 ──
    pad_fill = (args.pad_fill, args.pad_fill, args.pad_fill)
    tf = build_eval_transform(args.img_size, pad_fill)

    dataset = ImageListDataset(samples, transform=tf)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # ── 构建模型 ──
    print("加载模型 ...")
    if args.model == "effnetv2":
        model = build_effnetv2(device)
        model_label  = "effnetv2"
        variant_label = "tf_efficientnetv2_s"
    else:  # dinov2
        model = build_dinov2(args.dinov2_variant, device)
        model_label  = "dinov2"
        variant_label = args.dinov2_variant
    print("模型已加载。\n")

    # ── 提取特征 ──
    X, y, filenames = extract_features(model, loader, device, args.model)
    n_samples, n_dim = X.shape

    print(f"\n特征提取完成：X.shape={X.shape}  y.shape={y.shape}")

    # ── 保存 .npz ──
    npz_path = out_dir / "features.npz"
    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        filenames=np.array(filenames, dtype=object),
    )
    print(f"特征已保存 → {npz_path.resolve()}")

    # ── 保存 meta.json ──
    meta = {
        "model"        : model_label,
        "variant"      : variant_label,
        "img_size"     : args.img_size,
        "pad_fill"     : args.pad_fill,
        "n_samples"    : int(n_samples),
        "n_dim"        : int(n_dim),
        "class_to_idx" : class_to_idx,
        "label_counts" : {str(k): int(v) for k, v in label_counts.items()},
        "timestamp"    : datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"元数据已保存 → {meta_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="特征提取脚本（EfficientNetV2 / DINOv2）")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="图片目录（gusu/ + non_gusu/ 或配合 --pos_class 使用原始目录）"
    )
    parser.add_argument(
        "--model", type=str, required=True, choices=["effnetv2", "dinov2"],
        help="特征提取模型：effnetv2 | dinov2"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="输出目录（保存 features.npz + meta.json）"
    )
    parser.add_argument(
        "--dinov2_variant", type=str, default="vitb14",
        choices=["vits14", "vitb14", "vitl14", "vitg14"],
        help="DINOv2 变体（默认 vitb14）"
    )
    parser.add_argument(
        "--pos_class", type=str, default=None,
        help="正类目录名（原始多类目录模式，e.g. 姑苏）"
    )
    parser.add_argument("--img_size",   type=int, default=224, help="输入图像尺寸（默认 224）")
    parser.add_argument("--pad_fill",   type=int, default=245, help="补边颜色 0-255（默认 245）")
    parser.add_argument("--batch_size", type=int, default=16,  help="批大小（默认 16）")
    args = parser.parse_args()
    main(args)
