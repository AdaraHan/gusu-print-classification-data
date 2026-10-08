#!/usr/bin/env python3
"""
重新运行 EfficientNetV2-S 5-fold 交叉验证，并保存严格 OOF 结果。

本脚本基于原始 Step2_run_train_effnet_cv.py，保留以下实验口径：
  - class_to_idx 固定为 {"gusu": 0, "non_gusu": 1}
  - StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  - AspectResizePad 与原数据增强策略
  - 每个 fold 都从 timm 预训练权重重新初始化
  - Phase 1 冻结 backbone，Phase 2 解冻全部参数
  - 使用训练 fold 内类别权重
  - 以验证集 F1_gusu 选择最佳模型

新增：
  - 每折最佳模型权重与验证集索引
  - 每折逐图预测
  - 892 张图片的严格 OOF 预测与误判清单
  - 误判图片副本
  - 数据清单与 fold 归属
  - 与 2026-05-11 原五折汇总结果的核验
  - Grad-CAM 后续所需的模型、预处理和类别元数据

重要说明：
  本脚本得到的是“重新运行后的 OOF 误判清单”。即使汇总指标与原实验
  完全一致，也不能据此宣称逐图结果必然就是 2026-05-11 的原始 46 张误判。

建议用法（在 gusu_cls 根目录执行）：
    .venv/bin/python STEP6_run_train_effnet_cv_oof.py --preflight

    caffeinate -i .venv/bin/python STEP6_run_train_effnet_cv_oof.py \
        --data_dir data_v4_merged \
        --out_dir runs/effnetv2_v4_cv_oof \
        --reports_dir reports \
        --report_prefix effnetv2_v4 \
        --reference_results runs/effnetv2_v4_cv/cv_results.json \
        --copy_images
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
CLASS_TO_IDX = {"gusu": 0, "non_gusu": 1}
CSV_FIELDS = [
    "sample_index",
    "fold",
    "filepath",
    "filename",
    "true_label",
    "true_class",
    "pred_label",
    "pred_class",
    "prob_gusu",
    "prob_non_gusu",
    "pred_confidence",
    "prob_true_class",
    "is_misclassified",
    "error_type",
]


# ──────────────────────────────────────────────────────────
# 通用工具
# ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto"):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 --device cuda，但当前环境不可用 CUDA。")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("指定了 --device mps，但当前环境不可用 MPS。")
    return torch.device(requested)


def path_for_report(path: Path, project_root: Path):
    """优先返回相对项目根目录的可移植路径。"""
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return str(path.resolve())


def json_safe_args(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stat(vals):
    arr = np.array([v for v in vals if not np.isnan(v)], dtype=float)
    if len(arr) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "per_fold": [float(v) for v in vals],
        }
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "per_fold": [round(float(v), 4) for v in vals],
    }


def capture_cpu_state_dict(model):
    """把最佳权重复制到 CPU，降低训练设备上的额外显存/统一内存占用。"""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def release_device_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
    ):
        torch.mps.empty_cache()


# ──────────────────────────────────────────────────────────
# 预处理（与原脚本一致）
# ──────────────────────────────────────────────────────────

class EnsureRGB:
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return img.convert("RGB")


class AspectResizePad:
    def __init__(
        self,
        size=224,
        fill=(245, 245, 240),
        interpolation=Image.BILINEAR,
    ):
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
        return ImageOps.expand(
            img,
            border=(
                pad_w // 2,
                pad_h // 2,
                pad_w - pad_w // 2,
                pad_h - pad_h // 2,
            ),
            fill=self.fill,
        )


def build_transforms(img_size: int = 224, pad_fill=(245, 245, 240)):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        EnsureRGB(),
        AspectResizePad(size=img_size, fill=pad_fill),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8, fill=pad_fill),
        transforms.ColorJitter(
            brightness=0.08,
            contrast=0.08,
            saturation=0.06,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tf = transforms.Compose([
        EnsureRGB(),
        AspectResizePad(size=img_size, fill=pad_fill),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


# ──────────────────────────────────────────────────────────
# 数据集
# ──────────────────────────────────────────────────────────

def collect_samples(data_dir: Path):
    """
    扫描 data_dir/gusu/ 与 data_dir/non_gusu/。

    类别顺序和每类内部文件顺序与原脚本一致。
    """
    samples = []
    for class_name, label in CLASS_TO_IDX.items():
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"子目录不存在：{class_dir}")

        for file_path in sorted(class_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTS:
                samples.append((file_path, label))
    return samples, dict(CLASS_TO_IDX)


class ImageListDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        with Image.open(file_path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_splits(labels_arr, n_folds, seed):
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )
    return list(skf.split(np.zeros(len(labels_arr)), labels_arr))


def verify_dataset_counts(samples, args):
    labels = np.array([label for _, label in samples], dtype=np.int64)
    actual_total = len(samples)
    actual_gusu = int(np.sum(labels == CLASS_TO_IDX["gusu"]))
    actual_non_gusu = int(np.sum(labels == CLASS_TO_IDX["non_gusu"]))

    expected = {
        "总样本数": (actual_total, args.expected_samples),
        "gusu": (actual_gusu, args.expected_gusu),
        "non_gusu": (actual_non_gusu, args.expected_non_gusu),
    }
    mismatches = []
    for name, (actual, wanted) in expected.items():
        if wanted >= 0 and actual != wanted:
            mismatches.append(f"{name}: 实际 {actual}，预期 {wanted}")

    if mismatches:
        raise RuntimeError(
            "数据集数量与原实验不一致，已停止训练：\n  - "
            + "\n  - ".join(mismatches)
            + "\n如确需在新数据集上运行，请显式修改对应 --expected_* 参数。"
        )
    return labels


def dataset_order_digest(samples, project_root):
    """
    对“样本顺序 + 类别 + 相对路径 + 文件大小”生成审计摘要。

    该摘要用于发现文件清单或排序变化，不是图像内容哈希。
    """
    digest = hashlib.sha256()
    for sample_index, (file_path, label) in enumerate(samples):
        rel_path = path_for_report(file_path, project_root)
        size = file_path.stat().st_size
        line = f"{sample_index}\t{label}\t{rel_path}\t{size}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def build_dataset_manifest(samples, splits, project_root):
    fold_by_index = np.full(len(samples), -1, dtype=np.int64)
    for fold_idx, (_, val_indices) in enumerate(splits, start=1):
        if np.any(fold_by_index[val_indices] != -1):
            raise RuntimeError("fold 划分异常：同一样本被分配到多个验证 fold。")
        fold_by_index[val_indices] = fold_idx

    if np.any(fold_by_index == -1):
        raise RuntimeError("fold 划分异常：存在未被分配到验证 fold 的样本。")

    idx_to_class = {value: key for key, value in CLASS_TO_IDX.items()}
    rows = []
    for sample_index, (file_path, label) in enumerate(samples):
        rows.append({
            "sample_index": sample_index,
            "fold": int(fold_by_index[sample_index]),
            "filepath": path_for_report(file_path, project_root),
            "filename": file_path.name,
            "label": int(label),
            "class_name": idx_to_class[int(label)],
            "file_size_bytes": file_path.stat().st_size,
        })
    return rows, fold_by_index


# ──────────────────────────────────────────────────────────
# 模型与训练
# ──────────────────────────────────────────────────────────

def freeze_all_but_head(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if any(key in name for key in ["classifier", "fc", "head"]):
            parameter.requires_grad = True


def unfreeze_all(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, class_to_idx):
    """
    返回指标与逐图预测。

    AUC 以 gusu 为正类：先把 y 转为 is_gusu，再使用 P(gusu)。
    这只修正原脚本的 AUC 语义，不改变训练、最佳模型选择或其他指标。
    """
    gusu_idx = class_to_idx["gusu"]
    non_gusu_idx = class_to_idx["non_gusu"]
    criterion = nn.CrossEntropyLoss()

    all_targets = []
    all_preds = []
    all_prob_gusu = []
    all_prob_non_gusu = []
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            total_loss += criterion(logits, labels).item() * images.size(0)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_prob_gusu.extend(
                probs[:, gusu_idx].cpu().numpy().tolist()
            )
            all_prob_non_gusu.extend(
                probs[:, non_gusu_idx].cpu().numpy().tolist()
            )

    targets = np.asarray(all_targets, dtype=np.int64)
    preds = np.asarray(all_preds, dtype=np.int64)
    prob_gusu = np.asarray(all_prob_gusu, dtype=np.float64)
    prob_non_gusu = np.asarray(all_prob_non_gusu, dtype=np.float64)

    is_gusu = (targets == gusu_idx).astype(np.int64)
    try:
        auc = roc_auc_score(is_gusu, prob_gusu)
    except ValueError:
        auc = float("nan")

    metrics = {
        "loss": float(total_loss / len(loader.dataset)),
        "acc": float(accuracy_score(targets, preds)),
        "precision": float(precision_score(
            targets,
            preds,
            pos_label=gusu_idx,
            average="binary",
            zero_division=0,
        )),
        "recall": float(recall_score(
            targets,
            preds,
            pos_label=gusu_idx,
            average="binary",
            zero_division=0,
        )),
        "f1": float(f1_score(
            targets,
            preds,
            pos_label=gusu_idx,
            average="binary",
            zero_division=0,
        )),
        "auc": float(auc),
        "cm": confusion_matrix(
            targets,
            preds,
            labels=[class_to_idx["gusu"], class_to_idx["non_gusu"]],
        ).tolist(),
    }
    predictions = {
        "targets": targets,
        "preds": preds,
        "prob_gusu": prob_gusu,
        "prob_non_gusu": prob_non_gusu,
    }
    return metrics, predictions


def prediction_rows(
    sample_indices,
    fold_idx,
    samples,
    predictions,
    project_root,
):
    idx_to_class = {value: key for key, value in CLASS_TO_IDX.items()}
    rows = []

    for local_idx, sample_index in enumerate(sample_indices):
        file_path, stored_label = samples[int(sample_index)]
        true_label = int(predictions["targets"][local_idx])
        pred_label = int(predictions["preds"][local_idx])

        if true_label != int(stored_label):
            raise RuntimeError(
                f"样本 {sample_index} 的数据集标签与预测结果标签不一致。"
            )

        prob_gusu = float(predictions["prob_gusu"][local_idx])
        prob_non_gusu = float(predictions["prob_non_gusu"][local_idx])
        is_error = true_label != pred_label

        if not is_error:
            error_type = ""
        elif true_label == CLASS_TO_IDX["gusu"]:
            error_type = "gusu_to_non_gusu"
        else:
            error_type = "non_gusu_to_gusu"

        pred_confidence = (
            prob_gusu
            if pred_label == CLASS_TO_IDX["gusu"]
            else prob_non_gusu
        )
        prob_true_class = (
            prob_gusu
            if true_label == CLASS_TO_IDX["gusu"]
            else prob_non_gusu
        )

        rows.append({
            "sample_index": int(sample_index),
            "fold": int(fold_idx),
            "filepath": path_for_report(file_path, project_root),
            "filename": file_path.name,
            "true_label": true_label,
            "true_class": idx_to_class[true_label],
            "pred_label": pred_label,
            "pred_class": idx_to_class[pred_label],
            "prob_gusu": f"{prob_gusu:.8f}",
            "prob_non_gusu": f"{prob_non_gusu:.8f}",
            "pred_confidence": f"{pred_confidence:.8f}",
            "prob_true_class": f"{prob_true_class:.8f}",
            "is_misclassified": int(is_error),
            "error_type": error_type,
        })
    return rows


def train_fold(
    fold_idx,
    train_indices,
    val_indices,
    all_samples,
    class_to_idx,
    args,
    device,
    out_dir,
    project_root,
):
    try:
        import timm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "未安装 timm。请使用原项目的 .venv/bin/python 运行本脚本。"
        ) from exc

    pad_fill = (args.pad_fill, args.pad_fill, args.pad_fill)
    train_tf, eval_tf = build_transforms(args.img_size, pad_fill)

    train_samples = [all_samples[int(i)] for i in train_indices]
    val_samples = [all_samples[int(i)] for i in val_indices]

    train_ds = ImageListDataset(train_samples, transform=train_tf)
    val_ds = ImageListDataset(val_samples, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    train_labels = np.array(
        [label for _, label in train_samples],
        dtype=np.int64,
    )
    counts = np.bincount(train_labels, minlength=len(class_to_idx))
    if np.any(counts == 0):
        raise RuntimeError(
            f"Fold {fold_idx} 训练集存在空类别，counts={counts.tolist()}"
        )
    weights = len(train_labels) / (len(counts) * counts)
    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=len(class_to_idx),
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -1.0
    best_state = None
    best_phase = ""
    best_epoch = -1

    freeze_all_but_head(model)
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.lr_head,
    )
    print(
        f"\n  [Fold {fold_idx}] Phase 1 — "
        f"freeze backbone ({args.freeze_epochs} epochs)"
    )
    for epoch in range(1, args.freeze_epochs + 1):
        started = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        val_metrics, _ = evaluate(model, val_loader, device, class_to_idx)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = capture_cpu_state_dict(model)
            best_phase = "freeze_backbone"
            best_epoch = epoch
        print(
            f"    Ep{epoch}/{args.freeze_epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"({time.time() - started:.1f}s)"
        )

    unfreeze_all(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_ft)
    print(
        f"  [Fold {fold_idx}] Phase 2 — "
        f"fine-tune all ({args.ft_epochs} epochs)"
    )
    for epoch in range(1, args.ft_epochs + 1):
        started = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        val_metrics, _ = evaluate(model, val_loader, device, class_to_idx)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = capture_cpu_state_dict(model)
            best_phase = "fine_tune_all"
            best_epoch = epoch
        print(
            f"    Ep{epoch}/{args.ft_epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"({time.time() - started:.1f}s)"
        )

    if best_state is None:
        raise RuntimeError(
            "未产生最佳模型；freeze_epochs 与 ft_epochs 不能同时为 0。"
        )

    model.load_state_dict(best_state)
    final_metrics, predictions = evaluate(
        model,
        val_loader,
        device,
        class_to_idx,
    )
    rows = prediction_rows(
        val_indices,
        fold_idx,
        all_samples,
        predictions,
        project_root,
    )

    fold_dir = out_dir / "folds" / f"fold_{fold_idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=False)

    val_indices_path = fold_dir / "val_indices.npy"
    np.save(val_indices_path, np.asarray(val_indices, dtype=np.int64))

    fold_predictions_path = fold_dir / "val_predictions.csv"
    write_csv(fold_predictions_path, rows, CSV_FIELDS)

    checkpoint_path = fold_dir / "best_model.pt"
    checkpoint = {
        "checkpoint_format": "gusu_effnet_cv_oof_v1",
        "fold": int(fold_idx),
        "model_name": args.model,
        "num_classes": len(class_to_idx),
        "class_to_idx": dict(class_to_idx),
        "img_size": int(args.img_size),
        "pad_fill": int(args.pad_fill),
        "resize_policy": "AspectResizePad",
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "best_phase": best_phase,
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "metrics": final_metrics,
        "train_indices": [int(i) for i in train_indices],
        "val_indices": [int(i) for i in val_indices],
        "val_filepaths": [row["filepath"] for row in rows],
        "val_labels": [int(row["true_label"]) for row in rows],
        "state_dict": best_state,
    }
    torch.save(checkpoint, checkpoint_path)

    fold_meta = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"state_dict", "train_indices", "val_indices"}
    }
    fold_meta.update({
        "checkpoint": str(checkpoint_path.resolve()),
        "val_indices_file": str(val_indices_path.resolve()),
        "val_predictions_file": str(fold_predictions_path.resolve()),
        "n_train": len(train_indices),
        "n_val": len(val_indices),
    })
    with (fold_dir / "fold_results.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(fold_meta, f, ensure_ascii=False, indent=2)

    print(
        f"\n  [Fold {fold_idx}] Best F1_gusu={best_val_f1:.4f} "
        f"({best_phase}, epoch={best_epoch})"
    )
    print(
        f"  [Fold {fold_idx}] final acc={final_metrics['acc']:.4f} "
        f"precision={final_metrics['precision']:.4f} "
        f"recall={final_metrics['recall']:.4f} "
        f"f1={final_metrics['f1']:.4f} "
        f"auc={final_metrics['auc']:.4f}"
    )
    print(f"  [Fold {fold_idx}] 权重 → {checkpoint_path.resolve()}")
    print(f"  [Fold {fold_idx}] 预测 → {fold_predictions_path.resolve()}")

    del model
    release_device_cache(device)
    return final_metrics, predictions, rows


# ──────────────────────────────────────────────────────────
# 汇总、核验与误判图片复制
# ──────────────────────────────────────────────────────────

def compute_global_metrics(y_true, y_pred, prob_gusu):
    is_gusu = (y_true == CLASS_TO_IDX["gusu"]).astype(np.int64)
    try:
        auc = roc_auc_score(is_gusu, prob_gusu)
    except ValueError:
        auc = float("nan")

    return {
        "acc": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(
            y_true,
            y_pred,
            pos_label=CLASS_TO_IDX["gusu"],
            average="binary",
            zero_division=0,
        )), 6),
        "recall": round(float(recall_score(
            y_true,
            y_pred,
            pos_label=CLASS_TO_IDX["gusu"],
            average="binary",
            zero_division=0,
        )), 6),
        "f1": round(float(f1_score(
            y_true,
            y_pred,
            pos_label=CLASS_TO_IDX["gusu"],
            average="binary",
            zero_division=0,
        )), 6),
        "auc": round(float(auc), 6),
    }


def compare_with_reference(current_results, reference_path):
    comparison = {
        "reference_path": str(reference_path.resolve()),
        "reference_exists": reference_path.is_file(),
        "summary_match": None,
        "confusion_matrix_match": None,
        "metric_matches": {},
        "note": "",
    }
    if not reference_path.is_file():
        comparison["note"] = "未找到参考 JSON，跳过原汇总核验。"
        return comparison

    with reference_path.open(encoding="utf-8") as f:
        reference = json.load(f)

    current_cm = current_results["confusion_matrix_total"]
    reference_cm = reference.get("confusion_matrix_total")
    comparison["confusion_matrix_match"] = current_cm == reference_cm

    matches = {}
    for metric_name in ["acc", "precision", "recall", "f1"]:
        current_values = (
            current_results.get("metrics", {})
            .get(metric_name, {})
            .get("per_fold", [])
        )
        reference_values = (
            reference.get("metrics", {})
            .get(metric_name, {})
            .get("per_fold", [])
        )
        matches[metric_name] = current_values == reference_values
    comparison["metric_matches"] = matches

    comparison["summary_match"] = (
        comparison["confusion_matrix_match"]
        and all(matches.values())
    )
    if comparison["summary_match"]:
        comparison["note"] = (
            "本次逐折 acc/precision/recall/f1 与混淆矩阵均和参考汇总一致。"
        )
    else:
        comparison["note"] = (
            "本次结果未完全复现参考汇总；请将其作为新的复现实验结果，"
            "不要视为原始逐图误判的恢复。"
        )
    return comparison


def safe_image_copy_name(row):
    original = Path(row["filename"])
    stem = original.stem[:150]
    suffix = original.suffix.lower()
    transition = f"{row['true_class']}_to_{row['pred_class']}"
    return (
        f"{int(row['sample_index']):04d}"
        f"__fold{int(row['fold'])}"
        f"__{transition}"
        f"__{stem}{suffix}"
    )


def copy_misclassified_images(
    misclassified_rows,
    samples,
    destination,
):
    destination.mkdir(parents=True, exist_ok=False)
    copied = 0
    missing = 0
    copy_rows = []

    for row in misclassified_rows:
        sample_index = int(row["sample_index"])
        source = samples[sample_index][0]
        target = destination / safe_image_copy_name(row)

        status = "copied"
        if source.is_file():
            shutil.copy2(source, target)
            copied += 1
        else:
            status = "missing"
            missing += 1

        copy_rows.append({
            "sample_index": sample_index,
            "source_filepath": row["filepath"],
            "copied_filename": target.name,
            "status": status,
        })

    write_csv(
        destination / "copy_manifest.csv",
        copy_rows,
        [
            "sample_index",
            "source_filepath",
            "copied_filename",
            "status",
        ],
    )
    return copied, missing


def ensure_output_targets_are_new(
    out_dir,
    reports_dir,
    report_prefix,
    copy_images,
    reference_path,
):
    if out_dir.exists():
        raise FileExistsError(
            f"输出目录已存在：{out_dir}\n"
            "为避免混入旧结果，请换一个新的 --out_dir；脚本不会自动删除目录。"
        )

    if reference_path and out_dir.resolve() == reference_path.parent.resolve():
        raise RuntimeError(
            "新的 --out_dir 不能与参考实验目录相同，以免覆盖原结果。"
        )

    targets = [
        reports_dir / f"{report_prefix}_oof_predictions.csv",
        reports_dir / f"{report_prefix}_misclassified.csv",
    ]
    if copy_images:
        targets.append(
            reports_dir / f"{report_prefix}_misclassified_images"
        )
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "以下目标已存在，脚本为防止覆盖而停止：\n  - "
            + "\n  - ".join(existing)
            + "\n请改用新的 --report_prefix，或先人工确认并处理旧结果。"
        )


def print_preflight(
    args,
    samples,
    labels_arr,
    splits,
    project_root,
):
    idx_to_class = {value: key for key, value in CLASS_TO_IDX.items()}
    print("═" * 68)
    print("EfficientNet 五折 OOF 预检")
    print("═" * 68)
    print(f"项目目录       : {project_root}")
    print(f"数据目录       : {Path(args.data_dir).resolve()}")
    print(f"总样本数       : {len(samples)}")
    print(
        f"类别分布       : "
        f"gusu={int(np.sum(labels_arr == CLASS_TO_IDX['gusu']))}, "
        f"non_gusu={int(np.sum(labels_arr == CLASS_TO_IDX['non_gusu']))}"
    )
    print(f"类别编号       : {CLASS_TO_IDX}")
    print(f"K-folds        : {args.n_folds}")
    print(f"随机种子       : {args.seed}")
    print(f"模型           : {args.model}")
    print(f"冻结/微调轮数  : {args.freeze_epochs}/{args.ft_epochs}")
    print(f"batch_size     : {args.batch_size}")
    print()

    for fold_idx, (train_indices, val_indices) in enumerate(splits, start=1):
        train_counts = np.bincount(
            labels_arr[train_indices],
            minlength=len(CLASS_TO_IDX),
        )
        val_counts = np.bincount(
            labels_arr[val_indices],
            minlength=len(CLASS_TO_IDX),
        )
        train_text = ", ".join(
            f"{idx_to_class[idx]}={int(train_counts[idx])}"
            for idx in range(len(CLASS_TO_IDX))
        )
        val_text = ", ".join(
            f"{idx_to_class[idx]}={int(val_counts[idx])}"
            for idx in range(len(CLASS_TO_IDX))
        )
        print(
            f"Fold {fold_idx}: train={len(train_indices)} "
            f"({train_text}); val={len(val_indices)} ({val_text})"
        )

    print()
    print(f"训练输出       : {Path(args.out_dir).resolve()}")
    print(
        "OOF 预测       : "
        f"{(Path(args.reports_dir) / (args.report_prefix + '_oof_predictions.csv')).resolve()}"
    )
    print(
        "误判清单       : "
        f"{(Path(args.reports_dir) / (args.report_prefix + '_misclassified.csv')).resolve()}"
    )
    print(f"参考汇总       : {Path(args.reference_results).resolve()}")
    print()
    print("预检通过；本步骤未训练模型、未创建或修改任何输出文件。")


# ──────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────

def main(args):
    if args.freeze_epochs < 0 or args.ft_epochs < 0:
        raise ValueError("训练轮数不能为负数。")
    if args.freeze_epochs == 0 and args.ft_epochs == 0:
        raise ValueError("freeze_epochs 与 ft_epochs 不能同时为 0。")
    if args.n_folds < 2:
        raise ValueError("n_folds 必须至少为 2。")

    project_root = Path.cwd().resolve()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    reference_path = Path(args.reference_results)

    samples, class_to_idx = collect_samples(data_dir)
    labels_arr = verify_dataset_counts(samples, args)
    splits = build_splits(labels_arr, args.n_folds, args.seed)

    if args.preflight:
        print_preflight(
            args,
            samples,
            labels_arr,
            splits,
            project_root,
        )
        return

    ensure_output_targets_are_new(
        out_dir,
        reports_dir,
        args.report_prefix,
        args.copy_images,
        reference_path,
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    reports_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_device(args.device)

    manifest_rows, expected_fold_by_index = build_dataset_manifest(
        samples,
        splits,
        project_root,
    )
    manifest_path = out_dir / "dataset_manifest.csv"
    write_csv(
        manifest_path,
        manifest_rows,
        [
            "sample_index",
            "fold",
            "filepath",
            "filename",
            "label",
            "class_name",
            "file_size_bytes",
        ],
    )

    dataset_digest = dataset_order_digest(samples, project_root)
    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "data_dir": str(data_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "reports_dir": str(reports_dir.resolve()),
        "device": str(device),
        "class_to_idx": class_to_idx,
        "dataset_order_sha256": dataset_digest,
        "args": json_safe_args(args),
    }
    with (out_dir / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print("═" * 68)
    print("EfficientNetV2-S 5-fold OOF 训练")
    print("═" * 68)
    print(f"Using device       : {device}")
    print(f"Model              : {args.model}")
    print(f"data_dir           : {data_dir.resolve()}")
    print(f"out_dir            : {out_dir.resolve()}")
    print(f"freeze_epochs      : {args.freeze_epochs}")
    print(f"ft_epochs          : {args.ft_epochs}")
    print(f"n_folds            : {args.n_folds}")
    print(f"总样本数           : {len(samples)}")
    print(
        f"类别分布           : "
        f"gusu={int(np.sum(labels_arr == class_to_idx['gusu']))}, "
        f"non_gusu={int(np.sum(labels_arr == class_to_idx['non_gusu']))}"
    )
    print(f"dataset audit hash : {dataset_digest}")

    n_samples = len(samples)
    oof_fold = np.full(n_samples, -1, dtype=np.int64)
    oof_true = np.full(n_samples, -1, dtype=np.int64)
    oof_pred = np.full(n_samples, -1, dtype=np.int64)
    oof_prob_gusu = np.full(n_samples, np.nan, dtype=np.float64)
    oof_prob_non_gusu = np.full(n_samples, np.nan, dtype=np.float64)

    fold_metrics = {
        "acc": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc": [],
    }
    cm_total = np.zeros(
        (len(class_to_idx), len(class_to_idx)),
        dtype=np.int64,
    )
    all_rows = []

    for fold_idx, (train_indices, val_indices) in enumerate(
        splits,
        start=1,
    ):
        print(f"\n{'=' * 68}")
        print(
            f"Fold {fold_idx}/{args.n_folds} "
            f"train={len(train_indices)} val={len(val_indices)}"
        )
        print(f"{'=' * 68}")

        metrics, predictions, rows = train_fold(
            fold_idx,
            train_indices,
            val_indices,
            samples,
            class_to_idx,
            args,
            device,
            out_dir,
            project_root,
        )

        if np.any(oof_fold[val_indices] != -1):
            raise RuntimeError(
                f"Fold {fold_idx} 写入 OOF 时发现重复验证样本。"
            )

        oof_fold[val_indices] = fold_idx
        oof_true[val_indices] = predictions["targets"]
        oof_pred[val_indices] = predictions["preds"]
        oof_prob_gusu[val_indices] = predictions["prob_gusu"]
        oof_prob_non_gusu[val_indices] = predictions["prob_non_gusu"]
        all_rows.extend(rows)

        for metric_name in fold_metrics:
            fold_metrics[metric_name].append(metrics[metric_name])
        cm_total += np.asarray(metrics["cm"], dtype=np.int64)

    if np.any(oof_fold == -1):
        missing = np.flatnonzero(oof_fold == -1).tolist()
        raise RuntimeError(f"OOF 不完整，缺失样本索引：{missing}")
    if not np.array_equal(oof_fold, expected_fold_by_index):
        raise RuntimeError("OOF fold 编号与预先生成的数据清单不一致。")
    if not np.array_equal(oof_true, labels_arr):
        raise RuntimeError("OOF 真实标签与数据集标签不一致。")
    if np.isnan(oof_prob_gusu).any() or np.isnan(oof_prob_non_gusu).any():
        raise RuntimeError("OOF 概率存在缺失值。")

    all_rows.sort(key=lambda row: int(row["sample_index"]))
    if [int(row["sample_index"]) for row in all_rows] != list(range(n_samples)):
        raise RuntimeError("OOF CSV 行顺序或样本索引不完整。")

    oof_cm = confusion_matrix(
        oof_true,
        oof_pred,
        labels=[class_to_idx["gusu"], class_to_idx["non_gusu"]],
    )
    if not np.array_equal(oof_cm, cm_total):
        raise RuntimeError(
            "逐折累加混淆矩阵与 OOF 全量混淆矩阵不一致。"
        )

    global_metrics = compute_global_metrics(
        oof_true,
        oof_pred,
        oof_prob_gusu,
    )
    cv_results = {
        "model": args.model,
        "variant": "",
        "clf": "effnetv2_finetune_oof_rerun",
        "n_folds": args.n_folds,
        "seed": args.seed,
        "n_samples": n_samples,
        "n_dim": 0,
        "freeze_epochs": args.freeze_epochs,
        "ft_epochs": args.ft_epochs,
        "class_to_idx": class_to_idx,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_order_sha256": dataset_digest,
        "metrics": {
            name: stat(values)
            for name, values in fold_metrics.items()
        },
        "global_oof_metrics": global_metrics,
        "confusion_matrix_total": cm_total.tolist(),
    }
    comparison = compare_with_reference(cv_results, reference_path)
    cv_results["reference_comparison"] = comparison

    oof_csv_path = (
        reports_dir / f"{args.report_prefix}_oof_predictions.csv"
    )
    misclassified_csv_path = (
        reports_dir / f"{args.report_prefix}_misclassified.csv"
    )
    write_csv(oof_csv_path, all_rows, CSV_FIELDS)

    misclassified_rows = [
        row for row in all_rows
        if int(row["is_misclassified"]) == 1
    ]
    misclassified_rows.sort(
        key=lambda row: float(row["pred_confidence"]),
        reverse=True,
    )
    write_csv(
        misclassified_csv_path,
        misclassified_rows,
        CSV_FIELDS,
    )

    copied = None
    missing = None
    copied_dir = None
    if args.copy_images:
        copied_dir = (
            reports_dir
            / f"{args.report_prefix}_misclassified_images"
        )
        copied, missing = copy_misclassified_images(
            misclassified_rows,
            samples,
            copied_dir,
        )

    cv_results["outputs"] = {
        "dataset_manifest": str(manifest_path.resolve()),
        "oof_predictions": str(oof_csv_path.resolve()),
        "misclassified": str(misclassified_csv_path.resolve()),
        "misclassified_images": (
            str(copied_dir.resolve()) if copied_dir else None
        ),
    }
    cv_results_path = out_dir / "cv_results.json"
    with cv_results_path.open("w", encoding="utf-8") as f:
        json.dump(cv_results, f, ensure_ascii=False, indent=2)

    n_errors = len(misclassified_rows)
    gusu_to_non = int(
        np.sum(
            (oof_true == class_to_idx["gusu"])
            & (oof_pred == class_to_idx["non_gusu"])
        )
    )
    non_to_gusu = int(
        np.sum(
            (oof_true == class_to_idx["non_gusu"])
            & (oof_pred == class_to_idx["gusu"])
        )
    )

    summary_lines = [
        f"模型       : {args.model}",
        f"K-folds    : {args.n_folds}",
        f"样本总数   : {n_samples}",
        f"freeze_ep  : {args.freeze_epochs}",
        f"ft_ep      : {args.ft_epochs}",
        f"时间戳     : {cv_results['timestamp']}",
        f"正类       : gusu（idx={class_to_idx['gusu']}）",
        f"设备       : {device}",
        f"数据摘要   : {dataset_digest}",
        "",
        f"{'指标':<12}  {'mean':>8}  {'std':>8}  per-fold",
        "─" * 68,
    ]
    for metric_name, metric_stat in cv_results["metrics"].items():
        per_fold = "  ".join(
            f"{value:.4f}"
            for value in metric_stat["per_fold"]
        )
        summary_lines.append(
            f"{metric_name:<12}  "
            f"{metric_stat['mean']:>8.4f}  "
            f"{metric_stat['std']:>8.4f}  "
            f"{per_fold}"
        )
    summary_lines.extend([
        "",
        "累加混淆矩阵"
        "（rows=true, cols=pred, labels=[0,1]=[gusu,non_gusu]）：",
    ])
    for row in cm_total:
        summary_lines.append(
            "  " + "  ".join(f"{int(value):6d}" for value in row)
        )
    summary_lines.extend([
        "",
        f"OOF 样本数       : {n_samples}",
        f"误判总数         : {n_errors}",
        f"姑苏→非姑苏     : {gusu_to_non}",
        f"非姑苏→姑苏     : {non_to_gusu}",
        f"参考汇总核验     : {comparison['note']}",
        "",
        "说明：本清单属于本次重新训练结果，不等同于对旧实验逐图结果的恢复。",
    ])

    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print()
    print("═" * 68)
    print("5-fold 交叉验证结果（mean ± std，gusu 为正类）")
    print("═" * 68)
    for metric_name, metric_stat in cv_results["metrics"].items():
        print(
            f"  {metric_name:12s}: "
            f"{metric_stat['mean']:.4f} ± "
            f"{metric_stat['std']:.4f}"
        )
    print()
    print(
        "累加混淆矩阵"
        "（rows=true, cols=pred, labels=[0,1]=[gusu,non_gusu]）："
    )
    print(cm_total)
    print(f"OOF 样本数       : {n_samples}")
    print(f"误判总数         : {n_errors}")
    print(f"姑苏→非姑苏     : {gusu_to_non}")
    print(f"非姑苏→姑苏     : {non_to_gusu}")
    print(f"参考汇总核验     : {comparison['note']}")
    print()
    print(f"详细结果         → {cv_results_path.resolve()}")
    print(f"摘要文件         → {summary_path.resolve()}")
    print(f"数据清单         → {manifest_path.resolve()}")
    print(f"全部 OOF 预测    → {oof_csv_path.resolve()}")
    print(f"误判清单         → {misclassified_csv_path.resolve()}")
    if copied_dir:
        print(f"误判图片副本     → {copied_dir.resolve()}")
        print(f"成功复制/路径缺失: {copied}/{missing}")
    print()
    print(
        "注意：这是本次重新训练产生的逐图结果；"
        "即使参考汇总核验通过，也不要表述为旧 46 张误判的精确恢复。"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "EfficientNetV2-S 5-fold 交叉验证："
            "保存每折权重、严格 OOF 与误判清单"
        )
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data_v4_merged",
        help="图片目录，包含 gusu/ 与 non_gusu/（默认 data_v4_merged）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/effnetv2_v4_cv_oof",
        help="新的训练输出目录；必须不存在",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
        help="OOF 与误判报告目录（默认 reports）",
    )
    parser.add_argument(
        "--report_prefix",
        type=str,
        default="effnetv2_v4",
        help="报告文件名前缀（默认 effnetv2_v4）",
    )
    parser.add_argument(
        "--reference_results",
        type=str,
        default="runs/effnetv2_v4_cv/cv_results.json",
        help="旧实验 cv_results.json，仅用于汇总核验",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tf_efficientnetv2_s",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--pad_fill",
        type=int,
        default=245,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--ft_epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--lr_head",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--lr_ft",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="训练设备（默认 auto）",
    )
    parser.add_argument(
        "--expected_samples",
        type=int,
        default=892,
        help="原实验预期总样本数；设为 -1 可关闭此项核验",
    )
    parser.add_argument(
        "--expected_gusu",
        type=int,
        default=459,
        help="原实验预期 gusu 数量；设为 -1 可关闭此项核验",
    )
    parser.add_argument(
        "--expected_non_gusu",
        type=int,
        default=433,
        help="原实验预期 non_gusu 数量；设为 -1 可关闭此项核验",
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="将本次误判图片复制到 reports 下",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="只核验数据、参数与 fold 分布，不训练且不创建输出",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
