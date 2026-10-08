#!/usr/bin/env python3
"""
导出 DINOv2 特征分类器的逐图片 OOF（out-of-fold）预测结果。

默认复现已核实的原始实验：
  - features/dinov2_v4/features.npz
  - StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  - 每个 fold 仅在训练集上拟合 StandardScaler
  - RBF-SVM / LogReg 均使用 class_weight="balanced"

原始 SVM 脚本：
  pipeline_svm/train_svm.py
  SHA-256: 6303cdee056e1b61f7385d58b1baef5798ed4a229a7580b8a11067163f021641

默认输出：
  reports/dinov2_v4_rbf_oof_predictions.csv
  reports/dinov2_v4_rbf_misclassified.csv
  reports/dinov2_v4_logreg_oof_predictions.csv
  reports/dinov2_v4_logreg_misclassified.csv
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


EXPECTED_RBF_CM = np.array([[389, 44], [27, 432]], dtype=int)
EXPECTED_RBF_FOLD_ACC = [0.9218, 0.9274, 0.9101, 0.9101, 0.9326]


def build_classifier(clf_name: str):
    """使用已核实的原始实验参数构造分类器。"""
    if clf_name == "svm_rbf":
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )
    if clf_name == "logreg":
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
    raise ValueError(f"未知分类器：{clf_name}")


def load_class_names(features_path: Path):
    """优先从 meta.json 读取类别映射；缺失时使用本实验已确认的映射。"""
    meta_path = features_path.parent / "meta.json"
    meta = {}
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)

    class_to_idx = meta.get("class_to_idx", {})
    idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}

    # features.npz 已确认：gusu 样本的 y=1。
    idx_to_class.setdefault(0, "non_gusu")
    idx_to_class.setdefault(1, "gusu")
    return idx_to_class, meta


def resolve_image_path(project_root: Path, filename: str):
    """
    解析 features.npz 中记录的相对路径。

    兼容数据目录后来从 data_v4_merged 改名为
    Step2_input_data_v4_merged 的情况。
    """
    original = Path(filename)
    candidates = [
        project_root / original,
    ]

    parts = original.parts
    if parts:
        tail = Path(*parts[1:]) if len(parts) > 1 else Path(original.name)
        if parts[0] == "data_v4_merged":
            candidates.append(project_root / "Step2_input_data_v4_merged" / tail)
        elif parts[0] == "Step2_input_data_v4_merged":
            candidates.append(project_root / "data_v4_merged" / tail)

    # 最后再按“类别目录 + 文件名”尝试两个常见数据目录。
    if len(parts) >= 2:
        class_dir = parts[-2]
        candidates.extend(
            [
                project_root / "data_v4_merged" / class_dir / original.name,
                project_root / "Step2_input_data_v4_merged" / class_dir / original.name,
            ]
        )

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return str(candidate.resolve()), True

    return str((project_root / original).resolve()), False


def error_type(true_idx: int, pred_idx: int):
    if true_idx == pred_idx:
        return "correct"
    if true_idx == 0 and pred_idx == 1:
        return "false_positive_non_gusu_to_gusu"
    if true_idx == 1 and pred_idx == 0:
        return "false_negative_gusu_to_non_gusu"
    return f"class_{true_idx}_to_class_{pred_idx}"


def run_oof(
    X,
    y,
    filenames,
    clf_name,
    n_folds,
    seed,
    project_root,
    idx_to_class,
):
    n_samples = len(y)
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    fold_ids = np.full(n_samples, -1, dtype=int)
    predictions = np.full(n_samples, -1, dtype=int)
    prob_class_0 = np.full(n_samples, np.nan, dtype=float)
    prob_class_1 = np.full(n_samples, np.nan, dtype=float)
    decision_scores = np.full(n_samples, np.nan, dtype=float)
    fold_metrics = []

    print(f"\n=== {clf_name} ===")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 必须只在当前 fold 的训练集上拟合，避免数据泄露。
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        clf = build_classifier(clf_name)
        clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_val_scaled)
        y_prob_all = clf.predict_proba(X_val_scaled)

        class_0_col = int(np.flatnonzero(clf.classes_ == 0)[0])
        class_1_col = int(np.flatnonzero(clf.classes_ == 1)[0])
        y_prob_0 = y_prob_all[:, class_0_col]
        y_prob_1 = y_prob_all[:, class_1_col]

        raw_decision = clf.decision_function(X_val_scaled)
        if np.ndim(raw_decision) == 1:
            y_decision = np.asarray(raw_decision, dtype=float)
        else:
            y_decision = np.asarray(raw_decision[:, class_1_col], dtype=float)

        fold_ids[val_idx] = fold_idx
        predictions[val_idx] = y_pred
        prob_class_0[val_idx] = y_prob_0
        prob_class_1[val_idx] = y_prob_1
        decision_scores[val_idx] = y_decision

        metrics = {
            "fold": fold_idx,
            "n_val": len(val_idx),
            "acc": accuracy_score(y_val, y_pred),
            "precision": precision_score(
                y_val, y_pred, average="binary", zero_division=0
            ),
            "recall": recall_score(
                y_val, y_pred, average="binary", zero_division=0
            ),
            "f1": f1_score(y_val, y_pred, average="binary", zero_division=0),
            "auc": roc_auc_score(y_val, y_prob_1),
        }
        fold_metrics.append(metrics)

        print(
            f"Fold {fold_idx}/{n_folds}  "
            f"acc={metrics['acc']:.4f}  "
            f"prec={metrics['precision']:.4f}  "
            f"rec={metrics['recall']:.4f}  "
            f"f1={metrics['f1']:.4f}  "
            f"auc={metrics['auc']:.4f}"
        )

    if np.any(fold_ids < 1) or np.any(predictions < 0):
        raise RuntimeError("部分样本没有获得 OOF 预测，已停止导出。")

    rows = []

    for sample_idx in range(n_samples):
        true_idx = int(y[sample_idx])
        pred_idx = int(predictions[sample_idx])
        filename = str(filenames[sample_idx])
        resolved_path, source_exists = resolve_image_path(project_root, filename)

        rows.append(
            {
                "sample_index": sample_idx,
                "fold": int(fold_ids[sample_idx]),
                "filename": filename,
                "resolved_source_path": resolved_path,
                "source_exists": source_exists,
                "true_idx": true_idx,
                "true_class": idx_to_class.get(true_idx, f"class_{true_idx}"),
                "pred_idx": pred_idx,
                "pred_class": idx_to_class.get(pred_idx, f"class_{pred_idx}"),
                "prob_non_gusu": float(prob_class_0[sample_idx]),
                "prob_gusu": float(prob_class_1[sample_idx]),
                "prediction_confidence": float(
                    max(prob_class_0[sample_idx], prob_class_1[sample_idx])
                ),
                "distance_from_0_5": float(abs(prob_class_1[sample_idx] - 0.5)),
                "decision_score": float(decision_scores[sample_idx]),
                "is_correct": true_idx == pred_idx,
                "error_type": error_type(true_idx, pred_idx),
            }
        )

    cm = confusion_matrix(y, predictions, labels=[0, 1])
    return rows, fold_metrics, cm


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"没有可写入的数据：{path}")

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def copy_misclassified_images(rows, out_dir: Path):
    copied = 0
    missing = 0

    for row in rows:
        if row["is_correct"]:
            continue

        source = Path(row["resolved_source_path"])
        if not source.is_file():
            missing += 1
            continue

        category = (
            "false_positive_non_gusu_to_gusu"
            if row["true_idx"] == 0
            else "false_negative_gusu_to_non_gusu"
        )
        target_dir = out_dir / category
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{int(row['sample_index']):04d}_{source.name}"
        shutil.copy2(source, target)
        copied += 1

    return copied, missing


def print_summary(clf_name, rows, fold_metrics, cm):
    metric_names = ["acc", "precision", "recall", "f1", "auc"]
    print("\n交叉验证结果（mean ± std）")
    for name in metric_names:
        values = np.array([m[name] for m in fold_metrics], dtype=float)
        print(f"  {name:12s}: {values.mean():.4f} ± {values.std():.4f}")

    n_errors = sum(not row["is_correct"] for row in rows)
    n_false_positive = sum(
        row["error_type"] == "false_positive_non_gusu_to_gusu" for row in rows
    )
    n_false_negative = sum(
        row["error_type"] == "false_negative_gusu_to_non_gusu" for row in rows
    )

    print("\n累加混淆矩阵：")
    print(cm)
    print(f"OOF 样本数       : {len(rows)}")
    print(f"误判总数         : {n_errors}")
    print(f"非姑苏→姑苏     : {n_false_positive}")
    print(f"姑苏→非姑苏     : {n_false_negative}")

    if clf_name == "svm_rbf":
        actual_fold_acc = [round(m["acc"], 4) for m in fold_metrics]
        cm_ok = np.array_equal(cm, EXPECTED_RBF_CM)
        folds_ok = actual_fold_acc == EXPECTED_RBF_FOLD_ACC

        if cm_ok and folds_ok:
            print("原始 RBF-SVM 核验: 通过（混淆矩阵和五折准确率完全一致）")
        else:
            raise RuntimeError(
                "原始 RBF-SVM 核验失败：\n"
                f"实际混淆矩阵={cm.tolist()}，预期={EXPECTED_RBF_CM.tolist()}\n"
                f"实际五折准确率={actual_fold_acc}，"
                f"预期={EXPECTED_RBF_FOLD_ACC}"
            )


def main(args):
    project_root = Path(args.project_root).expanduser().resolve()
    features_path = project_root / args.features
    out_dir = project_root / args.out_dir

    if not features_path.is_file():
        raise FileNotFoundError(f"未找到特征文件：{features_path}")

    data = np.load(features_path, allow_pickle=True)
    required = {"X", "y", "filenames"}
    missing_fields = required.difference(data.files)
    if missing_fields:
        raise KeyError(f"features.npz 缺少字段：{sorted(missing_fields)}")

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    filenames = data["filenames"]
    idx_to_class, _ = load_class_names(features_path)

    if X.ndim != 2 or y.ndim != 1 or filenames.ndim != 1:
        raise ValueError(
            f"数据形状异常：X={X.shape}, y={y.shape}, filenames={filenames.shape}"
        )
    if not (len(X) == len(y) == len(filenames)):
        raise ValueError("X、y、filenames 的样本数不一致。")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError(f"当前脚本仅处理标签 0/1，实际标签为：{np.unique(y)}")

    print(f"项目目录     : {project_root}")
    print(f"特征文件     : {features_path}")
    print(f"特征形状     : {X.shape}")
    print(f"类别分布     : {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"K-folds      : {args.n_folds}")
    print(f"随机种子     : {args.seed}")

    output_specs = {
        "svm_rbf": (
            "dinov2_v4_rbf_oof_predictions.csv",
            "dinov2_v4_rbf_misclassified.csv",
        ),
        "logreg": (
            "dinov2_v4_logreg_oof_predictions.csv",
            "dinov2_v4_logreg_misclassified.csv",
        ),
    }

    for clf_name in ("svm_rbf", "logreg"):
        rows, fold_metrics, cm = run_oof(
            X=X,
            y=y,
            filenames=filenames,
            clf_name=clf_name,
            n_folds=args.n_folds,
            seed=args.seed,
            project_root=project_root,
            idx_to_class=idx_to_class,
        )

        print_summary(clf_name, rows, fold_metrics, cm)

        all_name, errors_name = output_specs[clf_name]
        all_path = out_dir / all_name
        errors_path = out_dir / errors_name
        error_rows = [row for row in rows if not row["is_correct"]]

        write_csv(all_path, rows)
        write_csv(errors_path, error_rows)

        print(f"全部 OOF 预测   → {all_path}")
        print(f"误判清单        → {errors_path}")

        if args.copy_images:
            image_out_dir = out_dir / f"{Path(errors_name).stem}_images"
            copied, missing = copy_misclassified_images(rows, image_out_dir)
            print(f"误判图片副本    → {image_out_dir}")
            print(f"成功复制/路径缺失: {copied}/{missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="导出 DINOv2 + RBF-SVM/LogReg 的逐图片 OOF 预测"
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=".",
        help="gusu_cls 项目根目录（默认当前目录）",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="features/dinov2_v4/features.npz",
        help="相对于项目根目录的特征文件路径",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="reports",
        help="相对于项目根目录的输出目录（默认 reports）",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="交叉验证折数（原始实验为 5）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="StratifiedKFold 随机种子（原始实验为 42）",
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="同时把误判原图复制到 reports 下的分类目录",
    )
    main(parser.parse_args())
