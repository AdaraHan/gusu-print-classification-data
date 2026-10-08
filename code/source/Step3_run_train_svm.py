# train_svm.py
"""
加载特征向量，使用 StratifiedKFold(5) 训练并评估分类器。
每个 fold 内独立 fit StandardScaler，防止数据泄露。
输出 mean±std 格式的指标，以及累加混淆矩阵。

用法示例：
    python train_svm.py \\
        --features_dir features/dinov2 \\
        --clf svm_rbf \\
        --out_dir runs/svm_dinov2_rbf

    # --clf 可选: svm_rbf | svm_linear | logreg
"""
import argparse
import json
from datetime import datetime
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


# ──────────────────────────────────────────────────────────
# 分类器工厂
# ──────────────────────────────────────────────────────────

def build_classifier(clf_name: str):
    if clf_name == "svm_rbf":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                   class_weight="balanced", random_state=42)
    elif clf_name == "svm_linear":
        return SVC(kernel="linear", C=1.0, probability=True,
                   class_weight="balanced", random_state=42)
    elif clf_name == "logreg":
        return LogisticRegression(C=1.0, max_iter=1000,
                                  class_weight="balanced", random_state=42)
    else:
        raise ValueError(f"未知分类器：{clf_name}，可选 svm_rbf | svm_linear | logreg")


# ──────────────────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────────────────

def main(args):
    features_dir = Path(args.features_dir)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载特征 ──
    npz_path = features_dir / "features.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"未找到特征文件：{npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)          # (N, D)
    y = data["y"].astype(np.int64)             # (N,)

    meta_path = features_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    n_samples, n_dim = X.shape
    class_to_idx = meta.get("class_to_idx", {})

    print(f"特征维度    : {n_dim}")
    print(f"样本总数    : {n_samples}")
    print(f"类别分布    : {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"分类器      : {args.clf}")
    print(f"K-folds     : {args.n_folds}")
    print(f"随机种子    : {args.seed}")
    print()

    # ── K-Fold 交叉验证 ──
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    fold_metrics = {
        "acc"      : [],
        "precision": [],
        "recall"   : [],
        "f1"       : [],
        "auc"      : [],
    }

    # 累加混淆矩阵（仅适用于二分类）
    n_classes = len(np.unique(y))
    cm_total  = np.zeros((n_classes, n_classes), dtype=int)

    all_true  = []
    all_pred  = []
    all_prob  = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # fit scaler on train only（防止数据泄露）
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        # 训练分类器
        clf = build_classifier(args.clf)
        clf.fit(X_train, y_train)

        # 预测
        y_pred = clf.predict(X_val)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_val)[:, 1]
        else:
            y_prob = clf.decision_function(X_val)

        # 指标
        acc  = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average="binary", zero_division=0)
        rec  = recall_score(y_val, y_pred, average="binary", zero_division=0)
        f1   = f1_score(y_val, y_pred, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_val, y_prob)
        except Exception:
            auc = float("nan")

        cm = confusion_matrix(y_val, y_pred, labels=list(range(n_classes)))
        cm_total += cm

        fold_metrics["acc"].append(acc)
        fold_metrics["precision"].append(prec)
        fold_metrics["recall"].append(rec)
        fold_metrics["f1"].append(f1)
        fold_metrics["auc"].append(auc)

        all_true.extend(y_val.tolist())
        all_pred.extend(y_pred.tolist())
        all_prob.extend(y_prob.tolist())

        print(
            f"Fold {fold_idx}/{args.n_folds}  "
            f"acc={acc:.4f}  prec={prec:.4f}  "
            f"rec={rec:.4f}  f1={f1:.4f}  auc={auc:.4f}"
        )

    # ── 汇总统计 ──
    def stat(vals):
        arr = np.array([v for v in vals if not np.isnan(v)])
        if len(arr) == 0:
            return {"mean": float("nan"), "std": float("nan"), "per_fold": vals}
        return {
            "mean"    : round(float(np.mean(arr)), 4),
            "std"     : round(float(np.std(arr)),  4),
            "per_fold": [round(float(v), 4) for v in vals],
        }

    cv_results = {
        "model"     : meta.get("model", "unknown"),
        "variant"   : meta.get("variant", ""),
        "clf"       : args.clf,
        "n_folds"   : args.n_folds,
        "seed"      : args.seed,
        "n_samples" : int(n_samples),
        "n_dim"     : int(n_dim),
        "timestamp" : datetime.now().isoformat(timespec="seconds"),
        "metrics"   : {k: stat(v) for k, v in fold_metrics.items()},
        "confusion_matrix_total": cm_total.tolist(),
    }

    # ── 打印摘要 ──
    print()
    print("═" * 60)
    print("交叉验证结果（mean ± std）")
    print("═" * 60)
    for metric, s in cv_results["metrics"].items():
        print(f"  {metric:12s}: {s['mean']:.4f} ± {s['std']:.4f}")
    print()
    print("累加混淆矩阵：")
    print(np.array(cm_total))
    print()

    # ── 保存 cv_results.json ──
    cv_path = out_dir / "cv_results.json"
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump(cv_results, f, ensure_ascii=False, indent=2)
    print(f"详细结果 → {cv_path.resolve()}")

    # ── 保存 summary.txt（人类可读）──
    lines = []
    lines.append(f"模型       : {cv_results['model']} {cv_results['variant']}")
    lines.append(f"分类器     : {args.clf}")
    lines.append(f"K-folds    : {args.n_folds}")
    lines.append(f"样本总数   : {n_samples}")
    lines.append(f"特征维度   : {n_dim}")
    lines.append(f"时间戳     : {cv_results['timestamp']}")
    lines.append("")
    lines.append(f"{'指标':<12}  {'mean':>8}  {'std':>8}  {'per-fold'}")
    lines.append("─" * 60)
    for metric, s in cv_results["metrics"].items():
        per = "  ".join(f"{v:.4f}" for v in s["per_fold"])
        lines.append(f"{metric:<12}  {s['mean']:>8.4f}  {s['std']:>8.4f}  {per}")
    lines.append("")
    lines.append("累加混淆矩阵：")
    for row in cm_total:
        lines.append("  " + "  ".join(f"{v:6d}" for v in row))

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"摘要文件   → {summary_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM / LogReg 5-fold 交叉验证训练")
    parser.add_argument(
        "--features_dir", type=str, required=True,
        help="特征目录（包含 features.npz + meta.json）"
    )
    parser.add_argument(
        "--clf", type=str, default="svm_rbf",
        choices=["svm_rbf", "svm_linear", "logreg"],
        help="分类器类型（默认 svm_rbf）"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="输出目录（保存 cv_results.json + summary.txt）"
    )
    parser.add_argument("--n_folds", type=int, default=5,  help="K-fold 折数（默认 5）")
    parser.add_argument("--seed",    type=int, default=42, help="随机种子（默认 42）")
    args = parser.parse_args()
    main(args)
