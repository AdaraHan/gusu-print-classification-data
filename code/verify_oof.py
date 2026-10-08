"""Verify the published per-image OOF results without training a model."""

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def auc(rows):
    ranked = sorted((float(r["prob_gusu"]), r["true_class"] == "gusu") for r in rows)
    positive = sum(is_positive for _, is_positive in ranked)
    negative = len(ranked) - positive
    if not positive or not negative:
        raise ValueError("AUC requires both classes")
    positive_ranks = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2
        positive_ranks += sum(flag for _, flag in ranked[start:end]) * average_rank
        start = end
    return (positive_ranks - positive * (positive + 1) / 2) / (positive * negative)


def metrics(rows):
    cm = Counter((r["true_class"], r["pred_class"]) for r in rows)
    tp = cm[("gusu", "gusu")]
    fn = cm[("gusu", "non_gusu")]
    fp = cm[("non_gusu", "gusu")]
    tn = cm[("non_gusu", "non_gusu")]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return {"accuracy": (tp + tn) / len(rows), "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall), "auc": auc(rows), "confusion_matrix": [[tp, fn], [fp, tn]]}


def main():
    manifest = {r["sample_id"]: r for r in read_csv(ROOT / "manifest.csv")}
    if len(manifest) != 892:
        raise ValueError("Unexpected manifest size")
    results = {}
    for path in sorted((ROOT / "predictions").glob("*_oof_predictions.csv")):
        rows = read_csv(path)
        if len(rows) != len(manifest) or len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError(f"Missing or duplicated OOF samples: {path.name}")
        for row in rows:
            source = manifest[row["sample_id"]]
            if (row["image_path"], row["true_class"], row["fold"]) != (source["image_path"], source["class"], source["fold"]):
                raise ValueError(f"Broken image/label/fold reference: {path.name} {row['sample_id']}")
            if not (ROOT / row["image_path"]).is_file():
                raise ValueError(f"Missing image: {row['image_path']}")
        fold_values = [metrics([r for r in rows if r["fold"] == str(fold)]) for fold in range(1, 6)]
        results[path.name] = {
            "fold_mean": {k: statistics.mean(x[k] for x in fold_values) for k in ("accuracy", "precision", "recall", "f1", "auc")},
            "fold_population_std": {k: statistics.pstdev(x[k] for x in fold_values) for k in ("accuracy", "precision", "recall", "f1", "auc")},
            "pooled_oof": metrics(rows),
            "n": len(rows),
        }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
