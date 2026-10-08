"""Optional DINOv2 classifier rerun using the published, fixed validation folds.

This creates new results in a new directory. It never replaces the published OOF
predictions, and a different sklearn environment may produce different scores.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]


def read_manifest():
    with (ROOT / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def model(name):
    if name == "dinov2_rbf_svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                   class_weight="balanced", random_state=42)
    if name == "dinov2_logreg":
        return LogisticRegression(C=1.0, max_iter=1000,
                                  class_weight="balanced", random_state=42)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["dinov2_rbf_svm", "dinov2_logreg"], required=True)
    parser.add_argument("--output", type=Path, required=True, help="new CSV path; must not exist")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.resolve().is_relative_to((ROOT / "predictions").resolve()):
        raise ValueError("choose an output outside the published predictions directory")
    manifest = read_manifest()
    with np.load(ROOT / "features" / "dinov2_v4_public.npz", allow_pickle=False) as features:
        X = features["X"].copy()
        y = features["y"].copy()
        ids = features["sample_id"].tolist()
        paths = features["image_path"].tolist()
    if X.shape != (892, 768) or len(set(ids)) != 892 or set(ids) != set(manifest):
        raise ValueError("feature/manifest mismatch")
    folds = []
    for i, sample_id in enumerate(ids):
        item = manifest[sample_id]
        if item["image_path"] != paths[i] or int(y[i]) != (1 if item["class"] == "gusu" else 0):
            raise ValueError(f"feature identity mismatch {sample_id}")
        folds.append(int(item["fold"]))
    folds = np.asarray(folds)
    output = []
    for fold in range(1, 6):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train])
        X_valid = scaler.transform(X[valid])
        classifier = model(args.model)
        classifier.fit(X_train, y[train])
        positive_col = list(classifier.classes_).index(1)
        probabilities = classifier.predict_proba(X_valid)[:, positive_col]
        predictions = classifier.predict(X_valid)
        for index, prob, predicted in zip(valid, probabilities, predictions):
            output.append({"sample_id": ids[index], "fold": fold,
                           "true_class": manifest[ids[index]]["class"],
                           "pred_class": "gusu" if predicted == 1 else "non_gusu",
                           "prob_gusu": format(float(prob), ".17g")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "fold", "true_class", "pred_class", "prob_gusu"])
        writer.writeheader()
        writer.writerows(sorted(output, key=lambda row: row["sample_id"]))
    print(json.dumps({"model": args.model, "rows": len(output), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
