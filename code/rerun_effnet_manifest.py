"""Optional EfficientNet rerun adapter using the published sample order and folds.

The historical training script is retained under code/source/. This adapter changes
only its data enumeration and fold construction to use manifest.csv after filenames
were anonymized. Run --preflight first. Training writes to a new output directory.
"""

import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "code" / "source" / "Step4_run_train_effnet_cv_oof.py"


def main():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("historical_effnet", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (ROOT / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["sample_id"].split("-")[1]))
    if len(rows) != 892 or len({row["sample_id"] for row in rows}) != 892:
        raise ValueError("manifest size or IDs changed")
    samples = []
    for row in rows:
        image = ROOT / row["image_path"]
        if not image.is_file():
            raise FileNotFoundError(image)
        samples.append((image, module.CLASS_TO_IDX[row["class"]]))
    folds = np.asarray([int(row["fold"]) for row in rows])
    if sorted(set(folds)) != [1, 2, 3, 4, 5]:
        raise ValueError("original validation folds missing")

    def collect_samples(_data_dir):
        return samples, dict(module.CLASS_TO_IDX)

    def build_splits(labels_arr, n_folds, seed):
        if len(labels_arr) != 892 or n_folds != 5 or seed != 42:
            raise ValueError("expected original five-fold settings")
        return [(np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)) for fold in range(1, 6)]

    module.collect_samples = collect_samples
    module.build_splits = build_splits
    os.chdir(ROOT)
    args = module.build_parser().parse_args()
    if args.data_dir == "data_v4_merged":
        args.data_dir = "images"
    if args.out_dir == "runs/effnetv2_v4_cv_oof":
        args.out_dir = "reruns/effnetv2_v4_cv_oof"
    if args.reports_dir == "reports":
        args.reports_dir = "reruns/reports"
    module.main(args)


if __name__ == "__main__":
    main()
