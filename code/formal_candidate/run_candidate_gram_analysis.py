#!/usr/bin/env python3
"""Reproducible candidate-sample Gram style-representation experiment.

The experiment intentionally does not reproduce or target manuscript values
0.0248/0.0157.  It uses a fixed ImageNet-pretrained CNN and reports cosine
distances between normalized Gram representations; smaller means more similar
under this representation.  Results remain candidate/exploratory until the
formal manifest and independent-work/edition units are manually frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gusu_gram_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gusu_gram_cache")

import cv2
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.models import squeezenet1_1


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_ROOT / "reports/formal_aesthetics_reproduction_20260727/formal_analysis_manifest.csv"
DUPLICATE_PAIR_PATH = PROJECT_ROOT / "reports/formal_aesthetics_reproduction_stage15_revision_20260727/duplicate_pair_priority.csv"
WEIGHTS_PATH = PROJECT_ROOT / "models/pretrained/squeezenet1_1-imagenet1k-v1-b8a52dc0.pth"

GROUP_MAP = {
    "清代姑苏版画（候选，待确认）": ("gusu_qing_candidate", "姑苏版画（候选）", 1),
    "20世纪50年代桃花坞年画（候选，待确认）": ("taohuawu_1950s_candidate", "桃花坞年画（候选）", 2),
    "清末杨柳青年画（候选，待确认）": ("yangliuqing_late_qing_candidate", "杨柳青年画（候选）", 3),
}
EXPECTED_COUNTS = {
    "gusu_qing_candidate": 459,
    "taohuawu_1950s_candidate": 9,
    "yangliuqing_late_qing_candidate": 100,
}
GROUP_ORDER = list(EXPECTED_COUNTS)
GROUP_SHORT = {
    "gusu_qing_candidate": "姑苏",
    "taohuawu_1950s_candidate": "桃花坞",
    "yangliuqing_late_qing_candidate": "杨柳青",
}

LONG_EDGE = 384
LAYER_SPECS = [(1, "features_1_relu_conv1", 64), (4, "features_4_fire3", 128)]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
RANDOM_SEED = 20260729

OUTPUT_FILES = [
    "analysis_manifest_snapshot.csv",
    "gram_embeddings.npz",
    "gram_pairwise_distances.csv",
    "gram_group_pair_summary.csv",
    "gram_layer_group_pair_summary.csv",
    "gram_per_image_cohesion.csv",
    "gram_group_centroid_distances.csv",
    "figure_gram_pair_distributions.png",
    "figure_gram_group_centroid_distances.png",
    "gram_parameters.json",
    "GRAM_VALIDATION_REPORT.md",
    "PAPER_SECTION_GRAM_CANDIDATE.md",
    "gram_candidate_analysis.ipynb",
    "gram_run_receipt.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image_rgb(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot_read_image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_long_edge(image: np.ndarray, long_edge: int = LONG_EDGE) -> np.ndarray:
    height, width = image.shape[:2]
    scale = long_edge / max(height, width)
    new_width = max(32, int(round(width * scale)))
    new_height = max(32, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def load_manifest() -> pd.DataFrame:
    frame = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig", dtype=str).fillna("")
    required = {"sample_id", "image_path", "sha256", "proposed_group", "source_prefix"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manifest_missing_columns: {sorted(missing)}")
    if len(frame) != 568 or frame["sample_id"].nunique() != 568:
        raise ValueError(f"manifest_expected_568_unique: rows={len(frame)} unique={frame['sample_id'].nunique()}")
    unknown = sorted(set(frame["proposed_group"]) - set(GROUP_MAP))
    if unknown:
        raise ValueError(f"unexpected_groups: {unknown}")
    mapped = frame["proposed_group"].map(GROUP_MAP)
    frame["group_id"] = mapped.map(lambda value: value[0])
    frame["group_label"] = mapped.map(lambda value: value[1])
    frame["group_order"] = mapped.map(lambda value: value[2])
    if frame.groupby("group_id").size().to_dict() != EXPECTED_COUNTS:
        raise ValueError("candidate_group_counts_mismatch")
    frame["resolved_image_path"] = frame["image_path"].map(
        lambda value: str((PROJECT_ROOT / value).resolve()) if not Path(value).is_absolute() else value
    )
    return frame.sort_values(["group_order", "sample_id"]).reset_index(drop=True)


def load_model() -> torch.nn.Module:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"missing_fixed_weights: {WEIGHTS_PATH}")
    model = squeezenet1_1(weights=None)
    try:
        state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    features = model.features.eval()
    for parameter in features.parameters():
        parameter.requires_grad_(False)
    return features


def gram_upper_unit(feature: torch.Tensor) -> np.ndarray:
    channels = int(feature.shape[1])
    flattened = feature[0].reshape(channels, -1)
    gram = (flattened @ flattened.T) / float(channels * flattened.shape[1])
    upper = torch.triu_indices(channels, channels, device=gram.device)
    vector = gram[upper[0], upper[1]].clone()
    off_diagonal = upper[0] != upper[1]
    vector[off_diagonal] *= math.sqrt(2.0)
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise ValueError("invalid_gram_norm")
    vector = vector / norm
    return vector.detach().cpu().numpy().astype(np.float32, copy=False)


def extract_embeddings(model: torch.nn.Module, manifest: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    collected: dict[str, list[np.ndarray]] = {name: [] for _, name, _ in LAYER_SPECS}
    metadata_rows: list[dict[str, Any]] = []
    capture_indices = {index: name for index, name, _ in LAYER_SPECS}
    maximum_layer = max(capture_indices)

    with torch.inference_mode():
        for index, row in manifest.iterrows():
            image_path = Path(row["resolved_image_path"])
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            observed_sha = sha256_file(image_path)
            if observed_sha.lower() != row["sha256"].lower():
                raise ValueError(f"sha256_mismatch: {row['sample_id']}")
            original = read_image_rgb(image_path)
            resized = resize_long_edge(original)
            array = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
            array = (array - IMAGENET_MEAN) / IMAGENET_STD
            tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
            activation = tensor
            captured: dict[str, np.ndarray] = {}
            for module_index, module in enumerate(model):
                activation = module(activation)
                if module_index in capture_indices:
                    name = capture_indices[module_index]
                    captured[name] = gram_upper_unit(activation)
                if module_index >= maximum_layer:
                    break
            if set(captured) != set(collected):
                raise ValueError(f"missing_layer_capture: {row['sample_id']} {sorted(captured)}")
            for name, vector in captured.items():
                collected[name].append(vector)
            metadata_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "image_path": row["image_path"],
                    "sha256": row["sha256"],
                    "proposed_group": row["proposed_group"],
                    "group_id": row["group_id"],
                    "group_label": row["group_label"],
                    "group_order": int(row["group_order"]),
                    "source_prefix": row["source_prefix"],
                    "series_id_candidate": row.get("series_id", ""),
                    "work_id_candidate": row.get("work_id", ""),
                    "original_width": int(original.shape[1]),
                    "original_height": int(original.shape[0]),
                    "analysis_width": int(resized.shape[1]),
                    "analysis_height": int(resized.shape[0]),
                    "analysis_scope": "candidate_exploratory_not_formal_manifest",
                }
            )
            if (index + 1) % 100 == 0 or index + 1 == len(manifest):
                print(f"Gram extracted {index + 1}/{len(manifest)}", flush=True)
    stacked = {name: np.stack(vectors).astype(np.float32) for name, vectors in collected.items()}
    return stacked, pd.DataFrame(metadata_rows)


def candidate_duplicate_pairs() -> set[frozenset[str]]:
    if not DUPLICATE_PAIR_PATH.exists():
        return set()
    pairs = pd.read_csv(DUPLICATE_PAIR_PATH, encoding="utf-8-sig", dtype=str).fillna("")
    required = {"sample_id_a", "sample_id_b"}
    if not required.issubset(pairs.columns):
        raise ValueError("duplicate_pair_file_missing_columns")
    return {
        frozenset((row.sample_id_a, row.sample_id_b))
        for row in pairs.itertuples(index=False)
        if row.sample_id_a and row.sample_id_b
    }


def calculate_pairwise(
    embeddings: dict[str, np.ndarray], metadata: pd.DataFrame, duplicate_pairs: set[frozenset[str]]
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    layer_distances: dict[str, np.ndarray] = {}
    for name, matrix in embeddings.items():
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError(f"embedding_not_unit_norm: {name}")
        similarity = np.clip(matrix @ matrix.T, -1.0, 1.0)
        layer_distances[name] = np.clip(1.0 - similarity, 0.0, 2.0).astype(np.float32)
    combined = np.mean(np.stack(list(layer_distances.values()), axis=0), axis=0).astype(np.float32)
    layer_distances["mean_equal_layer_cosine_distance"] = combined

    first, second = np.triu_indices(len(metadata), k=1)
    ids = metadata["sample_id"].to_numpy()
    groups = metadata["group_id"].to_numpy()
    labels = metadata["group_label"].to_numpy()
    orders = metadata["group_order"].to_numpy(dtype=int)
    rows: dict[str, Any] = {
        "sample_id_a": ids[first],
        "sample_id_b": ids[second],
        "group_id_a": groups[first],
        "group_id_b": groups[second],
        "group_label_a": labels[first],
        "group_label_b": labels[second],
        "within_candidate_group": groups[first] == groups[second],
    }
    pair_labels = []
    for ia, ib in zip(first, second):
        if orders[ia] <= orders[ib]:
            pair_labels.append(f"{GROUP_SHORT[groups[ia]]}—{GROUP_SHORT[groups[ib]]}")
        else:
            pair_labels.append(f"{GROUP_SHORT[groups[ib]]}—{GROUP_SHORT[groups[ia]]}")
    rows["candidate_group_pair"] = pair_labels
    rows["ahash_candidate_pair_pending_review"] = [
        frozenset((ids[ia], ids[ib])) in duplicate_pairs for ia, ib in zip(first, second)
    ]
    for name, distance_matrix in layer_distances.items():
        rows[name] = distance_matrix[first, second]
    return pd.DataFrame(rows), layer_distances


def summarize_pairs(pairwise: pd.DataFrame, metric: str, by_layer: bool = False) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sensitivity, subset in [
        ("all_candidate_pairs", pairwise),
        ("exclude_ahash_candidate_pairs_pending_review", pairwise[~pairwise["ahash_candidate_pair_pending_review"]]),
    ]:
        for group_pair, values_frame in subset.groupby("candidate_group_pair", sort=False):
            values = values_frame[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "sensitivity_scope": sensitivity,
                    "metric": metric,
                    "candidate_group_pair": group_pair,
                    "within_candidate_group": bool(values_frame["within_candidate_group"].iloc[0]),
                    "n_pairs": len(values),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                    "median": float(np.median(values)),
                    "q1": float(np.percentile(values, 25)),
                    "q3": float(np.percentile(values, 75)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "distance_direction": "smaller_is_more_similar",
                    "analysis_unit_warning": "pairs_are_not_independent",
                }
            )
    return pd.DataFrame(rows)


def per_image_cohesion(metadata: pd.DataFrame, distance: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = metadata["group_id"].to_numpy()
    ids = metadata["sample_id"].to_numpy()
    for index, row in metadata.iterrows():
        record: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "image_path": row["image_path"],
            "group_id": row["group_id"],
            "group_label": row["group_label"],
        }
        for group_id in GROUP_ORDER:
            mask = groups == group_id
            if group_id == row["group_id"]:
                mask[index] = False
            values = distance[index, mask]
            record[f"mean_distance_to_{group_id}"] = float(np.mean(values)) if len(values) else float("nan")
        same_mask = groups == row["group_id"]
        same_mask[index] = False
        same_indices = np.where(same_mask)[0]
        if len(same_indices):
            nearest_local = same_indices[np.argmin(distance[index, same_indices])]
            record["nearest_same_group_sample_id"] = ids[nearest_local]
            record["nearest_same_group_distance"] = float(distance[index, nearest_local])
        else:
            record["nearest_same_group_sample_id"] = ""
            record["nearest_same_group_distance"] = float("nan")
        all_values = distance[index].copy()
        all_values[index] = np.inf
        nearest = int(np.argmin(all_values))
        record["nearest_overall_sample_id"] = ids[nearest]
        record["nearest_overall_group_id"] = groups[nearest]
        record["nearest_overall_distance"] = float(all_values[nearest])
        rows.append(record)
    return pd.DataFrame(rows)


def centroid_distances(embeddings: dict[str, np.ndarray], metadata: pd.DataFrame) -> pd.DataFrame:
    layer_matrices: list[np.ndarray] = []
    for name, matrix in embeddings.items():
        centroids = []
        for group_id in GROUP_ORDER:
            centroid = matrix[metadata["group_id"].to_numpy() == group_id].mean(axis=0)
            centroid /= np.linalg.norm(centroid)
            centroids.append(centroid)
        centroids_array = np.stack(centroids)
        layer_matrices.append(np.clip(1.0 - centroids_array @ centroids_array.T, 0.0, 2.0))
    combined = np.mean(np.stack(layer_matrices), axis=0)
    rows = []
    for first, group_a in enumerate(GROUP_ORDER):
        for second, group_b in enumerate(GROUP_ORDER):
            rows.append(
                {
                    "group_a": group_a,
                    "group_a_label": GROUP_SHORT[group_a],
                    "group_b": group_b,
                    "group_b_label": GROUP_SHORT[group_b],
                    "mean_equal_layer_cosine_distance": float(combined[first, second]),
                    "distance_direction": "smaller_is_more_similar",
                }
            )
    return pd.DataFrame(rows)


def make_figures(pairwise: pd.DataFrame, centroid: pd.DataFrame) -> None:
    metric = "mean_equal_layer_cosine_distance"
    labels = ["姑苏—姑苏", "桃花坞—桃花坞", "杨柳青—杨柳青", "姑苏—桃花坞", "姑苏—杨柳青", "桃花坞—杨柳青"]
    rng = np.random.default_rng(RANDOM_SEED)
    data = []
    shown_labels = []
    for label in labels:
        values = pairwise.loc[pairwise["candidate_group_pair"] == label, metric].to_numpy(dtype=float)
        if len(values) > 5000:
            values = rng.choice(values, 5000, replace=False)
        if len(values):
            data.append(values)
            shown_labels.append(label.replace("桃花坞", "THW").replace("杨柳青", "YLQ").replace("姑苏", "Gusu"))
    fig, axis = plt.subplots(figsize=(10, 5.8), dpi=180)
    axis.boxplot(data, tick_labels=shown_labels, showfliers=False, patch_artist=True,
                 boxprops={"facecolor": "#9CC5A1", "alpha": 0.8}, medianprops={"color": "#9B2226", "linewidth": 1.5})
    axis.set_ylabel("Mean Gram cosine distance (smaller = more similar)")
    axis.set_title("Candidate-image Gram distance distributions")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_gram_pair_distributions.png", bbox_inches="tight")
    plt.close(fig)

    matrix = centroid.pivot(index="group_a_label", columns="group_b_label", values=metric).loc[
        [GROUP_SHORT[g] for g in GROUP_ORDER], [GROUP_SHORT[g] for g in GROUP_ORDER]
    ].to_numpy()
    fig, axis = plt.subplots(figsize=(6.4, 5.4), dpi=180)
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=max(0.01, float(matrix.max())))
    tick_labels = ["Gusu", "THW", "YLQ"]
    axis.set_xticks(range(3), tick_labels)
    axis.set_yticks(range(3), tick_labels)
    axis.set_title("Gram distance between candidate-group centroids")
    for row in range(3):
        for column in range(3):
            axis.text(column, row, f"{matrix[row, column]:.4f}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Cosine distance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_gram_group_centroid_distances.png", bbox_inches="tight")
    plt.close(fig)


def build_notebook() -> None:
    def markdown(source: str) -> dict[str, Any]:
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

    def code(source: str) -> dict[str, Any]:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}

    notebook = {
        "cells": [
            markdown("# Candidate-sample Gram style-representation experiment\n\n## tl;dr\nThis notebook verifies saved results from a fixed SqueezeNet 1.1 Gram experiment. It does not reproduce 0.0248/0.0157 and does not infer perspective or cultural causality."),
            markdown("## Context & Methods\n\nTwo fixed convolutional feature maps (`features.1`, `features.4`) are converted to spatially normalized Gram matrices. The Frobenius-preserving upper triangle is L2-normalized, and cosine distance is computed per layer then averaged. Smaller distance means more similar under this representation.\n\n### Key Assumptions\nCandidate folder groups are not a frozen formal manifest; pairwise distances are dependent; aHash pairs are review candidates rather than confirmed duplicates."),
            code("from pathlib import Path\nimport pandas as pd\nimport numpy as np\nfrom IPython.display import display, Image\nHERE = Path.cwd()\nif not (HERE / 'gram_group_pair_summary.csv').exists():\n    HERE = Path('reports/formal_gram_candidate_experiment_20260729')\nsummary = pd.read_csv(HERE / 'gram_group_pair_summary.csv', encoding='utf-8-sig')\ncentroid = pd.read_csv(HERE / 'gram_group_centroid_distances.csv', encoding='utf-8-sig')\nper_image = pd.read_csv(HERE / 'gram_per_image_cohesion.csv', encoding='utf-8-sig')\npairwise = pd.read_csv(HERE / 'gram_pairwise_distances.csv', encoding='utf-8-sig')\nassert len(per_image) == 568 and per_image.sample_id.nunique() == 568\nassert pairwise['mean_equal_layer_cosine_distance'].between(0, 2).all()\nprint({'images': len(per_image), 'pairs': len(pairwise), 'missing_primary_distance': int(pairwise['mean_equal_layer_cosine_distance'].isna().sum())})"),
            markdown("## Data"),
            code("display(summary[summary.sensitivity_scope.eq('all_candidate_pairs')].round(6))"),
            markdown("## Results"),
            code("display(centroid.pivot(index='group_a_label', columns='group_b_label', values='mean_equal_layer_cosine_distance').round(6))\ndisplay(Image(filename=str(HERE / 'figure_gram_pair_distributions.png')))"),
            markdown("## Takeaways\n\nInterpret only the observed representation distances. Do not convert them into claims about vanishing points, specific brightness mechanisms, craft standards, aesthetic superiority, or historical causality. Formal inference requires the manually frozen work/edition-level manifest and duplicate handling."),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": platform.python_version()}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (OUTPUT_DIR / "gram_candidate_analysis.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite outputs from this script")
    args = parser.parse_args()
    existing = [str(OUTPUT_DIR / name) for name in OUTPUT_FILES if (OUTPUT_DIR / name).exists()]
    if existing and not args.force:
        raise FileExistsError(f"refuse_to_overwrite_existing_outputs: {existing}")

    manifest = load_manifest()
    model = load_model()
    embeddings, metadata = extract_embeddings(model, manifest)
    duplicate_pairs = candidate_duplicate_pairs()
    pairwise, distance_matrices = calculate_pairwise(embeddings, metadata, duplicate_pairs)
    primary_metric = "mean_equal_layer_cosine_distance"
    group_summary = summarize_pairs(pairwise, primary_metric)
    layer_summary = pd.concat(
        [summarize_pairs(pairwise, name) for _, name, _ in LAYER_SPECS], ignore_index=True
    )
    cohesion = per_image_cohesion(metadata, distance_matrices[primary_metric])
    centroid = centroid_distances(embeddings, metadata)

    manifest.to_csv(OUTPUT_DIR / "analysis_manifest_snapshot.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        OUTPUT_DIR / "gram_embeddings.npz",
        sample_id=metadata["sample_id"].astype(str).to_numpy(dtype="U"),
        **embeddings,
    )
    pairwise.to_csv(OUTPUT_DIR / "gram_pairwise_distances.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(OUTPUT_DIR / "gram_group_pair_summary.csv", index=False, encoding="utf-8-sig")
    layer_summary.to_csv(OUTPUT_DIR / "gram_layer_group_pair_summary.csv", index=False, encoding="utf-8-sig")
    cohesion.to_csv(OUTPUT_DIR / "gram_per_image_cohesion.csv", index=False, encoding="utf-8-sig")
    centroid.to_csv(OUTPUT_DIR / "gram_group_centroid_distances.csv", index=False, encoding="utf-8-sig")
    make_figures(pairwise, centroid)

    parameters = {
        "analysis_status": "candidate_exploratory_not_formal_manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_counts": EXPECTED_COUNTS,
        "analysis_unit": "candidate_image; pairwise distances are dependent",
        "model": "torchvision squeezenet1_1",
        "weights": "ImageNet1K V1",
        "weights_path": str(WEIGHTS_PATH.relative_to(PROJECT_ROOT)),
        "weights_sha256": sha256_file(WEIGHTS_PATH),
        "weights_source_url": "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
        "layers": [{"index": index, "name": name, "channels": channels} for index, name, channels in LAYER_SPECS],
        "preprocessing": {
            "color": "RGB",
            "resize": f"preserve aspect ratio, long edge {LONG_EDGE}px, no crop, no padding",
            "scale": "[0,1]",
            "normalization_mean": IMAGENET_MEAN[:, 0, 0].tolist(),
            "normalization_sd": IMAGENET_STD[:, 0, 0].tolist(),
            "content_mask": "none; full resized image",
        },
        "gram": {
            "feature_shape": "C x H x W; flatten to F=C x HW",
            "matrix": "G = F F^T / (C*H*W)",
            "storage": "upper triangle with off-diagonal multiplied by sqrt(2), preserving Frobenius geometry",
            "vector_normalization": "L2 unit norm per layer",
        },
        "distance": {
            "per_layer": "cosine distance = 1 - dot(unit_gram_i, unit_gram_j)",
            "primary": "equal arithmetic mean across the two fixed layers",
            "direction": "smaller_is_more_similar",
            "range": [0, 2],
        },
        "duplicate_sensitivity": {
            "source": str(DUPLICATE_PAIR_PATH.relative_to(PROJECT_ROOT)),
            "candidate_pairs": len(duplicate_pairs),
            "interpretation": "aHash pairs are excluded only in sensitivity summary; they are not declared duplicates",
        },
        "forbidden_inferences": [
            "vanishing point or perspective geometry",
            "specific brightness or edge mechanism",
            "craft standard or aesthetic superiority",
            "historical causality",
            "formal population inference before manifest freeze",
        ],
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (OUTPUT_DIR / "gram_parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    primary = group_summary[group_summary["sensitivity_scope"] == "all_candidate_pairs"]
    within = primary[primary["within_candidate_group"]].copy()
    within_lines = "\n".join(
        f"| {row.candidate_group_pair} | {int(row.n_pairs)} | {row.mean:.6f} | {row.sd:.6f} | {row.median:.6f} |"
        for row in within.itertuples(index=False)
    )
    all_lines = "\n".join(
        f"| {row.candidate_group_pair} | {int(row.n_pairs)} | {row.mean:.6f} | {row.median:.6f} |"
        for row in primary.itertuples(index=False)
    )
    validation = f"""# Gram矩阵候选样本实验验证报告

## 总体判定

本次是新建的、可重跑的候选样本探索实验，不是对原稿0.0248或0.0157的恢复。固定使用ImageNet预训练SqueezeNet 1.1及两个预先指定卷积特征层，逐图计算空间归一化Gram矩阵，再用逐层余弦距离的等权平均进行比较。距离越小表示在该固定表征下越相似。

## 完整性

- 输入568个唯一sample_id，候选组为459、9、100。
- 每幅图重新核对路径和SHA-256。
- 保存568幅图的Gram向量、全部{len(pairwise):,}个无序图像对距离、逐图组内/组间距离、组质心距离和参数文件。
- 主距离无缺失，范围为{pairwise[primary_metric].min():.6f}—{pairwise[primary_metric].max():.6f}。
- 权重SHA-256：`{sha256_file(WEIGHTS_PATH)}`。
- aHash的{len(duplicate_pairs)}对候选关系未被认定为重复，只另做排除候选对的敏感性汇总。

## 组内距离（候选图像对）

| 候选组对 | 图像对数 | 均值 | 标准差 | 中位数 |
|---|---:|---:|---:|---:|
{within_lines}

## 解释限制

图像对共享样本，故{len(pairwise):,}个距离不是相互独立的统计单位；当前只作描述性汇总，不把图像对数当作推断样本量。三组目录标签、年代、独立作品/版次、组画和近重复仍待人工冻结，尤其桃花坞只有9张候选图。Gram表示丢弃空间位置，只能作为固定CNN层的整体纹理/风格表征，不能检测消失点，也不能直接证明亮度机制、边缘机制、工艺规范、美学传统或历史因果。
"""
    (OUTPUT_DIR / "GRAM_VALIDATION_REPORT.md").write_text(validation, encoding="utf-8")

    paper = f"""## CNN Gram矩阵的候选样本风格表征实验

为探索三组候选图像在卷积特征相关结构上的差异，本研究固定采用ImageNet预训练SqueezeNet 1.1，提取`features.1`与`features.4`两个浅层卷积阶段的特征图。输入图像保持纵横比，将长边缩放至{LONG_EDGE}像素，不裁切、不补边，并采用ImageNet均值和标准差归一化。对第$l$层特征$F_l\\in\\mathbb{{R}}^{{C_l\\times H_lW_l}}$计算

$$G_l=\\frac{{F_lF_l^\\mathrm{{T}}}}{{C_lH_lW_l}}.$$

将Gram矩阵按Frobenius几何展开并作L2归一化后，使用余弦距离$d_l=1-\\langle g_l^i,g_l^j\\rangle$比较图像，主指标为两层距离的等权平均；数值越小表示在该固定CNN表征下越相似。该定义与原稿中没有公式来源的0.0248和0.0157无关。

候选样本组内描述结果如下：

| 候选组 | 图像对数 | Gram距离均值 | 标准差 | 中位数 |
|---|---:|---:|---:|---:|
{within_lines}

完整组内与组间结果为：

| 候选组对 | 图像对数 | 均值 | 中位数 |
|---|---:|---:|---:|
{all_lines}

上述结果以图像对描述固定CNN表征距离。由于样本尚未完成作品/版次、组画、年代及近重复人工审核，图像对亦不相互独立，故不能据此作正式总体推断或将某组差异归因于工艺规范和历史嬗变。Gram矩阵不保留几何空间位置，不能作为消失点或透视结构证据。
"""
    (OUTPUT_DIR / "PAPER_SECTION_GRAM_CANDIDATE.md").write_text(paper, encoding="utf-8")
    build_notebook()

    receipt = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "manifest_rows": len(manifest),
        "unique_sample_ids": int(manifest["sample_id"].nunique()),
        "pairwise_rows": len(pairwise),
        "primary_distance_missing": int(pairwise[primary_metric].isna().sum()),
        "primary_distance_min": float(pairwise[primary_metric].min()),
        "primary_distance_max": float(pairwise[primary_metric].max()),
        "weights_sha256": sha256_file(WEIGHTS_PATH),
        "status": "passed",
    }
    (OUTPUT_DIR / "gram_run_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
