"""Map pairwise identifiability of the 14 precomputed model histories."""
import itertools
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from .evaluate import _bootstrap_ci, _classifier, _load_manifest


matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASSES = ("root", "null", "sft", "unlearn", "prune", "quant",
           "sft-unlearn", "unlearn-sft", "sft-prune", "prune-sft",
           "sft-quant", "quant-sft", "distill", "merge")
FEATURES = ("all_base_free", "delta_ref")


def _archive_array(archive, name):
    for key in (name, f"{name}.cnn"):
        if key in archive:
            return archive[key]
    raise KeyError(name)


def _load_run(path, keep_root):
    manifest = _load_manifest(path)
    archive = np.load(Path(path) / "fingerprints.npz")
    if archive["model_ids"].tolist() != [entry["model_id"] for entry in manifest]:
        raise ValueError(f"fingerprints do not match manifest order: {path}")
    keep = np.asarray([keep_root or entry["label"] != "root" for entry in manifest])
    all_base_free = np.concatenate([
        _archive_array(archive, name) for name in ("actgeom", "logit", "weights", "blackbox")
    ], axis=1)
    return ([entry for entry, selected in zip(manifest, keep) if selected],
            {"all_base_free": all_base_free[keep],
             "delta_ref": _archive_array(archive, "delta_ref")[keep]})


def _components(classes, equivalent_edges):
    neighbors = {name: set() for name in classes}
    for left, right in equivalent_edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    remaining = set(classes)
    components = []
    while remaining:
        start = next(name for name in classes if name in remaining)
        stack, component = [start], []
        remaining.remove(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in neighbors[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component, key=classes.index))
    return components


def _plot_matrix(matrix, classes, feature_name, path):
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, vmin=0.5, vmax=1.0, cmap="viridis")
    axis.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    axis.set_yticks(range(len(classes)), classes)
    axis.set_title(f"Pairwise identifiability: {feature_name}")
    for row, column in itertools.product(range(len(classes)), repeat=2):
        if row != column:
            value = matrix[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                      fontsize=7, color="white" if value < 0.72 else "black")
    figure.colorbar(image, ax=axis, label="balanced accuracy")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def evaluate_identifiability(out_dir="runs/identifiability"):
    loaded = [
        _load_run("runs/pilot_matched", True),
        _load_run("runs/pairs", False),
        _load_run("runs/openset", False),
    ]
    manifest = sum((run_manifest for run_manifest, _ in loaded), [])
    feature_sets = {
        name: np.concatenate([arrays[name] for _, arrays in loaded]) for name in FEATURES
    }
    labels = np.asarray([entry["label"] for entry in manifest])
    groups = np.asarray([entry["root_group"] for entry in manifest])
    counts = Counter(labels)
    assert set(counts) == set(CLASSES) and all(counts[name] == 40 for name in CLASSES)
    shared_roots = set(groups[labels == CLASSES[0]])
    assert len(shared_roots) == 40
    assert all(set(groups[labels == name]) == shared_roots for name in CLASSES)

    pair_results = []
    for left, right in itertools.combinations(CLASSES, 2):
        selected = np.isin(labels, (left, right))
        pair_y = (labels[selected] == right).astype(int)
        pair_groups = groups[selected]
        splitter = GroupKFold(n_splits=8)
        folds = list(splitter.split(pair_y, pair_y, pair_groups))
        assert all(set(pair_groups[train]).isdisjoint(pair_groups[test])
                   for train, test in folds)
        for feature_name, all_x in feature_sets.items():
            x = all_x[selected]
            classifier_results = []
            for classifier_name in ("logreg", "rf"):
                prediction = np.empty_like(pair_y)
                for train, test in folds:
                    model = _classifier(classifier_name)
                    model.fit(x[train], pair_y[train])
                    prediction[test] = model.predict(x[test])
                classifier_results.append({
                    "classifier": classifier_name,
                    "balanced_accuracy": balanced_accuracy_score(pair_y, prediction),
                    "balanced_accuracy_ci95": _bootstrap_ci(pair_y, prediction, pair_groups),
                })
            best = max(classifier_results, key=lambda row: row["balanced_accuracy"])
            low, high = best["balanced_accuracy_ci95"]
            verdict = ("equivalent" if high < 0.60 else
                       "distinguishable" if low > 0.70 else "uncertain")
            pair_results.append({
                "left": left, "right": right, "feature_set": feature_name,
                "classifiers": classifier_results, "best_classifier": best["classifier"],
                "balanced_accuracy": best["balanced_accuracy"],
                "balanced_accuracy_ci95": best["balanced_accuracy_ci95"],
                "verdict": verdict,
            })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrices, verdict_counts, equivalence_classes = {}, {}, {}
    for feature_name in FEATURES:
        rows = [row for row in pair_results if row["feature_set"] == feature_name]
        matrix = np.full((len(CLASSES), len(CLASSES)), np.nan)
        for row in rows:
            left, right = CLASSES.index(row["left"]), CLASSES.index(row["right"])
            matrix[left, right] = matrix[right, left] = row["balanced_accuracy"]
        matrices[feature_name] = [
            [None if row == column else float(matrix[row, column])
             for column in range(len(CLASSES))]
            for row in range(len(CLASSES))
        ]
        counts = Counter(row["verdict"] for row in rows)
        verdict_counts[feature_name] = {
            verdict: counts[verdict]
            for verdict in ("equivalent", "distinguishable", "uncertain")
        }
        equivalence_classes[feature_name] = _components(
            CLASSES, [(row["left"], row["right"]) for row in rows
                      if row["verdict"] == "equivalent"])

        print(f"\n{feature_name} (better-classifier balanced accuracy)")
        width = max(map(len, CLASSES)) + 2
        print(" " * width + "".join(f"{name:>{width}}" for name in CLASSES))
        for index, name in enumerate(CLASSES):
            cells = ["" if index == column else f"{matrix[index, column]:.3f}"
                     for column in range(len(CLASSES))]
            print(f"{name:<{width}}" + "".join(f"{cell:>{width}}" for cell in cells))
        print("verdict counts:", verdict_counts[feature_name])
        print("equivalence classes:", equivalence_classes[feature_name])
        filename = "map_base_free.png" if feature_name == "all_base_free" else "map_delta_ref.png"
        _plot_matrix(matrix, CLASSES, feature_name, out_dir / filename)

    payload = {
        "classes": list(CLASSES), "n_roots": 40, "n_models": len(manifest),
        "features": list(FEATURES), "pairs": pair_results, "matrices": matrices,
        "verdict_counts": verdict_counts, "equivalence_classes": equivalence_classes,
    }
    with open(out_dir / "results_identifiability.json", "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    return payload


if __name__ == "__main__":
    evaluate_identifiability()
