"""Contrast raw-weight PCA under root-held-out and random cross-validation."""
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold

from .baseline_features import _manifest
from .evaluate import LABELS, _bootstrap_ci, _classifier
from .evaluate_baselines import _balanced_accuracy, _dual_pca


CLASSIFIERS = ("logreg", "rf")


def evaluate(out_path="runs/revision/random_split_contrast.json",
             raw_path="runs/baselines/raw_features.npz"):
    entries = _manifest(Path("runs/pilot_matched"))
    y = np.asarray([LABELS.index(entry["label"]) for entry in entries])
    groups = np.asarray([entry["root_group"] for entry in entries])

    with np.load(raw_path) as archive:
        indices = np.flatnonzero(archive["run_names"] == "pilot_matched")
        if archive["model_ids"][indices].tolist() != [entry["model_id"] for entry in entries]:
            raise ValueError("raw weight features do not match pilot_matched manifest order")
        gram = archive["raw_weight_gram"][np.ix_(indices, indices)]

    folds = {
        "leave_root_out": list(GroupKFold(8).split(y, y, groups)),
        "random_split": list(StratifiedKFold(
            8, shuffle=True, random_state=0).split(y, y)),
    }
    results = []
    for split_name, split_folds in folds.items():
        predictions = {name: np.full(len(y), -1, dtype=int) for name in CLASSIFIERS}
        for train, test in split_folds:
            if split_name == "leave_root_out":
                assert set(groups[train]).isdisjoint(groups[test])
            train_x, (test_x,) = _dual_pca(gram, train, (test,), requested_components=128)
            assert train_x.shape[1] == test_x.shape[1] == 128
            for classifier_name in CLASSIFIERS:
                model = _classifier(classifier_name)
                model.fit(train_x, y[train])
                predictions[classifier_name][test] = model.predict(test_x)

        for classifier_name, prediction in predictions.items():
            assert (prediction >= 0).all()
            score = _balanced_accuracy(y, prediction)
            results.append({
                "split": split_name,
                "classifier": classifier_name,
                "balanced_accuracy": score,
                "ci95": _bootstrap_ci(
                    y, prediction, groups, scorer=_balanced_accuracy),
            })

    by_key = {(row["classifier"], row["split"]): row for row in results}
    print("classifier  leave-root-out              random split")
    for classifier_name in CLASSIFIERS:
        cells = []
        for split_name in folds:
            row = by_key[classifier_name, split_name]
            low, high = row["ci95"]
            cells.append(f"{row['balanced_accuracy']:.3f} [{low:.3f}, {high:.3f}]")
        print(f"{classifier_name:<10}  {cells[0]:<27} {cells[1]}")

    payload = {
        "task": "primary_6class",
        "metric": "balanced_accuracy",
        "labels": list(LABELS),
        "population": len(entries),
        "root_groups": len(np.unique(groups)),
        "folds": 8,
        "pca_components": 128,
        "bootstrap_unit": "root_group",
        "results": results,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as file:
        json.dump(payload, file, indent=2)
    return payload


if __name__ == "__main__":
    evaluate()
