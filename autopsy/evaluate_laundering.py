"""Evaluate attribution after unseen non-adaptive laundering transforms."""
import json
from pathlib import Path

import numpy as np
from .evaluate import LABELS, _bootstrap_ci, _classifier
from .evaluate_pairs import FEATURES, _load_run, _root_folds


LAUNDRIES = ("ft", "quant", "noise")


def _balanced_accuracy(y, prediction):
    return float(np.mean([
        np.mean(prediction[y == label] == label) for label in np.unique(y)
    ]))


def evaluate_laundering(out_dir="runs/laundered", source_dir="runs/pilot_matched"):
    out_dir = Path(out_dir)
    source_manifest, source_features = _load_run(source_dir)
    laundered_manifest, laundered_features = _load_run(out_dir)
    shared_groups = {entry["root_group"] for entry in laundered_manifest}
    source_keep = np.asarray([entry["root_group"] in shared_groups
                              for entry in source_manifest])
    source_manifest = [entry for entry, keep in zip(source_manifest, source_keep) if keep]
    source_features = {name: values[source_keep] for name, values in source_features.items()}
    laundered_keep = np.asarray([entry.get("laundry") in LAUNDRIES
                                 for entry in laundered_manifest])
    laundered_manifest = [entry for entry, keep in zip(laundered_manifest, laundered_keep)
                          if keep]
    laundered_features = {name: values[laundered_keep]
                          for name, values in laundered_features.items()}

    source_index = {entry["model_id"]: index for index, entry in enumerate(source_manifest)}
    originals = np.asarray([source_index[entry["model_id"].rsplit("__", 1)[0]]
                            for entry in laundered_manifest])
    source_y = np.asarray([LABELS.index(entry["label"]) for entry in source_manifest])
    source_groups = np.asarray([entry["root_group"] for entry in source_manifest])
    source_ids = np.asarray([entry["model_id"] for entry in source_manifest])
    laundered_y = np.asarray([LABELS.index(entry["label"]) for entry in laundered_manifest])
    laundered_groups = np.asarray([entry["root_group"] for entry in laundered_manifest])
    laundry_types = np.asarray([entry["laundry"] for entry in laundered_manifest])
    laundered_ids = set(entry["model_id"] for entry in laundered_manifest)
    reference_keep = np.asarray([entry["label"] != "root" for entry in source_manifest])
    reference_groups = source_groups[reference_keep]
    reference_y = source_y[reference_keep]
    folds = _root_folds(source_groups)

    results = []
    for feature_name in FEATURES:
        source_x = source_features[feature_name]
        laundered_x = laundered_features[feature_name]
        for classifier_name in ("logreg", "rf"):
            reference_prediction = np.empty(reference_keep.sum(), dtype=int)
            laundered_prediction = np.empty(len(laundered_manifest), dtype=int)
            for train_groups, test_groups in folds:
                train = np.flatnonzero(np.isin(source_groups, list(train_groups)))
                reference_test = np.flatnonzero(np.isin(reference_groups, list(test_groups)))
                laundered_test = np.flatnonzero(np.isin(laundered_groups, list(test_groups)))
                assert set(source_ids[train]).isdisjoint(laundered_ids)
                assert all(source_manifest[index].get("laundry") is None for index in train)
                model = _classifier(classifier_name)
                model.fit(source_x[train], source_y[train])
                reference_prediction[reference_test] = model.predict(
                    source_x[reference_keep][reference_test])
                laundered_prediction[laundered_test] = model.predict(
                    laundered_x[laundered_test])

            reference_accuracy = _balanced_accuracy(reference_y, reference_prediction)
            for laundry in LAUNDRIES:
                selected = laundry_types == laundry
                prediction = laundered_prediction[selected]
                results.append({
                    "laundry": laundry,
                    "feature_set": feature_name,
                    "classifier": classifier_name,
                    "balanced_accuracy": _balanced_accuracy(laundered_y[selected], prediction),
                    "balanced_accuracy_ci95": _bootstrap_ci(
                        laundered_y[selected], prediction, laundered_groups[selected],
                        scorer=_balanced_accuracy),
                    "unlaundered_balanced_accuracy": reference_accuracy,
                })

    utility_cost = {}
    source_acc = np.asarray([entry["eval_acc"] for entry in source_manifest])
    for laundry in LAUNDRIES:
        selected = laundry_types == laundry
        utility_cost[laundry] = float(np.mean(
            np.asarray([entry["eval_acc"] for entry, keep in zip(laundered_manifest, selected)
                        if keep]) - source_acc[originals[selected]]
        ))

    increments = {}
    for laundry in LAUNDRIES:
        best_tda = max(row["balanced_accuracy"] for row in results
                       if row["laundry"] == laundry and
                       row["feature_set"] == "all_base_free_tda")
        best_base = max(row["balanced_accuracy"] for row in results
                        if row["laundry"] == laundry and
                        row["feature_set"] == "all_base_free")
        increments[laundry] = best_tda - best_base

    print("laundry feature_set          classifier laundered 95% CI          unlaundered")
    for row in results:
        low, high = row["balanced_accuracy_ci95"]
        print(f"{row['laundry']:<7} {row['feature_set']:<20} {row['classifier']:<10} "
              f"{row['balanced_accuracy']:.3f}     [{low:.3f}, {high:.3f}]  "
              f"{row['unlaundered_balanced_accuracy']:.3f}")
    for laundry in LAUNDRIES:
        print(f"mean_eval_acc_delta[{laundry}] = {utility_cost[laundry]:.4f}")
        print(f"tda_laundering_increment[{laundry}] = {increments[laundry]:.3f}")

    payload = {
        "n_laundered": len(laundered_manifest),
        "results": results,
        "mean_eval_acc_delta": utility_cost,
        "tda_laundering_increment": increments,
    }
    with open(out_dir / "results_laundering.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload


if __name__ == "__main__":
    evaluate_laundering()
