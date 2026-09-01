"""Evaluate attribution after detector-aware adaptive laundering."""
import json
from pathlib import Path

import numpy as np

from .evaluate import COMBOS, LABELS, _bootstrap_ci, _classifier, _load_manifest
from .evaluate_laundering import _balanced_accuracy
from .evaluate_pairs import _root_folds


FEATURES = ("confound", "all_base_free", "all_base_free_tda", "topology",
            "weights", "actgeom+logit", "delta_ref")


def _load_run(path):
    path = Path(path)
    manifest = _load_manifest(path)
    archive = np.load(path / "fingerprints.npz")
    if archive["model_ids"].tolist() != [entry["model_id"] for entry in manifest]:
        raise ValueError(f"fingerprints do not match manifest order: {path}")
    arrays = {name: archive[name] for name in archive.files if name != "model_ids"}
    arrays.update({name: np.concatenate([archive[group] for group in members], axis=1)
                   for name, members in COMBOS.items()})
    return manifest, {name: arrays[name] for name in FEATURES}


def evaluate_adaptive(out_dir="runs/adaptive", source_dir="runs/pilot_matched"):
    out_dir = Path(out_dir)
    source_manifest, source_features = _load_run(source_dir)
    adaptive_manifest, adaptive_features = _load_run(out_dir)
    adaptive_keep = np.asarray([entry.get("lam") is not None
                                for entry in adaptive_manifest])
    adaptive_manifest = [entry for entry, keep in zip(adaptive_manifest, adaptive_keep)
                         if keep]
    adaptive_features = {name: values[adaptive_keep]
                         for name, values in adaptive_features.items()}
    shared_groups = {entry["root_group"] for entry in adaptive_manifest}
    source_keep = np.asarray([entry["root_group"] in shared_groups
                              for entry in source_manifest])
    source_manifest = [entry for entry, keep in zip(source_manifest, source_keep) if keep]
    source_features = {name: values[source_keep]
                       for name, values in source_features.items()}

    source_y = np.asarray([LABELS.index(entry["label"]) for entry in source_manifest])
    source_groups = np.asarray([entry["root_group"] for entry in source_manifest])
    source_ids = np.asarray([entry["model_id"] for entry in source_manifest])
    adaptive_y = np.asarray([LABELS.index(entry["label"]) for entry in adaptive_manifest])
    adaptive_groups = np.asarray([entry["root_group"] for entry in adaptive_manifest])
    adaptive_lams = np.asarray([float(entry["lam"]) for entry in adaptive_manifest])
    adaptive_ids = {entry["model_id"] for entry in adaptive_manifest}
    lams = sorted(set(adaptive_lams))
    reference_keep = np.asarray([entry["label"] != "root" for entry in source_manifest])
    reference_y = source_y[reference_keep]
    reference_groups = source_groups[reference_keep]
    folds = _root_folds(source_groups)

    results = []
    for feature_name in FEATURES:
        source_x = source_features[feature_name]
        adaptive_x = adaptive_features[feature_name]
        for classifier_name in ("logreg", "rf"):
            reference_prediction = np.empty(reference_keep.sum(), dtype=int)
            adaptive_prediction = np.empty(len(adaptive_manifest), dtype=int)
            for train_groups, test_groups in folds:
                train = np.flatnonzero(np.isin(source_groups, list(train_groups)))
                reference_test = np.flatnonzero(
                    np.isin(reference_groups, list(test_groups)))
                adaptive_test = np.flatnonzero(
                    np.isin(adaptive_groups, list(test_groups)))
                assert set(source_groups[train]).isdisjoint(
                    set(adaptive_groups[adaptive_test]))
                assert set(source_ids[train]).isdisjoint(adaptive_ids)
                assert all("__adaptive_" not in source_ids[index] for index in train), \
                    "adaptively laundered model entered a training fold"
                model = _classifier(classifier_name)
                model.fit(source_x[train], source_y[train])
                reference_prediction[reference_test] = model.predict(
                    source_x[reference_keep][reference_test])
                adaptive_prediction[adaptive_test] = model.predict(
                    adaptive_x[adaptive_test])

            reference_accuracy = _balanced_accuracy(reference_y, reference_prediction)
            for lam in lams:
                selected = adaptive_lams == lam
                prediction = adaptive_prediction[selected]
                results.append({
                    "lam": float(lam),
                    "feature_set": feature_name,
                    "classifier": classifier_name,
                    "balanced_accuracy": _balanced_accuracy(
                        adaptive_y[selected], prediction),
                    "balanced_accuracy_ci95": _bootstrap_ci(
                        adaptive_y[selected], prediction, adaptive_groups[selected],
                        scorer=_balanced_accuracy),
                    "unlaundered_balanced_accuracy": reference_accuracy,
                })

    source_acc = {entry["model_id"]: entry["eval_acc"] for entry in source_manifest}
    utility_cost = {
        str(lam): float(np.mean([
            entry["eval_acc"] - source_acc[entry["source_model_id"]]
            for entry in adaptive_manifest if float(entry["lam"]) == lam
        ])) for lam in lams
    }
    degradation_ranking = {}
    for lam in lams:
        ranking = []
        for feature_name in FEATURES:
            rows = [row for row in results
                    if row["lam"] == lam and row["feature_set"] == feature_name]
            unlaundered = max(row["unlaundered_balanced_accuracy"] for row in rows)
            adaptive = max(row["balanced_accuracy"] for row in rows)
            ranking.append({
                "feature_set": feature_name,
                "unlaundered_balanced_accuracy": unlaundered,
                "adaptive_balanced_accuracy": adaptive,
                "degradation": unlaundered - adaptive,
            })
        degradation_ranking[str(lam)] = sorted(
            ranking, key=lambda row: row["degradation"], reverse=True)

    print("lam feature_set          classifier adaptive 95% CI          unlaundered")
    for row in results:
        low, high = row["balanced_accuracy_ci95"]
        print(f"{row['lam']:<3g} {row['feature_set']:<20} {row['classifier']:<10} "
              f"{row['balanced_accuracy']:.3f}    [{low:.3f}, {high:.3f}]  "
              f"{row['unlaundered_balanced_accuracy']:.3f}")
    for lam in lams:
        print(f"mean_eval_acc_delta[{lam:g}] = {utility_cost[str(lam)]:.4f}")
        print(f"descriptor degradation ranking[{lam:g}]")
        for row in degradation_ranking[str(lam)]:
            print(f"  {row['feature_set']:<20} {row['degradation']:+.3f} "
                  f"({row['unlaundered_balanced_accuracy']:.3f} -> "
                  f"{row['adaptive_balanced_accuracy']:.3f})")

    payload = {
        "chance": 1 / len(LABELS),
        "n_adaptive": len(adaptive_manifest),
        "lams": lams,
        "results": results,
        "mean_eval_acc_delta": utility_cost,
        "descriptor_degradation_ranking": degradation_ranking,
        "training_data": "unlaundered pilot_matched only",
    }
    with open(out_dir / "results_adaptive.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload


if __name__ == "__main__":
    evaluate_adaptive()
