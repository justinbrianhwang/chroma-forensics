"""Evaluate operation presence, order, and composition generalization."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .evaluate import COMBOS, _bootstrap_ci, _classifier, _load_manifest


OPS = ("sft", "unlearn", "prune", "quant")
FEATURES = ("confound", "all_base_free", "all_base_free_tda", "topology", "delta_ref")
PAIR_TYPES = (("sft", "unlearn"), ("sft", "prune"), ("sft", "quant"))


def _load_run(path):
    path = Path(path)
    manifest = _load_manifest(path)
    archive = np.load(path / "fingerprints.npz")
    if archive["model_ids"].tolist() != [entry["model_id"] for entry in manifest]:
        raise ValueError(f"fingerprints do not match manifest order: {path}")
    feature_sets = {name: archive[name] for name in archive.files if name != "model_ids"}
    feature_sets.update({name: np.concatenate([archive[group] for group in members], axis=1)
                         for name, members in COMBOS.items()})
    return manifest, {name: feature_sets[name] for name in FEATURES}


def _root_folds(groups):
    splitter = GroupKFold(n_splits=min(8, len(np.unique(groups))))
    folds = []
    for train, test in splitter.split(groups, groups, groups):
        train_groups, test_groups = set(groups[train]), set(groups[test])
        assert train_groups.isdisjoint(test_groups)
        folds.append((train_groups, test_groups))
    return folds


def _indices(groups, folds):
    result = []
    for train_groups, test_groups in folds:
        train = np.flatnonzero(np.isin(groups, list(train_groups)))
        test = np.flatnonzero(np.isin(groups, list(test_groups)))
        assert set(groups[train]).isdisjoint(groups[test])
        result.append((train, test))
    return result


def _binary_oof(x, y, splits, classifier_name):
    probabilities = np.empty(len(y))
    for train, test in splits:
        model = _classifier(classifier_name)
        model.fit(x[train], y[train])
        probabilities[test] = model.predict_proba(x[test])[:, 1]
    return probabilities


def evaluate_pairs(out_dir="runs/pairs", singles_dir="runs/pilot_matched"):
    out_dir = Path(out_dir)
    singles_manifest, singles_x = _load_run(singles_dir)
    pairs_manifest, pairs_x = _load_run(out_dir)

    # Keep the roots from the singles run and discard their copied duplicates.
    keep_pairs = np.asarray([entry["label"] != "root" for entry in pairs_manifest])
    manifest = singles_manifest + [entry for entry in pairs_manifest if entry["label"] != "root"]
    feature_sets = {name: np.concatenate((singles_x[name], pairs_x[name][keep_pairs]))
                    for name in FEATURES}
    groups = np.asarray([entry["root_group"] for entry in manifest])
    histories = [entry["history"] for entry in manifest]
    folds = _root_folds(groups)
    splits = _indices(groups, folds)

    task1 = []
    for feature_name, x in feature_sets.items():
        for classifier_name in ("logreg", "rf"):
            per_op = {}
            for op in OPS:
                y = np.asarray([op in history for history in histories], dtype=int)
                probability = _binary_oof(x, y, splits, classifier_name)
                per_op[op] = {
                    "auroc": roc_auc_score(y, probability),
                    "f1": f1_score(y, probability >= 0.5),
                }
            task1.append({
                "feature_set": feature_name, "classifier": classifier_name,
                "per_op": per_op,
                "macro_auroc": float(np.mean([row["auroc"] for row in per_op.values()])),
                "macro_f1": float(np.mean([row["f1"] for row in per_op.values()])),
            })

    pair_history = [entry["history"] for entry in pairs_manifest]
    pair_groups = np.asarray([entry["root_group"] for entry in pairs_manifest])
    task2 = []
    for left, right in PAIR_TYPES:
        selected = np.asarray([len(history) == 2 and set(history) == {left, right}
                               for history in pair_history])
        y = np.asarray([history[0] == "sft" for history, keep in zip(pair_history, selected)
                        if keep], dtype=int)
        selected_groups = pair_groups[selected]
        pair_splits = _indices(selected_groups, folds)
        for feature_name in FEATURES:
            x = pairs_x[feature_name][selected]
            for classifier_name in ("logreg", "rf"):
                probability = _binary_oof(x, y, pair_splits, classifier_name)
                prediction = probability >= 0.5
                task2.append({
                    "pair_type": f"{left}/{right}", "feature_set": feature_name,
                    "classifier": classifier_name,
                    "accuracy": accuracy_score(y, prediction),
                    "accuracy_ci95": _bootstrap_ci(
                        y, prediction, selected_groups, scorer=accuracy_score),
                })

    def best_order(pair_type, feature_name):
        return max(row["accuracy"] for row in task2
                   if row["pair_type"] == pair_type and row["feature_set"] == feature_name)

    tda_order_increment = float(np.mean([
        best_order(f"{left}/{right}", "all_base_free_tda") -
        best_order(f"{left}/{right}", "all_base_free") for left, right in PAIR_TYPES
    ]))

    single_groups = np.asarray([entry["root_group"] for entry in singles_manifest])
    pair_only = np.asarray([len(history) == 2 for history in pair_history])
    test_groups = pair_groups[pair_only]
    composition_splits = []
    for train_roots, test_roots in folds:
        train = np.flatnonzero(np.isin(single_groups, list(train_roots)))
        test = np.flatnonzero(np.isin(test_groups, list(test_roots)))
        assert set(single_groups[train]).isdisjoint(test_groups[test])
        composition_splits.append((train, test))

    task3 = []
    test_histories = [history for history in pair_history if len(history) == 2]
    for feature_name in FEATURES:
        train_x, test_x = singles_x[feature_name], pairs_x[feature_name][pair_only]
        for classifier_name in ("logreg", "rf"):
            per_op = {}
            for op in OPS:
                train_y = np.asarray([op in entry["history"] for entry in singles_manifest], dtype=int)
                test_y = np.asarray([op in history for history in test_histories], dtype=int)
                probabilities = np.empty(len(test_y))
                for train, test in composition_splits:
                    model = _classifier(classifier_name)
                    model.fit(train_x[train], train_y[train])
                    probabilities[test] = model.predict_proba(test_x[test])[:, 1]
                per_op[op] = {
                    "recall": recall_score(test_y, probabilities >= 0.5),
                    "auroc": (roc_auc_score(test_y, probabilities)
                              if len(np.unique(test_y)) == 2 else None),
                }
            task3.append({"feature_set": feature_name, "classifier": classifier_name,
                          "per_op": per_op})

    print("\nTask 1: operation presence")
    print("per-op cells: AUROC/F1")
    print("feature_set          classifier  " + "  ".join(f"{op:>13}" for op in OPS) +
          "  macro_auc macro_f1")
    for row in task1:
        cells = "  ".join(f"{row['per_op'][op]['auroc']:.3f}/{row['per_op'][op]['f1']:.3f}"
                          for op in OPS)
        print(f"{row['feature_set']:<20} {row['classifier']:<10} {cells}  "
              f"{row['macro_auroc']:.3f}     {row['macro_f1']:.3f}")

    print("\nTask 2: order prediction")
    print("pair_type    feature_set          classifier accuracy 95% CI")
    for row in task2:
        low, high = row["accuracy_ci95"]
        print(f"{row['pair_type']:<12} {row['feature_set']:<20} {row['classifier']:<10} "
              f"{row['accuracy']:.3f}    [{low:.3f}, {high:.3f}]")
    print(f"tda_order_increment = {tda_order_increment:.3f}")

    print("\nTask 3: leave-composition-out")
    print("per-op cells: recall/AUROC")
    print("feature_set          classifier  " + "  ".join(f"{op:>13}" for op in OPS))
    for row in task3:
        cells = "  ".join(
            f"{row['per_op'][op]['recall']:.3f}/" +
            (f"{row['per_op'][op]['auroc']:.3f}"
             if row['per_op'][op]['auroc'] is not None else "n/a") for op in OPS)
        print(f"{row['feature_set']:<20} {row['classifier']:<10} {cells}")

    payload = {
        "n_models": len(manifest), "operations": list(OPS),
        "task1_presence": task1, "task2_order": task2,
        "tda_order_increment": tda_order_increment,
        "task3_leave_composition_out": task3,
    }
    with open(out_dir / "results_pairs.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload


if __name__ == "__main__":
    evaluate_pairs()
