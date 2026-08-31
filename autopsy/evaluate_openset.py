"""Leakage-safe open-set evaluation of unknown post-training operations."""
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from .evaluate import LABELS, _classifier
from .evaluate_pairs import FEATURES, _load_run, _root_folds


SCORES = ("msp", "energy", "maha")
UNKNOWN_LABELS = ("distill", "merge")


def _mahalanobis(train_x, train_y, test_x):
    scaler = StandardScaler().fit(train_x)
    train_z = scaler.transform(train_x)
    test_z = scaler.transform(test_x)
    means = np.stack([train_z[train_y == label].mean(0) for label in range(len(LABELS))])
    variance = np.mean((train_z - means[train_y]) ** 2, axis=0).clip(1e-12)
    return np.sqrt(np.min(
        np.sum((test_z[:, None, :] - means[None, :, :]) ** 2 / variance, axis=2), axis=1
    ))


def _scores(model, train_x, train_y, test_x, classifier_name):
    probability = model.predict_proba(test_x)
    if classifier_name == "logreg":
        energy = -logsumexp(model.decision_function(test_x), axis=1)
    else:
        with np.errstate(divide="ignore"):
            energy = -logsumexp(model.predict_log_proba(test_x), axis=1)
    return {
        "msp": 1 - probability.max(1),
        "energy": energy,
        "maha": _mahalanobis(train_x, train_y, test_x),
    }


def _fpr95(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    return float(fpr[np.flatnonzero(tpr >= 0.95)[0]])


def _coverage_accuracy(y, prediction, score, coverage):
    retained = np.argsort(score)[:int(np.ceil(coverage * len(y)))]
    y, prediction = y[retained], prediction[retained]
    return float(np.mean([np.mean(prediction[y == label] == label) for label in np.unique(y)]))


def evaluate_openset(out_dir="runs/openset", known_dir="runs/pilot_matched"):
    out_dir = Path(out_dir)
    known_manifest, known_features = _load_run(known_dir)
    unknown_manifest, unknown_features = _load_run(out_dir)
    shared_groups = {entry["root_group"] for entry in unknown_manifest}
    known_keep = np.asarray([entry["root_group"] in shared_groups for entry in known_manifest])
    unknown_keep = np.asarray([entry["label"] in UNKNOWN_LABELS for entry in unknown_manifest])
    known_manifest = [entry for entry, keep in zip(known_manifest, known_keep) if keep]
    unknown_manifest = [entry for entry, keep in zip(unknown_manifest, unknown_keep) if keep]
    known_features = {name: values[known_keep] for name, values in known_features.items()}
    unknown_features = {name: values[unknown_keep] for name, values in unknown_features.items()}

    known_y = np.asarray([LABELS.index(entry["label"]) for entry in known_manifest])
    known_groups = np.asarray([entry["root_group"] for entry in known_manifest])
    unknown_groups = np.asarray([entry["root_group"] for entry in unknown_manifest])
    unknown_kinds = np.asarray([entry["label"] for entry in unknown_manifest])
    known_ids = np.asarray([entry["model_id"] for entry in known_manifest])
    unknown_ids = set(entry["model_id"] for entry in unknown_manifest)
    folds = _root_folds(known_groups)

    results = []
    for feature_name in FEATURES:
        known_x, unknown_x = known_features[feature_name], unknown_features[feature_name]
        for classifier_name in ("logreg", "rf"):
            known_prediction = np.empty(len(known_y), dtype=int)
            known_scores = {name: np.empty(len(known_y)) for name in SCORES}
            unknown_scores = {name: np.empty(len(unknown_manifest)) for name in SCORES}
            for train_groups, test_groups in folds:
                train = np.flatnonzero(np.isin(known_groups, list(train_groups)))
                known_test = np.flatnonzero(np.isin(known_groups, list(test_groups)))
                unknown_test = np.flatnonzero(np.isin(unknown_groups, list(test_groups)))
                assert set(known_ids[train]).isdisjoint(unknown_ids)
                assert all(known_manifest[index]["label"] not in UNKNOWN_LABELS for index in train)
                model = _classifier(classifier_name)
                model.fit(known_x[train], known_y[train])
                known_prediction[known_test] = model.predict(known_x[known_test])
                fold_known = _scores(model, known_x[train], known_y[train],
                                     known_x[known_test], classifier_name)
                fold_unknown = _scores(model, known_x[train], known_y[train],
                                       unknown_x[unknown_test], classifier_name)
                for score_name in SCORES:
                    known_scores[score_name][known_test] = fold_known[score_name]
                    unknown_scores[score_name][unknown_test] = fold_unknown[score_name]

            for score_name in SCORES:
                y_unknown = np.concatenate((np.zeros(len(known_y), dtype=int),
                                            np.ones(len(unknown_manifest), dtype=int)))
                score = np.concatenate((known_scores[score_name], unknown_scores[score_name]))
                per_unknown = {}
                for kind in UNKNOWN_LABELS:
                    selected = unknown_kinds == kind
                    per_unknown[kind] = float(roc_auc_score(
                        np.concatenate((np.zeros(len(known_y), dtype=int),
                                        np.ones(selected.sum(), dtype=int))),
                        np.concatenate((known_scores[score_name],
                                        unknown_scores[score_name][selected])),
                    ))
                results.append({
                    "feature_set": feature_name,
                    "classifier": classifier_name,
                    "score": score_name,
                    "auroc": float(roc_auc_score(y_unknown, score)),
                    "fpr_at_95_tpr": _fpr95(y_unknown, score),
                    "balanced_accuracy_at_90_coverage": _coverage_accuracy(
                        known_y, known_prediction, known_scores[score_name], 0.9),
                    "balanced_accuracy_at_70_coverage": _coverage_accuracy(
                        known_y, known_prediction, known_scores[score_name], 0.7),
                    "per_unknown_auroc": per_unknown,
                })

    best_tda = max(row["auroc"] for row in results
                   if row["feature_set"] == "all_base_free_tda")
    best_base = max(row["auroc"] for row in results
                    if row["feature_set"] == "all_base_free")
    tda_increment = best_tda - best_base
    print("feature_set          classifier score   AUROC FPR@95 BA@90 BA@70 distill merge")
    for row in results:
        print(f"{row['feature_set']:<20} {row['classifier']:<10} {row['score']:<7} "
              f"{row['auroc']:.3f} {row['fpr_at_95_tpr']:.3f}  "
              f"{row['balanced_accuracy_at_90_coverage']:.3f}  "
              f"{row['balanced_accuracy_at_70_coverage']:.3f}  "
              f"{row['per_unknown_auroc']['distill']:.3f}   "
              f"{row['per_unknown_auroc']['merge']:.3f}")
    print(f"tda_openset_increment = {tda_increment:.3f}")

    payload = {
        "n_known": len(known_manifest),
        "n_unknown": len(unknown_manifest),
        "unknown_labels": list(UNKNOWN_LABELS),
        "results": results,
        "tda_openset_increment": tda_increment,
    }
    with open(out_dir / "results_openset.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload


if __name__ == "__main__":
    evaluate_openset()
