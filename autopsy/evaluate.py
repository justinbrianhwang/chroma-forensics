"""Evaluate operation attribution with leakage-safe and contrast splits."""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LABELS = ("root", "null", "sft", "unlearn", "prune", "quant")
COMBOS = {
    "actgeom+logit": ("actgeom", "logit"),
    "all_base_free": ("actgeom", "logit", "weights", "blackbox"),
    "all_base_free_tda": ("actgeom", "logit", "weights", "blackbox", "topology"),
    "actgeom+topology": ("actgeom", "topology"),
}


def _load_manifest(out_dir):
    with open(Path(out_dir) / "manifest.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def _classifier(name):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0,
                                                                   random_state=0))
    return RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0)


def _bootstrap_ci(y_true, y_pred, groups, scorer=balanced_accuracy_score):
    unique_groups = np.unique(groups)
    rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(0)
    scores = np.empty(1000)
    for index in range(len(scores)):
        sampled = rng.choice(unique_groups, len(unique_groups), replace=True)
        selected = np.concatenate([rows[group] for group in sampled])
        scores[index] = scorer(y_true[selected], y_pred[selected])
    return np.percentile(scores, (2.5, 97.5)).tolist()


def evaluate(out_dir="runs/pilot"):
    out_dir = Path(out_dir)
    manifest = _load_manifest(out_dir)
    archive = np.load(out_dir / "fingerprints.npz")
    model_ids = [entry["model_id"] for entry in manifest]
    if archive["model_ids"].tolist() != model_ids:
        raise ValueError("fingerprints do not match manifest order")

    y = np.asarray([LABELS.index(entry["label"]) for entry in manifest])
    groups = np.asarray([entry["root_group"] for entry in manifest])
    feature_sets = {name: archive[name] for name in archive.files if name != "model_ids"}
    feature_sets.update({name: np.concatenate([archive[group] for group in members], axis=1)
                         for name, members in COMBOS.items()})

    n_groups = len(np.unique(groups))
    leave_root_out = GroupKFold(n_splits=min(8, n_groups))
    min_class_count = min(np.bincount(y))
    random_split = StratifiedKFold(n_splits=min(8, min_class_count), shuffle=True, random_state=0)
    splits = {
        "leave_root_out": list(leave_root_out.split(y, y, groups)),
        "random_split": list(random_split.split(y, y)),
    }

    results = []
    predictions = {}
    for feature_name, x in feature_sets.items():
        for classifier_name in ("logreg", "rf"):
            for split_name, folds in splits.items():
                oof = np.empty_like(y)
                for train, test in folds:
                    model = _classifier(classifier_name)
                    model.fit(x[train], y[train])
                    oof[test] = model.predict(x[test])
                balanced_accuracy = balanced_accuracy_score(y, oof)
                result = {
                    "feature_set": feature_name,
                    "classifier": classifier_name,
                    "split": split_name,
                    "balanced_accuracy": balanced_accuracy,
                    "macro_f1": f1_score(y, oof, average="macro"),
                    "balanced_accuracy_ci95": _bootstrap_ci(y, oof, groups),
                }
                results.append(result)
                predictions[(feature_name, classifier_name, split_name)] = oof

    base_free = set(feature_sets) - {"confound", "delta_ref"}
    candidates = [result for result in results
                  if result["split"] == "leave_root_out" and result["feature_set"] in base_free]
    best = max(candidates, key=lambda result: (result["balanced_accuracy"], result["macro_f1"]))
    best_predictions = predictions[(best["feature_set"], best["classifier"], best["split"])]
    best_base_free = {
        "feature_set": best["feature_set"],
        "classifier": best["classifier"],
        "split": best["split"],
        "per_class_recall": dict(zip(LABELS, recall_score(
            y, best_predictions, labels=range(len(LABELS)), average=None
        ).tolist())),
    }

    results.sort(key=lambda result: (-result["balanced_accuracy"], -result["macro_f1"],
                                     result["split"], result["feature_set"], result["classifier"]))
    print("feature_set                              classifier split             bal_acc macro_f1 95% CI")
    for result in results:
        low, high = result["balanced_accuracy_ci95"]
        print(f"{result['feature_set']:<40} {result['classifier']:<10} {result['split']:<17} "
              f"{result['balanced_accuracy']:.3f}   {result['macro_f1']:.3f}    [{low:.3f}, {high:.3f}]")
    best_tda = max(result["balanced_accuracy"] for result in results
                   if result["split"] == "leave_root_out" and
                   result["feature_set"] == "all_base_free_tda")
    best_base = max(result["balanced_accuracy"] for result in results
                    if result["split"] == "leave_root_out" and
                    result["feature_set"] == "all_base_free")
    tda_increment_lro = best_tda - best_base
    print(f"tda_increment_lro = {tda_increment_lro:.3f}")
    print("best base-free per-class recall:", best_base_free)

    payload = {
        "chance": 1 / len(LABELS),
        "labels": list(LABELS),
        "results": results,
        "best_base_free": best_base_free,
        "tda_increment_lro": tda_increment_lro,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload


if __name__ == "__main__":
    evaluate()
