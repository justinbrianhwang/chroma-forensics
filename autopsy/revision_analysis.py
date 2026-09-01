"""SPEC9 re-analyses from committed manifests, fingerprints, and result JSONs."""
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from .evaluate import LABELS, _classifier, _load_manifest
from .evaluate_grid import PROBE_GROUPS
from .evaluate_pairs import _load_run as _load_feature_run
from .identifiability import CLASSES, FEATURES, _load_run as _load_identity_run


def _oof(x, y, groups, classifier_name):
    prediction = np.empty_like(y)
    folds = GroupKFold(n_splits=min(8, len(np.unique(groups))))
    for train, test in folds.split(y, y, groups):
        assert set(groups[train]).isdisjoint(groups[test])
        model = _classifier(classifier_name)
        model.fit(x[train], y[train])
        prediction[test] = model.predict(x[test])
    return prediction


def _group_scores(y, prediction, groups):
    unique_groups = np.unique(groups)
    classes = np.unique(y)
    scores = []
    for group in unique_groups:
        selected = groups == group
        assert np.array_equal(np.unique(y[selected]), classes)
        scores.append(balanced_accuracy_score(y[selected], prediction[selected]))
    return np.asarray(scores, dtype=float)


def _bootstrap_distribution(y, prediction, groups, n_bootstrap=1000):
    scores = _group_scores(y, prediction, groups)
    rng = np.random.default_rng(0)
    samples = rng.integers(len(scores), size=(n_bootstrap, len(scores)))
    return scores[samples].mean(1)


def _bh(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 1.0
    for position in range(len(order) - 1, -1, -1):
        index = order[position]
        running = min(running, p_values[index] * len(order) / (position + 1))
        adjusted[index] = running
    return adjusted


def _bootstrap_variants(y, prediction, groups, inner=200):
    scores = _group_scores(y, prediction, groups)
    theta = float(scores.mean())
    rng = np.random.default_rng(0)
    outer_indices = rng.integers(len(scores), size=(1000, len(scores)))
    outer = scores[outer_indices].mean(1)
    percentile = np.percentile(outer, (2.5, 97.5))
    basic = np.asarray((2 * theta - percentile[1], 2 * theta - percentile[0]))

    original_inner = scores[rng.integers(
        len(scores), size=(inner, len(scores)))].mean(1)
    original_se = original_inner.std(ddof=1)
    t_values = np.empty(len(outer))
    for index, sampled in enumerate(scores[outer_indices]):
        inner_scores = sampled[rng.integers(
            len(sampled), size=(inner, len(sampled)))].mean(1)
        inner_se = inner_scores.std(ddof=1)
        t_values[index] = ((outer[index] - theta) / inner_se
                           if inner_se > 0 else 0.0)
    t_low, t_high = np.percentile(t_values, (2.5, 97.5))
    studentized = np.asarray((theta - t_high * original_se,
                              theta - t_low * original_se))
    return {
        "balanced_accuracy": theta,
        "percentile": percentile.tolist(),
        "basic": basic.tolist(),
        "studentized": studentized.tolist(),
        "outer_bootstrap": 1000,
        "inner_bootstrap": inner,
    }


def _identity_reanalysis():
    loaded = [
        _load_identity_run("runs/pilot_matched", True),
        _load_identity_run("runs/pairs", False),
        _load_identity_run("runs/openset", False),
    ]
    manifest = sum((run_manifest for run_manifest, _ in loaded), [])
    feature_sets = {
        name: np.concatenate([arrays[name] for _, arrays in loaded]) for name in FEATURES
    }
    labels = np.asarray([entry["label"] for entry in manifest])
    groups = np.asarray([entry["root_group"] for entry in manifest])
    with open("runs/identifiability/results_identifiability.json") as stream:
        stored = json.load(stream)
    stored_rows = {(row["left"], row["right"], row["feature_set"]): row
                   for row in stored["pairs"]}

    pair_rows = []
    for feature_name in FEATURES:
        feature_rows = []
        for left, right in itertools.combinations(CLASSES, 2):
            selected = np.isin(labels, (left, right))
            y = (labels[selected] == right).astype(int)
            pair_groups = groups[selected]
            classifier_name = stored_rows[left, right, feature_name]["best_classifier"]
            prediction = _oof(feature_sets[feature_name][selected], y, pair_groups,
                              classifier_name)
            distribution = _bootstrap_distribution(y, prediction, pair_groups)
            row = {
                "left": left,
                "right": right,
                "feature_set": feature_name,
                "classifier": classifier_name,
                "balanced_accuracy": balanced_accuracy_score(y, prediction),
                "balanced_accuracy_ci95": np.percentile(
                    distribution, (2.5, 97.5)).tolist(),
                "p_equivalent": {
                    str(delta): float(np.mean(distribution <= 0.5 + delta))
                    for delta in (0.05, 0.10, 0.15)
                },
                "bootstrap_p_gt_chance": float(
                    (1 + np.count_nonzero(distribution <= 0.5)) /
                    (len(distribution) + 1)),
            }
            feature_rows.append(row)
        adjusted = _bh(np.asarray([
            row["bootstrap_p_gt_chance"] for row in feature_rows]))
        for row, q_value in zip(feature_rows, adjusted):
            row["bh_q"] = float(q_value)
        pair_rows.extend(feature_rows)

    verdict_grid = {}
    bh_significant = {}
    for feature_name in FEATURES:
        rows = [row for row in pair_rows if row["feature_set"] == feature_name]
        grid_rows = []
        for delta in (0.05, 0.10, 0.15):
            for threshold in (0.65, 0.70, 0.75):
                counts = Counter()
                for row in rows:
                    distribution_equivalent = row["p_equivalent"][str(delta)] >= 0.95
                    low = row["balanced_accuracy_ci95"][0]
                    verdict = ("equivalent" if distribution_equivalent else
                               "distinguishable" if low > threshold else "uncertain")
                    counts[verdict] += 1
                grid_rows.append({
                    "delta": delta,
                    "distinguishable_threshold": threshold,
                    **{name: counts[name] for name in (
                        "equivalent", "distinguishable", "uncertain")},
                })
        verdict_grid[feature_name] = grid_rows
        bh_significant[feature_name] = sum(row["bh_q"] <= 0.05 for row in rows)
    return pair_rows, verdict_grid, bh_significant, stored


def _recency_test(stored):
    uncertain = {
        (row["left"], row["right"])
        for row in stored["pairs"]
        if row["feature_set"] == "all_base_free" and row["verdict"] == "uncertain"
    }
    assert len(uncertain) == 6

    def final_op(label):
        if label == "root":
            return None
        if label in {"distill", "merge"}:
            return "near-identity"
        return label.rsplit("-", 1)[-1]

    counts = np.zeros((2, 2), dtype=int)
    categorized = []
    for pair in itertools.combinations(CLASSES, 2):
        left, right = pair
        category = (final_op(left) == final_op(right) or
                    "near-identity" in {final_op(left), final_op(right)})
        is_uncertain = pair in uncertain
        counts[0 if is_uncertain else 1, 0 if category else 1] += 1
        if category:
            categorized.append(pair)
    odds_ratio, p_value = fisher_exact(counts, alternative="greater")
    return {
        "rows": ["uncertain", "not_uncertain"],
        "columns": ["same_final_or_near_identity", "other"],
        "table": counts.tolist(),
        "odds_ratio": None if not np.isfinite(odds_ratio) else float(odds_ratio),
        "p_value_greater": float(p_value),
        "uncertain_pairs": [list(pair) for pair in sorted(uncertain)],
        "category_pair_count": len(categorized),
    }


def _hard_subset():
    manifest, features = _load_feature_run("runs/pilot_matched")
    selected = np.asarray([entry["label"] in {"root", "null", "sft", "unlearn"}
                           for entry in manifest])
    labels = ("root", "null", "sft", "unlearn")
    y = np.asarray([labels.index(entry["label"])
                    for entry, keep in zip(manifest, selected) if keep])
    groups = np.asarray([entry["root_group"]
                         for entry, keep in zip(manifest, selected) if keep])
    rows = []
    for feature_name in ("all_base_free", "confound"):
        for classifier_name in ("logreg", "rf"):
            prediction = _oof(features[feature_name][selected], y, groups,
                              classifier_name)
            distribution = _bootstrap_distribution(y, prediction, groups)
            rows.append({
                "feature_set": feature_name,
                "classifier": classifier_name,
                "balanced_accuracy": balanced_accuracy_score(y, prediction),
                "balanced_accuracy_ci95": np.percentile(
                    distribution, (2.5, 97.5)).tolist(),
            })
    return {"chance": 0.25, "labels": list(labels), "results": rows}


def _bootstrap_checks():
    manifest, features = _load_feature_run("runs/pilot_matched")
    y = np.asarray([LABELS.index(entry["label"]) for entry in manifest])
    groups = np.asarray([entry["root_group"] for entry in manifest])
    matched_prediction = _oof(features["all_base_free"], y, groups, "logreg")

    grid_manifest = _load_manifest("runs/grid")
    archive = np.load(Path("runs/grid") / "fingerprints.npz")
    keep = np.asarray([entry.get("replicate", 0) == 0 for entry in grid_manifest])
    grid_y = np.asarray([LABELS.index(entry["label"])
                         for entry, selected in zip(grid_manifest, keep) if selected])
    grid_groups = np.asarray([entry["root_group"]
                              for entry, selected in zip(grid_manifest, keep) if selected])
    grid_x = np.concatenate([archive[name] for name in PROBE_GROUPS], axis=1)[keep]
    grid_prediction = _oof(grid_x, grid_y, grid_groups, "rf")
    return {
        "matched_6way_all_base_free_logreg": _bootstrap_variants(
            y, matched_prediction, groups),
        "grid_pooled_probe_base_free_rf": _bootstrap_variants(
            grid_y, grid_prediction, grid_groups),
    }


def _utility_table():
    rows = []
    for run, population_fields in (
        ("runs/grid", ("dataset", "arch")),
        ("runs/pilot_matched", ()),
    ):
        manifest = _load_manifest(run)
        if run == "runs/grid":
            manifest = [entry for entry in manifest if entry.get("replicate", 0) == 0]
            populations = sorted({tuple(entry[field] for field in population_fields)
                                  for entry in manifest})
        else:
            populations = [("pilot_matched",)]
        for population in populations:
            selected = [entry for entry in manifest
                        if (run != "runs/grid" or
                            tuple(entry[field] for field in population_fields) == population)]
            statistics = {}
            for label in LABELS:
                values = np.asarray([entry["eval_acc"] for entry in selected
                                     if entry["label"] == label])
                statistics[label] = {
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "n": len(values),
                }
            rows.append({"population": "/".join(population), **statistics})
    return {"grid_replicates_included": False, "rows": rows}


def _tally():
    run_names = ("pilot", "pilot_matched", "pairs", "openset", "laundered", "grid")
    counts = {name: len(_load_manifest(Path("runs") / name)) for name in run_names}
    adaptive_manifest = Path("runs/adaptive/manifest.jsonl")
    if adaptive_manifest.exists():
        counts["adaptive"] = len(_load_manifest("runs/adaptive"))
    counts["calibration_roots"] = 3
    return {"counts": counts, "grand_total": sum(counts.values())}


def _map_ci_widths(stored):
    result = {}
    for feature_name in FEATURES:
        widths = [row["balanced_accuracy_ci95"][1] -
                  row["balanced_accuracy_ci95"][0]
                  for row in stored["pairs"] if row["feature_set"] == feature_name]
        result[feature_name] = {"min": min(widths), "max": max(widths)}
    return result


def evaluate_revision(out_dir="runs/revision"):
    pair_rows, verdict_grid, bh_significant, stored = _identity_reanalysis()
    recency = _recency_test(stored)
    hard_subset = _hard_subset()
    bootstrap_variants = _bootstrap_checks()
    utility = _utility_table()
    tally = _tally()
    map_ci_widths = _map_ci_widths(stored)

    for feature_name in FEATURES:
        print(f"\nTOST verdict sensitivity: {feature_name}")
        print("delta threshold equivalent distinguishable uncertain")
        for row in verdict_grid[feature_name]:
            print(f"{row['delta']:<5.2f} {row['distinguishable_threshold']:<9.2f} "
                  f"{row['equivalent']:<10} {row['distinguishable']:<15} "
                  f"{row['uncertain']}")
        print(f"BH-significant at q=0.05: {bh_significant[feature_name]}/91")
    print("\nrecency enrichment Fisher table")
    print("                         category other")
    for label, row in zip(recency["rows"], recency["table"]):
        print(f"{label:<24} {row[0]:<8} {row[1]}")
    print(f"Fisher greater p={recency['p_value_greater']:.6g}")

    print("\n4-way hard subset (chance=0.25)")
    print("feature_set          classifier balanced_accuracy 95% CI")
    for row in hard_subset["results"]:
        low, high = row["balanced_accuracy_ci95"]
        print(f"{row['feature_set']:<20} {row['classifier']:<10} "
              f"{row['balanced_accuracy']:.3f}             [{low:.3f}, {high:.3f}]")

    print("\nbootstrap CI variants")
    print("headline                              score percentile          basic               studentized")
    for name, row in bootstrap_variants.items():
        cells = " ".join(f"[{row[key][0]:.3f}, {row[key][1]:.3f}]"
                         for key in ("percentile", "basic", "studentized"))
        print(f"{name:<37} {row['balanced_accuracy']:.3f} {cells}")

    print("\nroot utility (mean +/- sd; replicate=0)")
    print("population              " + " ".join(f"{label:>14}" for label in LABELS))
    for row in utility["rows"]:
        cells = " ".join(f"{row[label]['mean']:.3f}+/-{row[label]['sd']:.3f}"
                         for label in LABELS)
        print(f"{row['population']:<23} {cells}")

    print("\nmodel tally")
    for name, count in tally["counts"].items():
        print(f"{name:<20} {count}")
    print(f"{'grand_total':<20} {tally['grand_total']}")
    print("\nmap CI widths")
    for feature_name, row in map_ci_widths.items():
        print(f"{feature_name:<20} min={row['min']:.4f} max={row['max']:.4f}")

    payload = {
        "tost_pairs": pair_rows,
        "tost_verdict_sensitivity": verdict_grid,
        "bh_significant_q05": bh_significant,
        "recency_enrichment": recency,
        "hard_subset_4way": hard_subset,
        "bootstrap_variants": bootstrap_variants,
        "root_utility": utility,
        "model_tally": tally,
        "map_ci_widths": map_ci_widths,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results_revision.json", "w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    return payload


if __name__ == "__main__":
    evaluate_revision()
