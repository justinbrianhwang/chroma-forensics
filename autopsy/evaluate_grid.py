"""Evaluate SPEC5b generalization splits for the architecture/dataset grid."""
import json
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .evaluate import LABELS, _bootstrap_ci, _classifier, _load_manifest


PROBE_GROUPS = ("blackbox", "logit", "actgeom")
ARCHITECTURES = ("cnn", "resnet", "vgg", "depthwise", "seresnet")
OPERATIONS = ("null", "sft", "unlearn", "prune", "quant")


def _balanced_accuracy(y, prediction):
    return float(np.mean([
        np.mean(prediction[y == label] == label) for label in np.unique(y)
    ]))


def _root_folds(groups):
    splitter = GroupKFold(n_splits=min(8, len(np.unique(groups))))
    folds = list(splitter.split(groups, groups, groups))
    for train, test in folds:
        assert set(groups[train]).isdisjoint(groups[test])
    return folds


def _predict_oof(x, y, folds, classifier_name):
    prediction = np.empty_like(y)
    for train, test in folds:
        model = _classifier(classifier_name)
        model.fit(x[train], y[train])
        prediction[test] = model.predict(x[test])
    return prediction


def _result(feature_set, classifier, split, direction, y, prediction, groups,
            n_train=None, per_op_recall=None):
    row = {
        "feature_set": feature_set,
        "classifier": classifier,
        "split": split,
        "direction": direction,
        "n_train": int(n_train) if n_train is not None else int(len(y)),
        "n_test": int(len(y)),
        "balanced_accuracy": _balanced_accuracy(y, prediction),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "balanced_accuracy_ci95": _bootstrap_ci(
            y, prediction, groups, scorer=_balanced_accuracy),
    }
    if per_op_recall is not None:
        row["per_op_recall"] = per_op_recall
    return row


def _paired_sign_flip(y, prediction_a, prediction_b, groups, n_permutations=10000,
                      seed=0):
    differences = np.asarray([
        _balanced_accuracy(y[groups == group], prediction_a[groups == group]) -
        _balanced_accuracy(y[groups == group], prediction_b[groups == group])
        for group in np.unique(groups)
    ])
    observed = abs(differences.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_permutations, len(differences)))
    permuted = np.abs((signs * differences).mean(1))
    return float((1 + np.count_nonzero(permuted >= observed - 1e-15)) /
                 (n_permutations + 1))


def _holm(p_values):
    names = sorted(p_values, key=p_values.get)
    corrected = {}
    running = 0.0
    for rank, name in enumerate(names):
        running = max(running, (len(names) - rank) * p_values[name])
        corrected[name] = min(1.0, running)
    return corrected


def _ablation(archive, feature_names, suffix):
    arrays = []
    for group in PROBE_GROUPS:
        names = feature_names[group]
        selected = [index for index, name in enumerate(names)
                    if suffix == "both" or name.endswith(f"_{suffix}")]
        arrays.append(archive[group][:, selected])
    return np.concatenate(arrays, axis=1)


def _intensity_masks(manifest):
    labels = np.asarray([entry["label"] for entry in manifest])
    intensities = np.asarray([
        np.nan if entry["intensity"] is None else float(entry["intensity"])
        for entry in manifest
    ])
    unlearn = intensities[labels == "unlearn"]
    unlearn_q25, unlearn_q50, unlearn_q75 = np.quantile(unlearn, (0.25, 0.5, 0.75))
    strongest = (
        ((labels == "sft") & (intensities >= 0.925)) |
        ((labels == "unlearn") & (intensities >= unlearn_q75)) |
        ((labels == "prune") & (intensities >= 0.625)) |
        ((labels == "quant") & (intensities == 4)) |
        ((labels == "null") & (intensities == 3))
    )
    second = (
        ((labels == "sft") & (intensities >= 0.575) & (intensities < 0.75)) |
        ((labels == "unlearn") & (intensities >= unlearn_q25) &
         (intensities < unlearn_q50)) |
        ((labels == "prune") & (intensities >= 0.375) & (intensities < 0.5)) |
        ((labels == "quant") & (intensities == 5)) |
        ((labels == "null") & (intensities == 2))
    )
    definitions = {
        "strongest_quartile": {
            "sft": ">=0.925", "unlearn": f">={unlearn_q75:.12g}",
            "prune": ">=0.625", "quant": "bits=4", "null": "epochs=3",
        },
        "second_quartile": {
            "sft": "[0.575,0.75)",
            "unlearn": f"[{unlearn_q25:.12g},{unlearn_q50:.12g})",
            "prune": "[0.375,0.5)", "quant": "bits=5", "null": "epochs=2",
        },
    }
    return {"strongest_quartile": strongest, "second_quartile": second}, definitions


def _variance_decomposition(manifest, probe_base):
    cell = np.asarray([
        entry["dataset"] == "cifar10" and entry["arch"] == "cnn"
        for entry in manifest
    ])
    standardized = StandardScaler().fit_transform(probe_base[cell])
    n_components = min(10, *standardized.shape)
    scores = PCA(n_components=n_components).fit_transform(standardized)
    cell_manifest = [entry for entry, selected in zip(manifest, cell) if selected]
    shares = {}
    for operation in OPERATIONS:
        selected = np.asarray([entry["label"] == operation for entry in cell_manifest])
        operation_scores = scores[selected]
        operation_entries = [entry for entry in cell_manifest if entry["label"] == operation]
        roots = np.asarray([entry["root_group"] for entry in operation_entries])
        for root in np.unique(roots):
            replicates = sorted(entry["replicate"] for entry in operation_entries
                                if entry["root_group"] == root)
            assert replicates == [0, 1, 2]
        means = np.stack([operation_scores[roots == root].mean(0)
                          for root in np.unique(roots)])
        fitted = np.stack([operation_scores[roots == root].mean(0)
                           for root in roots])
        between = means.var(0)
        within = np.mean((operation_scores - fitted) ** 2, axis=0)
        shares[operation] = float(np.mean(between / np.maximum(between + within, 1e-30)))
    return shares


def _population_normalize(x, populations, variant):
    normalized = np.empty_like(x, dtype=float)
    for population in sorted(set(populations)):
        selected = np.asarray([value == population for value in populations])
        values = x[selected]
        if variant == "zscore":
            normalized[selected] = ((values - values.mean(0)) /
                                    np.maximum(values.std(0), 1e-9))
        else:
            normalized[selected] = ((rankdata(values, axis=0) - 1) /
                                    max(len(values) - 1, 1))
    return normalized


def _transductive_rows(feature_sets, y, groups, architectures, datasets,
                       normalization):
    rows = []
    family_rows = {}
    for held_arch in ARCHITECTURES:
        train = np.flatnonzero(architectures != held_arch)
        test = np.flatnonzero(architectures == held_arch)
        direction = f"holdout_{held_arch}"
        for feature_name, x in feature_sets.items():
            for classifier_name in ("logreg", "rf"):
                model = _classifier(classifier_name)
                model.fit(x[train], y[train])
                prediction = model.predict(x[test])
                row = _result(feature_name, classifier_name, "leave_family_out",
                              direction, y[test], prediction, groups[test], len(train))
                row["normalization"] = normalization
                rows.append(row)
                family_rows.setdefault((feature_name, classifier_name), []).append(row)
    for direction_rows in family_rows.values():
        mean_row = dict(direction_rows[0])
        mean_row.update(
            direction="mean",
            n_train=int(np.mean([row["n_train"] for row in direction_rows])),
            n_test=int(np.mean([row["n_test"] for row in direction_rows])),
            balanced_accuracy=float(np.mean([
                row["balanced_accuracy"] for row in direction_rows])),
            macro_f1=float(np.mean([row["macro_f1"] for row in direction_rows])),
            balanced_accuracy_ci95=[float(np.mean([
                row["balanced_accuracy_ci95"][bound] for row in direction_rows
            ])) for bound in (0, 1)],
        )
        rows.append(mean_row)

    for train_value, test_value in (("cifar10", "svhn"), ("svhn", "cifar10")):
        train = np.flatnonzero(datasets == train_value)
        test = np.flatnonzero(datasets == test_value)
        direction = f"{train_value}_to_{test_value}"
        for feature_name, x in feature_sets.items():
            for classifier_name in ("logreg", "rf"):
                model = _classifier(classifier_name)
                model.fit(x[train], y[train])
                prediction = model.predict(x[test])
                row = _result(feature_name, classifier_name, "leave_dataset_out",
                              direction, y[test], prediction, groups[test], len(train))
                row["normalization"] = normalization
                rows.append(row)
    return rows


def evaluate_transductive(out_dir="runs/grid"):
    out_dir = Path(out_dir)
    full_manifest = _load_manifest(out_dir)
    archive = np.load(out_dir / "fingerprints.npz")
    if archive["model_ids"].tolist() != [entry["model_id"] for entry in full_manifest]:
        raise ValueError("fingerprints do not match manifest order")

    results_path = out_dir / "results_grid.json"
    with open(results_path) as f:
        payload = json.load(f)
    raw_rows = payload["results"]
    replicates = np.asarray([entry.get("replicate", 0) for entry in full_manifest])
    classification_rows = np.flatnonzero(replicates == 0)
    manifest = [full_manifest[index] for index in classification_rows]
    y = np.asarray([LABELS.index(entry["label"]) for entry in manifest])
    groups = np.asarray([entry["root_group"] for entry in manifest])
    architectures = np.asarray([entry["arch"] for entry in manifest])
    datasets = np.asarray([entry["dataset"] for entry in manifest])
    populations = [(entry["arch"], entry["dataset"]) for entry in manifest]
    full_probe_base = np.concatenate([archive[group] for group in PROBE_GROUPS], axis=1)
    feature_sets = {
        "confound": archive["confound"][classification_rows],
        "probe_base_free": full_probe_base[classification_rows],
        "probe_base_free_tda": np.concatenate(
            (full_probe_base, archive["topology"]), axis=1)[classification_rows],
    }

    variant_rows = {}
    normalized_sets = {}
    for variant in ("zscore", "quantile"):
        normalized_sets[variant] = {
            name: _population_normalize(x, populations, variant)
            for name, x in feature_sets.items()
        }
        variant_rows[variant] = _transductive_rows(
            normalized_sets[variant], y, groups, architectures, datasets, variant)

    raw_lookup = {
        (row["split"], row["direction"], row["feature_set"], row["classifier"]): row
        for row in raw_rows
    }
    normalized_lookup = {
        variant: {
            (row["split"], row["direction"], row["feature_set"], row["classifier"]): row
            for row in rows
        } for variant, rows in variant_rows.items()
    }

    self_checks = []
    folds = _root_folds(groups)
    print("transductive zscore leave-root-out self-check")
    print("feature_set                  classifier unnormalized zscore difference")
    for feature_name, x in normalized_sets["zscore"].items():
        for classifier_name in ("logreg", "rf"):
            prediction = _predict_oof(x, y, folds, classifier_name)
            score = _balanced_accuracy(y, prediction)
            raw = raw_lookup[("leave_root_out", "pooled", feature_name, classifier_name)][
                "balanced_accuracy"]
            difference = score - raw
            assert abs(difference) <= 0.05, (
                f"zscore leave-root-out changed {feature_name}/{classifier_name} by "
                f"{difference:.6f}")
            row = {"feature_set": feature_name, "classifier": classifier_name,
                   "unnormalized": raw, "zscore": score, "difference": difference}
            self_checks.append(row)
            print(f"{feature_name:<28} {classifier_name:<10} {raw:.3f}        "
                  f"{score:.3f}  {difference:+.3f}")

    print("\ntransductive population-normalization comparison")
    print("split                 direction           feature_set                  classifier "
          "unnormalized zscore (95% CI)          quantile (95% CI)")
    comparisons = []
    keys = [key for key in normalized_lookup["zscore"]]
    for key in keys:
        split, direction, feature_name, classifier_name = key
        raw = raw_lookup[key]
        zscore = normalized_lookup["zscore"][key]
        quantile = normalized_lookup["quantile"][key]
        comparisons.append({
            "split": split, "direction": direction, "feature_set": feature_name,
            "classifier": classifier_name,
            "unnormalized_balanced_accuracy": raw["balanced_accuracy"],
            "zscore_balanced_accuracy": zscore["balanced_accuracy"],
            "zscore_balanced_accuracy_ci95": zscore["balanced_accuracy_ci95"],
            "quantile_balanced_accuracy": quantile["balanced_accuracy"],
            "quantile_balanced_accuracy_ci95": quantile["balanced_accuracy_ci95"],
        })
        zl, zh = zscore["balanced_accuracy_ci95"]
        ql, qh = quantile["balanced_accuracy_ci95"]
        print(f"{split:<21} {direction:<19} {feature_name:<28} {classifier_name:<10} "
              f"{raw['balanced_accuracy']:.3f}        {zscore['balanced_accuracy']:.3f} "
              f"[{zl:.3f}, {zh:.3f}]  {quantile['balanced_accuracy']:.3f} "
              f"[{ql:.3f}, {qh:.3f}]")

    payload["transductive"] = {
        "population": "architecture_x_dataset",
        "statistics_include_replicates": False,
        "variants": variant_rows,
        "leave_root_out_zscore_self_check": self_checks,
        "comparisons": comparisons,
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    return payload["transductive"]


def evaluate_grid(out_dir="runs/grid"):
    out_dir = Path(out_dir)
    full_manifest = _load_manifest(out_dir)
    archive = np.load(out_dir / "fingerprints.npz")
    full_model_ids = [entry["model_id"] for entry in full_manifest]
    if archive["model_ids"].tolist() != full_model_ids:
        raise ValueError("fingerprints do not match manifest order")
    with open(out_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    full_probe_base = np.concatenate([archive[group] for group in PROBE_GROUPS], axis=1)
    variance_shares = _variance_decomposition(full_manifest, full_probe_base)
    replicates = np.asarray([entry.get("replicate", 0) for entry in full_manifest])
    classification_rows = np.flatnonzero(replicates == 0)
    assert not np.any(replicates[classification_rows]), \
        "operation-seed replicates entered classification rows"
    manifest = [full_manifest[index] for index in classification_rows]
    assert all(entry.get("replicate", 0) == 0 for entry in manifest), \
        "operation-seed replicates entered classification splits"
    model_ids = [entry["model_id"] for entry in manifest]
    y = np.asarray([LABELS.index(entry["label"]) for entry in manifest])
    groups = np.asarray([entry["root_group"] for entry in manifest])
    architectures = np.asarray([entry["arch"] for entry in manifest])
    datasets = np.asarray([entry["dataset"] for entry in manifest])
    probe_base = full_probe_base[classification_rows]
    feature_sets = {
        "confound": archive["confound"][classification_rows],
        "probe_base_free": probe_base,
        "probe_base_free_tda": np.concatenate(
            (probe_base, archive["topology"][classification_rows]), axis=1),
    }
    results = []
    endpoint_predictions = {}

    # 1. Pooled leave-root-out.
    root_folds = _root_folds(groups)
    for feature_name, x in feature_sets.items():
        for classifier_name in ("logreg", "rf"):
            prediction = _predict_oof(x, y, root_folds, classifier_name)
            results.append(_result(feature_name, classifier_name, "leave_root_out", "pooled",
                                   y, prediction, groups))
            if classifier_name == "logreg":
                endpoint_predictions.setdefault("leave_root_out", {})[feature_name] = prediction

    # Weight descriptors are valid only inside one architecture.
    full_architectures = np.asarray([entry["arch"] for entry in full_manifest])
    for arch in ARCHITECTURES:
        selected = architectures == arch
        expected_ids = np.asarray(model_ids)[selected].tolist()
        full_arch_rows = np.flatnonzero(full_architectures == arch)
        if archive[f"model_ids.{arch}"].tolist() != np.asarray(full_model_ids)[full_arch_rows].tolist():
            raise ValueError(f"{arch} weight rows do not match manifest")
        weights = archive[f"weights.{arch}"][replicates[full_arch_rows] == 0]
        if len(weights) != len(expected_ids):
            raise ValueError(f"{arch} classification weight rows do not match manifest")
        x = np.concatenate((probe_base[selected], weights), axis=1)
        arch_y, arch_groups = y[selected], groups[selected]
        folds = _root_folds(arch_groups)
        for classifier_name in ("logreg", "rf"):
            prediction = _predict_oof(x, arch_y, folds, classifier_name)
            results.append(_result("all_base_free", classifier_name, "leave_root_out",
                                   f"within_{arch}", arch_y, prediction, arch_groups))

    # 2. Hold out each of the five architecture families across both datasets.
    endpoint_predictions["leave_family_out"] = {
        name: np.empty_like(y) for name in ("confound", "probe_base_free")
    }
    family_rows = {}
    for held_arch in ARCHITECTURES:
        train = np.flatnonzero(architectures != held_arch)
        test = np.flatnonzero(architectures == held_arch)
        assert set(architectures[train]).isdisjoint(architectures[test])
        assert set(groups[train]).isdisjoint(groups[test])
        train_populations = {(datasets[index], architectures[index]) for index in train}
        test_populations = {(datasets[index], architectures[index]) for index in test}
        assert train_populations.isdisjoint(test_populations)
        direction = f"holdout_{held_arch}"
        for feature_name, x in feature_sets.items():
            for classifier_name in ("logreg", "rf"):
                model = _classifier(classifier_name)
                model.fit(x[train], y[train])
                prediction = model.predict(x[test])
                row = _result(feature_name, classifier_name, "leave_family_out",
                              direction, y[test], prediction, groups[test], len(train))
                results.append(row)
                family_rows.setdefault((feature_name, classifier_name), []).append(row)
                if classifier_name == "logreg" and feature_name in endpoint_predictions["leave_family_out"]:
                    endpoint_predictions["leave_family_out"][feature_name][test] = prediction
    for rows in family_rows.values():
        mean_row = dict(rows[0])
        mean_row.update(
            direction="mean",
            n_train=int(np.mean([row["n_train"] for row in rows])),
            n_test=int(np.mean([row["n_test"] for row in rows])),
            balanced_accuracy=float(np.mean([row["balanced_accuracy"] for row in rows])),
            macro_f1=float(np.mean([row["macro_f1"] for row in rows])),
            balanced_accuracy_ci95=[float(np.mean([
                row["balanced_accuracy_ci95"][bound] for row in rows
            ])) for bound in (0, 1)],
        )
        results.append(mean_row)

    # 3. Hold out each dataset across every architecture.
    endpoint_predictions["leave_dataset_out"] = {
        name: np.empty_like(y) for name in ("confound", "probe_base_free")
    }
    for train_value, test_value in (("cifar10", "svhn"), ("svhn", "cifar10")):
        train = np.flatnonzero(datasets == train_value)
        test = np.flatnonzero(datasets == test_value)
        assert set(datasets[train]).isdisjoint(datasets[test])
        assert set(groups[train]).isdisjoint(groups[test])
        train_populations = {(datasets[index], architectures[index]) for index in train}
        test_populations = {(datasets[index], architectures[index]) for index in test}
        assert train_populations.isdisjoint(test_populations)
        direction = f"{train_value}_to_{test_value}"
        for feature_name, x in feature_sets.items():
            for classifier_name in ("logreg", "rf"):
                model = _classifier(classifier_name)
                model.fit(x[train], y[train])
                prediction = model.predict(x[test])
                results.append(_result(feature_name, classifier_name, "leave_dataset_out",
                                       direction, y[test], prediction, groups[test], len(train)))
                if classifier_name == "logreg" and feature_name in endpoint_predictions["leave_dataset_out"]:
                    endpoint_predictions["leave_dataset_out"][feature_name][test] = prediction

    # 4. Strongest-quartile extrapolation and second-quartile interpolation.
    fold_roots = [set(groups[test]) for _, test in root_folds]
    intensity_masks, intensity_definitions = _intensity_masks(manifest)
    for variant, held in intensity_masks.items():
        def held_coverage(pair):
            roots = fold_roots[pair[0]] | fold_roots[pair[1]]
            return int(np.count_nonzero(held & np.isin(groups, list(roots))))

        test_fold_pair = max(itertools.combinations(range(len(root_folds)), 2),
                             key=held_coverage)
        test_roots = fold_roots[test_fold_pair[0]] | fold_roots[test_fold_pair[1]]
        is_test_root = np.isin(groups, list(test_roots))
        train = np.flatnonzero(~is_test_root & ~held)
        test = np.flatnonzero(is_test_root & held)
        assert len(test), f"no test models for {variant}"
        assert set(groups[train]).isdisjoint(groups[test])
        for feature_name, x in feature_sets.items():
            for classifier_name in ("logreg", "rf"):
                model = _classifier(classifier_name)
                model.fit(x[train], y[train])
                prediction = model.predict(x[test])
                per_op = {}
                for operation in OPERATIONS:
                    selected = np.asarray([manifest[index]["label"] == operation
                                           for index in test])
                    per_op[operation] = (float(np.mean(prediction[selected] == y[test][selected]))
                                         if selected.any() else None)
                results.append(_result(feature_name, classifier_name, "leave_intensity_out",
                                       variant, y[test], prediction, groups[test],
                                       len(train), per_op))

    # 5. Probe ablation by existing feature-name suffixes.
    for suffix in ("id", "ood", "both"):
        x = _ablation(archive, feature_names, suffix)[classification_rows]
        for classifier_name in ("logreg", "rf"):
            prediction = _predict_oof(x, y, root_folds, classifier_name)
            results.append(_result(f"probe_base_free_{suffix}", classifier_name,
                                   "probe_ablation", suffix, y, prediction, groups))

    self_check_p = _paired_sign_flip(
        y, endpoint_predictions["leave_root_out"]["probe_base_free"],
        endpoint_predictions["leave_root_out"]["probe_base_free"], groups)
    assert self_check_p > 0.5
    raw_p = {
        split: _paired_sign_flip(
            y, predictions["probe_base_free"], predictions["confound"], groups)
        for split, predictions in endpoint_predictions.items()
    }
    corrected_p = _holm(raw_p)

    print("feature_set                  classifier split                 direction           bal_acc macro_f1 95% CI")
    for row in results:
        low, high = row["balanced_accuracy_ci95"]
        print(f"{row['feature_set']:<28} {row['classifier']:<10} {row['split']:<21} "
              f"{row['direction']:<19} {row['balanced_accuracy']:.3f}   "
              f"{row['macro_f1']:.3f}    [{low:.3f}, {high:.3f}]")
        if "per_op_recall" in row:
            print("  held-intensity recall:", row["per_op_recall"])
    print("paired root-group sign-flip tests (probe_base_free vs confound, logreg)")
    for split in raw_p:
        print(f"  {split}: raw={raw_p[split]:.6f} holm={corrected_p[split]:.6f}")
    print(f"sign-flip self-check p={self_check_p:.6f}")
    print("cifar10/cnn probe_base_free top-10-PC variance decomposition")
    print("operation  root-variance share")
    for operation in OPERATIONS:
        print(f"{operation:<10} {variance_shares[operation]:.6f}")

    payload = {
        "chance": 1 / len(LABELS),
        "labels": list(LABELS),
        "n_classification_models": len(manifest),
        "n_replicate_models": int(np.count_nonzero(replicates)),
        "results": results,
        "intensity_holdouts": intensity_definitions,
        "root_variance_share": variance_shares,
        "sign_flip": {
            split: {"raw_p": raw_p[split], "holm_p": corrected_p[split]}
            for split in raw_p
        },
        "sign_flip_self_check_p": self_check_p,
    }
    with open(out_dir / "results_grid.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload


if __name__ == "__main__":
    evaluate_grid()
