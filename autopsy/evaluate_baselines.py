"""Evaluate prior-work baselines under the repository's root-held-out protocol."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baseline_features import PAIR_LABELS, _manifest
from .evaluate import COMBOS, LABELS, _bootstrap_ci, _classifier


TASK_PRIMARY = "primary_6class"
TASK_PAIR = "hardest_order_pair"
TASK_FT = "ft_laundering"


def _stored_features(path, keep=lambda entry: True):
    path = Path(path)
    manifest = _manifest(path)
    archive = np.load(path / "fingerprints.npz")
    if archive["model_ids"].tolist() != [entry["model_id"] for entry in manifest]:
        raise ValueError(f"fingerprints do not match manifest order: {path}")
    selected = np.asarray([keep(entry) for entry in manifest])
    all_base_free = np.concatenate([archive[name] for name in COMBOS["all_base_free"]], axis=1)
    entries = [entry for entry, retain in zip(manifest, selected) if retain]
    return entries, all_base_free[selected], archive["delta_ref"][selected]


def _folds(entries):
    groups = np.asarray([entry["root_group"] for entry in entries])
    splitter = GroupKFold(n_splits=min(8, len(np.unique(groups))))
    result = []
    for train, test in splitter.split(groups, groups, groups):
        assert set(groups[train]).isdisjoint(groups[test])
        result.append((train, test))
    return result


def _primary_ft_folds(primary_entries, ft_entries):
    ft_groups = np.asarray([entry["root_group"] for entry in ft_entries])
    source_groups = np.asarray([entry["root_group"] for entry in primary_entries])
    result = []
    for train, test in _folds(primary_entries):
        test_groups = set(source_groups[test])
        ft_test = np.flatnonzero(np.isin(ft_groups, list(test_groups)))
        assert set(source_groups[train]).isdisjoint(ft_groups[ft_test])
        result.append((train, test, ft_test))
    assert sorted(np.concatenate([fold[2] for fold in result]).tolist()) == list(range(len(ft_entries)))
    return result


def _profile_logreg():
    return make_pipeline(StandardScaler(with_mean=False), LogisticRegression(
        max_iter=3000, C=1.0, random_state=0))


def _nearest_centroid():
    return make_pipeline(StandardScaler(), NearestCentroid())


def _balanced_accuracy(y, prediction):
    return float(np.mean([
        np.mean(prediction[y == label] == label) for label in np.unique(y)
    ]))


def _fit_primary_ft(source_x, ft_x, source_y, folds, factory):
    source_prediction = np.full(len(source_y), -1, dtype=int)
    ft_prediction = np.full(len(ft_x), -1, dtype=int)
    for train, test, ft_test in folds:
        model = factory()
        model.fit(source_x[train], source_y[train])
        source_prediction[test] = model.predict(source_x[test])
        ft_prediction[ft_test] = model.predict(ft_x[ft_test])
    assert (source_prediction >= 0).all() and (ft_prediction >= 0).all()
    return source_prediction, ft_prediction


def _fit_pair(x, y, folds, factory):
    prediction = np.full(len(y), -1, dtype=int)
    for train, test in folds:
        model = factory()
        model.fit(x[train], y[train])
        prediction[test] = model.predict(x[test])
    assert (prediction >= 0).all()
    return prediction


def _dual_pca(gram, train, tests, requested_components=128):
    """Return exact PCA scores using only train/train and test/train dot products."""
    assert all(set(train).isdisjoint(test) for test in tests)
    n_components = min(requested_components, len(train) - 1)
    if len(train) > requested_components:
        assert n_components == requested_components
    train_gram = gram[np.ix_(train, train)].astype(np.float64)
    train_mean = train_gram.mean(0)
    grand_mean = train_mean.mean()
    centered = train_gram - train_mean[:, None] - train_mean[None, :] + grand_mean
    centered = (centered + centered.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    floor = max(eigenvalues[0] * 1e-12, 1e-12)
    scales = np.sqrt(np.maximum(eigenvalues, floor))
    train_scores = eigenvectors * scales
    test_scores = []
    for test in tests:
        cross = gram[np.ix_(test, train)].astype(np.float64)
        cross -= cross.mean(1, keepdims=True)
        cross -= train_mean[None, :]
        cross += grand_mean
        test_scores.append(cross @ eigenvectors / scales)
    return train_scores, test_scores


def _fit_pca_primary_ft(gram, source_indices, ft_indices, source_y, folds, classifiers):
    predictions = {name: (np.full(len(source_y), -1, dtype=int),
                          np.full(len(ft_indices), -1, dtype=int)) for name in classifiers}
    for train, test, ft_test in folds:
        train_global = source_indices[train]
        test_global = source_indices[test]
        ft_global = ft_indices[ft_test]
        train_x, (test_x, ft_x) = _dual_pca(
            gram, train_global, (test_global, ft_global))
        for name in classifiers:
            model = _classifier(name)
            model.fit(train_x, source_y[train])
            predictions[name][0][test] = model.predict(test_x)
            predictions[name][1][ft_test] = model.predict(ft_x)
    assert all((source >= 0).all() and (ft >= 0).all()
               for source, ft in predictions.values())
    return predictions


def _fit_pca_pair(gram, indices, y, folds, classifiers):
    predictions = {name: np.full(len(y), -1, dtype=int) for name in classifiers}
    for train, test in folds:
        train_x, (test_x,) = _dual_pca(gram, indices[train], (indices[test],))
        for name in classifiers:
            model = _classifier(name)
            model.fit(train_x, y[train])
            predictions[name][test] = model.predict(test_x)
    assert all((prediction >= 0).all() for prediction in predictions.values())
    return predictions


@torch.no_grad()
def _cka_matrix(act_sub):
    activations = torch.from_numpy(act_sub).cuda().float()
    activations -= activations.mean(1, keepdim=True)
    grams = torch.bmm(activations, activations.transpose(1, 2)).flatten(1)
    grams = F.normalize(grams, dim=1, eps=1e-30)
    similarities = (grams @ grams.T).clamp(-1, 1)
    assert similarities.diagonal().min().item() > 0.999
    return similarities.cpu().numpy()


def _reef(similarities, train_indices, test_indices, y_train):
    values = similarities[np.ix_(test_indices, train_indices)]
    one_nn = y_train[values.argmax(1)]
    classes = np.unique(y_train)
    class_scores = np.column_stack([values[:, y_train == label].mean(1) for label in classes])
    class_mean = classes[class_scores.argmax(1)]
    return one_nn, class_mean


def _reef_primary_ft(similarities, source_indices, ft_indices, source_y, folds):
    result = {name: (np.full(len(source_y), -1, dtype=int),
                     np.full(len(ft_indices), -1, dtype=int))
              for name in ("one_nn", "class_mean")}
    for train, test, ft_test in folds:
        source_predictions = _reef(similarities, source_indices[train],
                                   source_indices[test], source_y[train])
        ft_predictions = _reef(similarities, source_indices[train],
                               ft_indices[ft_test], source_y[train])
        for index, name in enumerate(result):
            result[name][0][test] = source_predictions[index]
            result[name][1][ft_test] = ft_predictions[index]
    return result


def _reef_pair(similarities, indices, y, folds):
    result = {name: np.full(len(y), -1, dtype=int) for name in ("one_nn", "class_mean")}
    for train, test in folds:
        fold_predictions = _reef(similarities, indices[train], indices[test], y[train])
        for index, name in enumerate(result):
            result[name][test] = fold_predictions[index]
    return result


def _result(task, method, y, prediction, groups, reference_aware=False, **extra):
    score = _balanced_accuracy(y, prediction)
    return {"task": task, "method": method, "score": score,
            "balanced_accuracy": score,
            "ci95": _bootstrap_ci(y, prediction, groups, scorer=_balanced_accuracy),
            "reference_aware": reference_aware, **extra}


def evaluate_baselines(out_dir="runs/baselines"):
    out_dir = Path(out_dir)
    raw_path = out_dir / "raw_features.npz"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path}; run baseline extraction first")

    primary_entries, primary_base, primary_delta_ref = _stored_features(
        "runs/pilot_matched")
    pair_entries, pair_base, _ = _stored_features(
        "runs/pairs", lambda entry: entry["label"] in PAIR_LABELS)
    ft_entries, ft_base, ft_delta_ref = _stored_features(
        "runs/laundered", lambda entry: entry.get("laundry") == "ft")

    archive = np.load(raw_path)
    run_names, model_ids = archive["run_names"], archive["model_ids"]
    primary_indices = np.flatnonzero(run_names == "pilot_matched")
    pair_indices = np.flatnonzero(run_names == "pairs")
    ft_indices = np.flatnonzero(run_names == "laundered_ft")
    for indices, entries in ((primary_indices, primary_entries), (pair_indices, pair_entries),
                             (ft_indices, ft_entries)):
        if model_ids[indices].tolist() != [entry["model_id"] for entry in entries]:
            raise ValueError("raw baseline features do not match manifest order")

    primary_y = np.asarray([LABELS.index(entry["label"]) for entry in primary_entries])
    pair_y = np.asarray([PAIR_LABELS.index(entry["label"]) for entry in pair_entries])
    ft_y = np.asarray([LABELS.index(entry["label"]) for entry in ft_entries])
    primary_groups = np.asarray([entry["root_group"] for entry in primary_entries])
    pair_groups = np.asarray([entry["root_group"] for entry in pair_entries])
    ft_groups = np.asarray([entry["root_group"] for entry in ft_entries])
    primary_folds = _primary_ft_folds(primary_entries, ft_entries)
    pair_folds = _folds(pair_entries)

    profile = archive["probe_profile"].astype(np.float32)
    act_means = archive["act_means"].astype(np.float32)
    weight_gram = archive["raw_weight_gram"]
    profile_tensor = torch.from_numpy(profile).cuda()
    profile_tensor -= profile_tensor[0].clone()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    profile_gram = (profile_tensor @ profile_tensor.T).cpu().numpy()
    torch.backends.cuda.matmul.allow_tf32 = old_tf32
    del profile_tensor
    similarities = _cka_matrix(archive["act_sub"])

    roots = {entry["root_group"]: act_means[index]
             for index, entry in zip(primary_indices, primary_entries)
             if entry["label"] == "root"}
    if len(roots) != len(np.unique(primary_groups)):
        raise ValueError("each primary root group must have one root activation row")
    delta_act = np.asarray([act_means[index] - roots[entry["root_group"]]
                            for index, entry in enumerate(
                                primary_entries + pair_entries + ft_entries)])
    # The concatenated manifests and raw archive share this exact population order.
    assert np.array_equal(np.concatenate((primary_indices, pair_indices, ft_indices)),
                          np.arange(len(model_ids)))
    primary_delta_act = delta_act[:len(primary_entries)]
    pair_delta_act = delta_act[len(primary_entries):len(primary_entries) + len(pair_entries)]
    ft_delta_act = delta_act[-len(ft_entries):]

    predictions = {TASK_PRIMARY: {}, TASK_PAIR: {}, TASK_FT: {}}

    reef_main = _reef_primary_ft(similarities, primary_indices, ft_indices,
                                 primary_y, primary_folds)
    reef_pair = _reef_pair(similarities, pair_indices, pair_y, pair_folds)

    weight_main = _fit_pca_primary_ft(weight_gram, primary_indices, ft_indices,
                                      primary_y, primary_folds, ("logreg", "rf"))
    weight_pair = _fit_pca_pair(weight_gram, pair_indices, pair_y, pair_folds,
                                ("logreg", "rf"))
    profile_main = _fit_pca_primary_ft(profile_gram, primary_indices, ft_indices,
                                       primary_y, primary_folds, ("rf",))
    profile_pair = _fit_pca_pair(profile_gram, pair_indices, pair_y, pair_folds, ("rf",))

    for name in ("logreg", "rf"):
        predictions[TASK_PRIMARY][f"raw_weights_pca_{name}"] = weight_main[name][0]
        predictions[TASK_FT][f"raw_weights_pca_{name}"] = weight_main[name][1]
        predictions[TASK_PAIR][f"raw_weights_pca_{name}"] = weight_pair[name]
    predictions[TASK_PRIMARY]["raw_probe_profile_pca_rf"] = profile_main["rf"][0]
    predictions[TASK_FT]["raw_probe_profile_pca_rf"] = profile_main["rf"][1]
    predictions[TASK_PAIR]["raw_probe_profile_pca_rf"] = profile_pair["rf"]

    source_prediction, ft_prediction = _fit_primary_ft(
        profile[primary_indices], profile[ft_indices], primary_y, primary_folds,
        _profile_logreg)
    predictions[TASK_PRIMARY]["raw_probe_profile_logreg"] = source_prediction
    predictions[TASK_FT]["raw_probe_profile_logreg"] = ft_prediction
    predictions[TASK_PAIR]["raw_probe_profile_logreg"] = _fit_pair(
        profile[pair_indices], pair_y, pair_folds, _profile_logreg)

    feature_methods = (
        ("delta_act_full", primary_delta_act, pair_delta_act, ft_delta_act, True, ("logreg", "rf")),
        ("all_base_free", primary_base, pair_base, ft_base, False, ("logreg", "rf")),
    )
    for method, source_x, pair_x, target_x, aware, classifiers in feature_methods:
        for classifier in classifiers:
            primary_prediction, ft_prediction = _fit_primary_ft(
                source_x, target_x, primary_y, primary_folds,
                lambda classifier=classifier: _classifier(classifier))
            name = f"{method}_{classifier}"
            predictions[TASK_PRIMARY][name] = primary_prediction
            predictions[TASK_FT][name] = ft_prediction
            predictions[TASK_PAIR][name] = _fit_pair(
                pair_x, pair_y, pair_folds,
                lambda classifier=classifier: _classifier(classifier))

    primary_prediction, ft_prediction = _fit_primary_ft(
        primary_delta_ref, ft_delta_ref, primary_y, primary_folds,
        lambda: _classifier("logreg"))
    predictions[TASK_PRIMARY]["delta_ref_logreg"] = primary_prediction
    predictions[TASK_FT]["delta_ref_logreg"] = ft_prediction
    primary_prediction, ft_prediction = _fit_primary_ft(
        primary_base, ft_base, primary_y, primary_folds, _nearest_centroid)
    predictions[TASK_PRIMARY]["nearest_centroid"] = primary_prediction
    predictions[TASK_FT]["nearest_centroid"] = ft_prediction

    task_data = {
        TASK_PRIMARY: (primary_y, primary_groups),
        TASK_PAIR: (pair_y, pair_groups),
        TASK_FT: (ft_y, ft_groups),
    }
    reef_variants = {}
    reef_sets = {TASK_PRIMARY: {name: value[0] for name, value in reef_main.items()},
                 TASK_FT: {name: value[1] for name, value in reef_main.items()},
                 TASK_PAIR: reef_pair}
    for task, variants in reef_sets.items():
        y, groups = task_data[task]
        scored = {name: _result(task, f"reef_cka_{name}", y, prediction, groups)
                  for name, prediction in variants.items()}
        selected = max(scored, key=lambda name: scored[name]["score"])
        reef_variants[task] = {**scored, "selected": selected}
        predictions[task]["reef_cka"] = variants[selected]

    primary_order = ("reef_cka", "raw_weights_pca_logreg", "raw_weights_pca_rf",
                     "raw_probe_profile_logreg", "raw_probe_profile_pca_rf",
                     "delta_act_full_logreg", "delta_act_full_rf",
                     "all_base_free_logreg", "all_base_free_rf", "delta_ref_logreg",
                     "nearest_centroid")
    pair_order = primary_order[:9]
    results = []
    for task, order in ((TASK_PRIMARY, primary_order), (TASK_PAIR, pair_order),
                        (TASK_FT, primary_order)):
        y, groups = task_data[task]
        for method in order:
            extra = ({"selected_variant": reef_variants[task]["selected"]}
                     if method == "reef_cka" else {})
            results.append(_result(task, method, y, predictions[task][method], groups,
                                   method.startswith(("delta_act_full", "delta_ref")), **extra))

    print("method                         task                score 95% CI          reference_aware")
    for row in results:
        low, high = row["ci95"]
        print(f"{row['method']:<30} {row['task']:<19} {row['score']:.3f} "
              f"[{low:.3f}, {high:.3f}]  {str(row['reference_aware']).lower()}")

    payload = {"metric": "balanced_accuracy", "results": results,
               "reef_cka_variants": reef_variants,
               "populations": {TASK_PRIMARY: len(primary_entries), TASK_PAIR: len(pair_entries),
                               TASK_FT: len(ft_entries)}}
    with open(out_dir / "results_baselines.json", "w") as file:
        json.dump(payload, file, indent=2)
    return payload


if __name__ == "__main__":
    evaluate_baselines()
