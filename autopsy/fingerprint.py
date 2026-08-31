"""Extract fixed-length fingerprints from the pilot model zoo."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ripser import ripser
from scipy.spatial.distance import pdist

from . import zoo


QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
PAIR_QUANTILES = (0.1, 0.5, 0.9)
STAGES = ("a1", "a2", "a3", "z")


def _load_manifest(out_dir):
    with open(Path(out_dir) / "manifest.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def _forward(model, x):
    logits, flipped, activations = [], [], [[] for _ in STAGES]
    for start in range(0, len(x), 1000):
        xb = x[start:start + 1000]
        with torch.autocast("cuda", torch.bfloat16):
            out, acts = model(xb, acts=True)
            flip_out = model(xb.flip(-1))
        logits.append(out.float())
        flipped.append(flip_out.float())
        for values, act in zip(activations, acts):
            values.append(act.float())
    return torch.cat(logits), torch.cat(flipped), [torch.cat(x) for x in activations]


def _entropy(probabilities):
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)


def _confound(entry, outputs):
    values = [float(entry["eval_acc"])]
    names = ["eval_acc"]
    for suffix, (logits, _, _) in outputs.items():
        probabilities = logits.softmax(1)
        values.extend((probabilities.max(1).values.mean().item(),
                       _entropy(probabilities).mean().item()))
        names.extend((f"mean_max_softmax_prob_{suffix}",
                      f"mean_predictive_entropy_{suffix}"))
    return values, names


def _blackbox(outputs):
    values, names = [], []
    for suffix, (logits, flipped, _) in outputs.items():
        predictions = logits.argmax(1)
        histogram = torch.bincount(predictions, minlength=10).float() / len(predictions)
        histogram_entropy = -(histogram * histogram.clamp_min(1e-12).log()).sum()
        values.extend(histogram.tolist())
        values.extend((histogram_entropy.item(),
                       (predictions != flipped.argmax(1)).float().mean().item()))
        names.extend([f"predicted_class_{i}_{suffix}" for i in range(10)])
        names.extend((f"predicted_class_entropy_{suffix}", f"flip_rate_{suffix}"))
    return values, names


def _ece(probabilities, labels, n_bins=15):
    confidence, predictions = probabilities.max(1)
    bins = torch.clamp((confidence * n_bins).long(), max=n_bins - 1)
    result = confidence.new_zeros(())
    for index in range(n_bins):
        mask = bins == index
        if mask.any():
            result += mask.float().mean() * (
                confidence[mask].mean() - (predictions[mask] == labels[mask]).float().mean()
            ).abs()
    return result.item()


def _logit(outputs, probe_id_y):
    values, names = [], []
    for suffix, (logits, _, _) in outputs.items():
        probabilities = logits.softmax(1)
        entropy = _entropy(probabilities)
        top_two = probabilities.topk(2, dim=1).values
        statistics = (("predictive_entropy", entropy),
                      ("top1_top2_margin", top_two[:, 0] - top_two[:, 1]),
                      ("max_logit", logits.max(1).values))
        for name, statistic in statistics:
            values.extend(torch.quantile(statistic, statistic.new_tensor(QUANTILES)).tolist())
            names.extend(f"{name}_q{q:g}_{suffix}" for q in QUANTILES)
        values.extend(probabilities.sort(1, descending=True).values.mean(0).tolist())
        names.extend(f"mean_sorted_softmax_{rank}_{suffix}" for rank in range(1, 11))
        if suffix == "id":
            values.append(_ece(probabilities, probe_id_y))
            names.append("ece15_id")
    return values, names


def _spectrum_features(activation):
    centered = activation - activation.mean(0, keepdim=True)
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered / (len(centered) - 1)).clamp_min(0)
    total = eigenvalues.sum()
    if total > 0:
        normalized = eigenvalues / total
        nonzero = normalized > 0
        effective_rank = torch.exp(-(normalized[nonzero] * normalized[nonzero].log()).sum())
        participation_ratio = total.square() / eigenvalues.square().sum().clamp_min(1e-30)
    else:
        effective_rank = participation_ratio = total

    singular_values = torch.sqrt(eigenvalues.flip(0) * (len(centered) - 1)).clamp_min(1e-30)
    singular_values = singular_values[:min(32, len(singular_values))]
    ranks = torch.arange(1, len(singular_values) + 1, device=activation.device,
                         dtype=activation.dtype).log()
    log_values = singular_values.log()
    slope = ((ranks - ranks.mean()) * (log_values - log_values.mean())).sum()
    slope /= (ranks - ranks.mean()).square().sum().clamp_min(1e-30)
    return effective_rank.item(), participation_ratio.item(), slope.item()


def _geometry_features(activation, sample_indices, triangle):
    effective_rank, participation_ratio, slope = _spectrum_features(activation)
    sample = activation[sample_indices]
    distances = torch.cdist(sample, sample)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.topk(2, largest=False).values
    valid = (nearest[:, 0] > 1e-12) & (nearest[:, 1] > nearest[:, 0])
    if valid.any():
        twonn = 1.0 / (nearest[valid, 1] / nearest[valid, 0]).log().mean().clamp_min(1e-12)
    else:
        twonn = distances.new_zeros(())

    pairwise_distances = distances[triangle]
    normalized = F.normalize(sample, dim=1, eps=1e-12)
    pairwise_cosines = (normalized @ normalized.T)[triangle]
    quantiles = activation.new_tensor(PAIR_QUANTILES)
    values = [effective_rank, participation_ratio, slope, twonn.item()]
    values.extend(torch.quantile(pairwise_distances, quantiles).tolist())
    values.extend(torch.quantile(pairwise_cosines, quantiles).tolist())
    return values


def _linear_cka(x, y):
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    cross = x.T @ y
    numerator = cross.square().sum()
    denominator = torch.sqrt((x.T @ x).square().sum() * (y.T @ y).square().sum())
    return (numerator / denominator.clamp_min(1e-30)).item()


def _actgeom(outputs, sample_indices, triangle):
    values, names = [], []
    metric_names = ("effective_rank", "participation_ratio", "loglog_slope", "twonn")
    for suffix, (_, _, activations) in outputs.items():
        for stage, activation in zip(STAGES, activations):
            values.extend(_geometry_features(activation, sample_indices, triangle))
            names.extend(f"{stage}_{metric}_{suffix}" for metric in metric_names)
            names.extend(f"{stage}_euclidean_q{q:g}_{suffix}" for q in PAIR_QUANTILES)
            names.extend(f"{stage}_cosine_q{q:g}_{suffix}" for q in PAIR_QUANTILES)
        for left in range(len(STAGES)):
            for right in range(left + 1, len(STAGES)):
                values.append(_linear_cka(activations[left], activations[right]))
                names.append(f"cka_{STAGES[left]}_{STAGES[right]}_{suffix}")
    return values, names


def _persistence_entropy(lifetimes):
    total = lifetimes.sum()
    if total <= 0:
        return 0.0
    probabilities = lifetimes / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities)).sum())


def _topology(outputs, sample_indices):
    values, names = [], []
    h0_names = ("h0_death_q0.25", "h0_death_q0.5", "h0_death_q0.75",
                "h0_death_q0.9", "h0_total_persistence", "h0_persistence_entropy")
    h1_names = ("h1_n_bars", *(f"h1_life_top{i}" for i in range(1, 6)),
                "h1_total_persistence", "h1_persistence_entropy", "h1_max_birth")
    for suffix, (_, _, activations) in outputs.items():
        for stage, activation in zip(STAGES, activations):
            points = activation[sample_indices].cpu().numpy().astype(np.float32, copy=False)
            scale = np.median(pdist(points, metric="euclidean"))
            points /= scale if scale > 0 else 1.0
            h0, h1 = ripser(points, maxdim=1)["dgms"]

            deaths = h0[np.isfinite(h0[:, 1]), 1]
            lifetimes = h1[:, 1] - h1[:, 0]
            top_lifetimes = np.zeros(5, dtype=np.float32)
            top_lifetimes[:min(5, len(lifetimes))] = np.sort(lifetimes)[::-1][:5]
            values.extend(np.quantile(deaths, (0.25, 0.5, 0.75, 0.9)).tolist())
            values.extend((float(deaths.sum()), _persistence_entropy(deaths)))
            values.extend((float(len(h1)), *top_lifetimes.tolist(), float(lifetimes.sum()),
                           _persistence_entropy(lifetimes),
                           float(h1[:, 0].max()) if len(h1) else 0.0))
            names.extend(f"{stage}_{name}_{suffix}" for name in (*h0_names, *h1_names))
    return values, names


def _weight_tensors(state):
    return [(name, tensor) for name, tensor in state.items() if tensor.dim() > 1]


def _weights(state):
    values, names = [], []
    statistic_names = ("mean", "std", "abs_q0.5", "abs_q0.9", "abs_q0.99",
                       "zero_fraction", "l2_norm", "spectral_norm", "stable_rank",
                       "singular_1", "singular_2", "singular_3", "singular_4", "singular_5")
    for tensor_name, tensor in _weight_tensors(state):
        matrix = tensor.flatten(1).cuda()
        singular = torch.linalg.svdvals(matrix)
        top = torch.zeros(5, device=matrix.device)
        top[:min(5, len(singular))] = singular[:5]
        spectral = singular[0]
        flat = matrix.flatten()
        tensor_values = [flat.mean().item(), flat.std(unbiased=False).item()]
        tensor_values.extend(torch.quantile(flat.abs(), flat.new_tensor((0.5, 0.9, 0.99))).tolist())
        tensor_values.extend(((flat == 0).float().mean().item(), torch.linalg.vector_norm(flat).item(),
                              spectral.item(), (flat.square().sum() / spectral.square().clamp_min(1e-30)).item()))
        tensor_values.extend(top.tolist())
        values.extend(tensor_values)
        names.extend(f"{tensor_name}.{name}" for name in statistic_names)
    return values, names


def _delta_ref(entry, state, outputs, root_state, root_outputs):
    names = [f"{name}.relative_l2" for name, _ in _weight_tensors(state)]
    for suffix in outputs:
        names.append(f"kl_model_root_{suffix}")
        names.extend(f"{stage}_mean_activation_shift_{suffix}" for stage in STAGES)
    if entry["label"] == "root":
        return [0.0] * len(names), names

    root_weights = dict(_weight_tensors(root_state))
    values = []
    for name, tensor in _weight_tensors(state):
        root_tensor = root_weights[name]
        values.append((torch.linalg.vector_norm(tensor - root_tensor) /
                       torch.linalg.vector_norm(root_tensor).clamp_min(1e-30)).item())
    for suffix, (logits, _, activations) in outputs.items():
        root_logits, _, root_activations = root_outputs[suffix]
        log_probabilities = logits.log_softmax(1)
        root_log_probabilities = root_logits.log_softmax(1)
        probabilities = log_probabilities.exp()
        values.append((probabilities * (log_probabilities - root_log_probabilities)).sum(1).mean().item())
        for activation, root_activation in zip(activations, root_activations):
            values.append(torch.linalg.vector_norm(
                activation.mean(0) - root_activation.mean(0)).item())
    return values, names


def extract_fingerprints(out_dir="runs/pilot"):
    out_dir = Path(out_dir)
    manifest = _load_manifest(out_dir)
    if not manifest:
        raise ValueError("manifest is empty")

    started = time.time()
    splits = zoo.load_splits(seed=0)
    public_probes = {"id": splits["probe_id"], "ood": splits["probe_ood"]}
    probe_id_y = splits["probe_id_y"]
    probe_sets = {dataset: (public_probes, probe_id_y)
                  for dataset in {entry.get("dataset", "cifar10") for entry in manifest}}
    del splits
    probes = next(iter(probe_sets.values()))[0]
    generator = torch.Generator().manual_seed(0)
    sample_indices = torch.randperm(len(probes["id"]), generator=generator)[:512].cuda()
    triangle = torch.triu_indices(512, 512, offset=1, device="cuda")
    triangle = (triangle[0], triangle[1])

    root_entries = {entry["root_group"]: entry for entry in manifest if entry["label"] == "root"}
    if len(root_entries) != len({entry["root_group"] for entry in manifest}):
        raise ValueError("each root_group must contain one root checkpoint")

    probe_groups = {name: [] for name in (
        "confound", "blackbox", "logit", "actgeom", "topology"
    )}
    arch_groups = {arch: {"weights": [], "delta_ref": []}
                   for arch in {entry.get("arch", "cnn") for entry in manifest}}
    feature_names = {}
    cached_group = cached_state = cached_outputs = None

    def load(entry):
        dataset = entry.get("dataset", "cifar10")
        entry_probes = probe_sets[dataset][0]
        state = {name: tensor.float() for name, tensor in torch.load(
            out_dir / "models" / f"{entry['model_id']}.pt", map_location="cpu", weights_only=True
        ).items()}
        model = zoo.new_model(0, arch=entry.get("arch", "cnn"))
        model.load_state_dict(state)
        model.eval()
        outputs = {suffix: _forward(model, probe) for suffix, probe in entry_probes.items()}
        del model
        return state, outputs

    for index, entry in enumerate(manifest, 1):
        state, outputs = load(entry)
        if entry["root_group"] != cached_group:
            cached_group = entry["root_group"]
            if entry["label"] == "root":
                cached_state, cached_outputs = state, outputs
            else:
                cached_state, cached_outputs = load(root_entries[cached_group])

        probe_id_y = probe_sets[entry.get("dataset", "cifar10")][1]
        descriptors = {
            "confound": _confound(entry, outputs),
            "blackbox": _blackbox(outputs),
            "logit": _logit(outputs, probe_id_y),
            "actgeom": _actgeom(outputs, sample_indices, triangle),
            "topology": _topology(outputs, sample_indices),
        }
        for group, (values, names) in descriptors.items():
            if group not in feature_names:
                feature_names[group] = names
            if names != feature_names[group] or len(values) != len(feature_names[group]):
                raise AssertionError(f"unstable {group} feature layout for {entry['model_id']}")
            probe_groups[group].append(values)
        arch = entry.get("arch", "cnn")
        for group, (values, names) in {
            "weights": _weights(state),
            "delta_ref": _delta_ref(entry, state, outputs, cached_state, cached_outputs),
        }.items():
            key = f"{group}.{arch}"
            if key not in feature_names:
                feature_names[key] = names
            if names != feature_names[key] or len(values) != len(feature_names[key]):
                raise AssertionError(f"unstable {key} feature layout for {entry['model_id']}")
            arch_groups[arch][group].append(values)
        print(f"fingerprint: {index}/{len(manifest)} {entry['model_id']}", flush=True)

    arrays = {name: np.asarray(rows, dtype=np.float64) for name, rows in probe_groups.items()}
    grid_layout = any("arch" in entry for entry in manifest)
    if grid_layout:
        for arch, descriptors in arch_groups.items():
            selected = [entry["model_id"] for entry in manifest if entry.get("arch", "cnn") == arch]
            arrays[f"model_ids.{arch}"] = np.asarray(selected)
            arrays.update({f"{group}.{arch}": np.asarray(rows, dtype=np.float64)
                           for group, rows in descriptors.items()})
    else:
        arrays.update({group: np.asarray(rows, dtype=np.float64)
                       for group, rows in arch_groups["cnn"].items()})
        for group in ("weights", "delta_ref"):
            feature_names[group] = feature_names.pop(f"{group}.cnn")
    for name, array in arrays.items():
        if name.startswith("model_ids."):
            continue
        assert np.isfinite(array).all(), f"NaN/Inf in fingerprint group {name}"
        print(f"  {name}: {array.shape[1]} features")
    np.savez(out_dir / "fingerprints.npz", model_ids=np.asarray(
        [entry["model_id"] for entry in manifest]), **arrays)
    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"fingerprints: {len(manifest)} models in {time.time() - started:.1f}s -> {out_dir}")
    return arrays


if __name__ == "__main__":
    extract_fingerprints()
