"""Extract raw features used by the external baseline comparison."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import zoo
from .fingerprint import _forward, _weight_tensors


RUNS = (("pilot_matched", Path("runs/pilot_matched")),
        ("pairs", Path("runs/pairs")),
        ("laundered_ft", Path("runs/laundered")))
PAIR_LABELS = ("sft-unlearn", "unlearn-sft")
CACHE_VERSION = 2


def _manifest(path):
    with open(path / "manifest.jsonl") as file:
        return [json.loads(line) for line in file if line.strip()]


def _records():
    records = []
    for run_name, path in RUNS:
        manifest = _manifest(path)
        if run_name == "pairs":
            manifest = [entry for entry in manifest if entry["label"] in PAIR_LABELS]
        elif run_name == "laundered_ft":
            manifest = [entry for entry in manifest if entry.get("laundry") == "ft"]
        records.extend((run_name, path, entry) for entry in manifest)
    counts = {name: sum(record[0] == name for record in records) for name, _ in RUNS}
    expected = {"pilot_matched": 240, "pairs": 80, "laundered_ft": 200}
    if counts != expected:
        raise ValueError(f"unexpected baseline populations: {counts}, expected {expected}")
    return records


def _cache_matches(path, records):
    if not path.exists():
        return False
    with np.load(path) as archive:
        return (archive.get("cache_version", 0) == CACHE_VERSION
                and archive["model_ids"].tolist() == [record[2]["model_id"] for record in records]
                and archive["run_names"].tolist() == [record[0] for record in records]
                and archive["probe_profile"].shape == (len(records), 40_000)
                and archive["act_means"].shape == (len(records), 1_536)
                and archive["act_sub"].shape == (len(records), 512, 256)
                and archive["raw_weight_gram"].shape == (len(records), len(records)))


@torch.no_grad()
def extract_baseline_features(out_dir="runs/baselines"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "raw_features.npz"
    records = _records()
    if _cache_matches(cache, records):
        print(f"baselines: skipping existing raw feature archive with {len(records)} rows")
        return cache

    started = time.time()
    splits = zoo.load_splits(seed=0)
    probes = {"id": splits["probe_id"], "ood": splits["probe_ood"]}
    sample_indices = torch.randperm(len(probes["id"]),
                                    generator=torch.Generator().manual_seed(0))[:512].cuda()
    del splits

    n_rows = len(records)
    probe_profile = np.empty((n_rows, 40_000), dtype=np.float16)
    act_means = np.empty((n_rows, 1_536), dtype=np.float16)
    act_sub = np.empty((n_rows, 512, 256), dtype=np.float16)
    raw_weights = None
    weight_names = None

    for index, (run_name, run_dir, entry) in enumerate(records):
        state = {name: tensor.float() for name, tensor in torch.load(
            run_dir / "models" / f"{entry['model_id']}.pt", map_location="cpu",
            weights_only=True).items()}
        tensors = _weight_tensors(state)
        names = [name for name, _ in tensors]
        flat_weights = torch.cat([tensor.flatten() for _, tensor in tensors])
        if raw_weights is None:
            weight_names = names
            raw_weights = torch.empty((n_rows, len(flat_weights)), device="cuda")
        if names != weight_names or len(flat_weights) != raw_weights.shape[1]:
            raise AssertionError(f"unstable raw weight layout for {entry['model_id']}")
        raw_weights[index].copy_(flat_weights.cuda())

        model = zoo.new_model(0, arch=entry.get("arch", "cnn"))
        model.load_state_dict(state)
        model.eval()
        outputs = {suffix: _forward(model, probe) for suffix, probe in probes.items()}
        probe_profile[index] = torch.cat([
            outputs[suffix][0].softmax(1).flatten() for suffix in ("id", "ood")
        ]).half().cpu().numpy()
        act_means[index] = torch.cat([
            activation.mean(0) for suffix in ("id", "ood")
            for activation in outputs[suffix][2]
        ]).half().cpu().numpy()
        act_sub[index] = outputs["id"][2][-1][sample_indices].half().cpu().numpy()
        del model, state, outputs, flat_weights
        print(f"baseline features: {index + 1}/{n_rows} {run_name}/{entry['model_id']}",
              flush=True)

    # PCA is translation invariant; shifting first prevents cancellation when a
    # train fold centers the much larger unshifted dot products.
    raw_weights -= raw_weights[0].clone()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    raw_weight_gram = (raw_weights @ raw_weights.T).cpu().numpy()
    torch.backends.cuda.matmul.allow_tf32 = old_tf32
    del raw_weights
    arrays = {"probe_profile": probe_profile, "act_means": act_means,
              "act_sub": act_sub, "raw_weight_gram": raw_weight_gram}
    for name, array in arrays.items():
        assert np.isfinite(array).all(), f"NaN/Inf in {name}"
    assert act_sub.shape == (n_rows, 512, 256)

    np.savez(cache, cache_version=np.asarray(CACHE_VERSION),
             model_ids=np.asarray([record[2]["model_id"] for record in records]),
             run_names=np.asarray([record[0] for record in records]), **arrays)
    print(f"baseline features: {n_rows} models in {time.time() - started:.1f}s -> {cache}")
    return cache


if __name__ == "__main__":
    extract_baseline_features()
