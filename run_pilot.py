"""Run the model-autopsy pilot pipeline."""
import argparse
import json
from pathlib import Path

import numpy as np

from autopsy import (baseline_features, evaluate, evaluate_adaptive, evaluate_baselines,
                     evaluate_grid, evaluate_laundering, evaluate_openset, evaluate_pairs,
                     fingerprint, identifiability, revision_analysis, zoo)


def _manifest(out_dir):
    path = out_dir / "manifest.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _fingerprints_match(out_dir, manifest):
    path = out_dir / "fingerprints.npz"
    if not path.exists():
        return False
    with np.load(path) as archive:
        return ("model_ids" in archive and archive["model_ids"].tolist() ==
                [entry["model_id"] for entry in manifest])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("zoo", "fingerprint", "eval", "all", "pairs",
                                             "openset", "launder", "grid", "transductive",
                                             "identify", "baselines", "adaptive", "revision"),
                        default="all")
    parser.add_argument("--n-roots", type=int, default=40)
    parser.add_argument("--n-roots-per-cell", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--roots-from", type=Path, default=Path("runs/pilot_matched"))
    parser.add_argument("--source", type=Path, default=Path("runs/pilot_matched"))
    args = parser.parse_args()
    if args.out is None:
        args.out = {"pairs": Path("runs/pairs"), "openset": Path("runs/openset"),
                    "launder": Path("runs/laundered"),
                    "grid": Path("runs/grid"), "transductive": Path("runs/grid"),
                    "identify": Path("runs/identifiability"),
                    "baselines": Path("runs/baselines"),
                    "adaptive": Path("runs/adaptive"),
                    "revision": Path("runs/revision")}.get(
                        args.stage, Path("runs/pilot"))
    if args.smoke:
        args.n_roots = 2
        args.n_roots_per_cell = 2
        print("smoke: n_roots=2, n_roots_per_cell=2")

    stages = {
        "adaptive": ("adaptive", "fingerprint", "eval_adaptive"),
        "revision": ("revision",),
        "baselines": ("baseline_features", "eval_baselines"),
        "grid": ("grid", "fingerprint", "eval_grid"),
        "all": ("zoo", "fingerprint", "eval"),
        "pairs": ("pairs", "fingerprint", "eval_pairs"),
        "openset": ("unknowns", "fingerprint", "eval_openset"),
        "launder": ("laundered", "fingerprint", "eval_laundering"),
    }.get(args.stage, (args.stage,))
    manifest = _manifest(args.out)
    if "grid" in stages:
        expected = args.n_roots_per_cell * (10 * 6 + 5 * 2)
        if len(manifest) == expected:
            print(f"grid: skipping existing manifest with {expected} rows")
        else:
            manifest = zoo.build_grid(args.out, n_roots_per_cell=args.n_roots_per_cell)
    if "zoo" in stages:
        expected = args.n_roots * 6
        if len(manifest) == expected:
            print(f"zoo: skipping existing manifest with {expected} rows")
        else:
            manifest = zoo.build_zoo(args.out, n_roots=args.n_roots)
    if "pairs" in stages:
        expected = args.n_roots * 7
        if len(manifest) == expected:
            print(f"pairs: skipping existing manifest with {expected} rows")
        else:
            manifest = zoo.build_pairs(args.out, args.roots_from, n_roots=args.n_roots)
    if "unknowns" in stages:
        expected = args.n_roots * 3
        if len(manifest) == expected:
            print(f"unknowns: skipping existing manifest with {expected} rows")
        else:
            manifest = zoo.build_unknowns(args.out, args.roots_from, n_roots=args.n_roots)
    if "laundered" in stages:
        expected = args.n_roots * (1 + len(zoo.OPERATIONS) * len(zoo.LAUNDRIES))
        if len(manifest) == expected:
            print(f"laundered: skipping existing manifest with {expected} rows")
        else:
            manifest = zoo.build_laundered(args.out, args.source, n_roots=args.n_roots)
    if "adaptive" in stages:
        expected = args.n_roots * (1 + len(zoo.OPERATIONS) * 2)
        if len(manifest) == expected:
            print(f"adaptive: skipping existing manifest with {expected} rows")
        else:
            manifest = zoo.build_adaptive(
                args.out, args.source, n_roots=args.n_roots)
    if "fingerprint" in stages:
        manifest = _manifest(args.out)
        if not manifest:
            raise FileNotFoundError(f"{args.out / 'manifest.jsonl'}; run the zoo stage first")
        if _fingerprints_match(args.out, manifest):
            print(f"fingerprint: skipping existing archive with {len(manifest)} rows")
        else:
            fingerprint.extract_fingerprints(args.out)
    if "eval" in stages:
        evaluate.evaluate(args.out)
    if "eval_grid" in stages:
        evaluate_grid.evaluate_grid(args.out)
    if "eval_pairs" in stages:
        evaluate_pairs.evaluate_pairs(args.out, args.roots_from)
    if "eval_openset" in stages:
        evaluate_openset.evaluate_openset(args.out, args.roots_from)
    if "eval_laundering" in stages:
        evaluate_laundering.evaluate_laundering(args.out, args.source)
    if "eval_adaptive" in stages:
        evaluate_adaptive.evaluate_adaptive(args.out, args.source)
    if "transductive" in stages:
        evaluate_grid.evaluate_transductive(args.out)
    if "identify" in stages:
        identifiability.evaluate_identifiability(args.out)
    if "baseline_features" in stages:
        baseline_features.extract_baseline_features(args.out)
    if "eval_baselines" in stages:
        evaluate_baselines.evaluate_baselines(args.out)
    if "revision" in stages:
        revision_analysis.evaluate_revision(args.out)


if __name__ == "__main__":
    main()
