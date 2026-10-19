"""Run a small, reproducible CatBoost defaults transfer study.

Example: python run_experiment.py --iterations 300 --seeds 42 --meta-candidates 6 --hpo-trials 5
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
import sklearn

from configs import Configuration, META_CANDIDATES, PAPER_TD, library_default
from datasets import SPECS, load_dataset
from experiment import aggregate_nrmse, refit_and_test, split_indices, validate_config


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000, help="Maximum CatBoost trees (paper: 1000)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--meta-candidates", type=int, default=12, help="First N configurations from configs.py")
    parser.add_argument("--hpo-trials", type=int, default=20, help="Random search trials per held-out dataset and split")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/uci"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--skip-hpo", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Two tasks, 50 trees, no meta-search or HPO")
    args = parser.parse_args()
    if args.iterations < 10 or args.threads < 1 or args.meta_candidates < 1 or args.hpo_trials < 1:
        parser.error("iterations >= 10 and other counts >= 1 are required")
    if args.meta_candidates > len(META_CANDIDATES):
        parser.error(f"At most {len(META_CANDIDATES)} meta candidates are defined")
    if args.smoke:
        args.iterations, args.seeds, args.skip_hpo = 50, [42], True
    return args


def row_for(
    spec, seed: int, config: Configuration, x, y, cats,
    indices, iterations: int, threads: int, stage: str,
) -> dict:
    train, val, test = indices
    validation = validate_config(x, y, cats, train, val, config, iterations, seed, threads)
    testing = refit_and_test(x, y, cats, train, val, test, config, validation["best_iteration"], seed, threads)
    return {"stage": stage, "dataset": spec.slug, "role": spec.role, "seed": seed,
            "configuration": config.name, "rows": len(x), "features": x.shape[1],
            **validation, **testing}


def hpo_candidates(seed: int, count: int):
    rng = np.random.default_rng(seed)
    for index in range(count):
        yield Configuration(f"hpo_{index}", {
            "boosting_type": "Plain", "bootstrap_type": "Bernoulli",
            "depth": int(rng.integers(4, 11)),
            "learning_rate": float(np.exp(rng.uniform(np.log(0.02), np.log(0.2)))),
            "l2_leaf_reg": float(np.exp(rng.uniform(np.log(0.001), np.log(10.0)))),
            "subsample": float(rng.uniform(0.65, 1.0)),
            "random_strength": float(rng.uniform(0.0, 2.0)),
            "one_hot_max_size": 20,
        })


def main() -> None:
    args = options()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    specs = [spec for spec in SPECS if spec.slug in ("concrete", "power_plant")] if args.smoke else SPECS
    data = {spec.slug: load_dataset(spec, args.cache_dir) for spec in specs}
    splits = {(spec.slug, seed): split_indices(len(data[spec.slug][0]), seed)
              for spec in specs for seed in args.seeds}
    run_start = time.perf_counter()
    rows: list[dict] = []
    for spec in specs:
        x, y, cats = data[spec.slug]
        for seed in args.seeds:
            for config in (library_default(), PAPER_TD):
                result = row_for(spec, seed, config, x, y, cats, splits[spec.slug, seed],
                                 args.iterations, args.threads, "baseline")
                rows.append(result)
                print(f"baseline {spec.slug:20} seed={seed} {config.name:16} nRMSE={result['test_nrmse']:.4f}", flush=True)

    chosen = None
    search_rows = []
    if not args.smoke:
        for config in META_CANDIDATES[:args.meta_candidates]:
            for spec in specs:
                if spec.role != "meta_train":
                    continue
                x, y, cats = data[spec.slug]
                for seed in args.seeds:
                    train, val, _ = splits[spec.slug, seed]
                    validation = validate_config(x, y, cats, train, val, config,
                                                 args.iterations, seed, args.threads)
                    sd = float(np.std(y.iloc[train], ddof=0))
                    search_rows.append({"configuration": config.name, "dataset": spec.slug,
                                        "seed": seed, "val_nrmse": validation["val_rmse"] / sd,
                                        **validation})
            print(f"meta-search {config.name} complete", flush=True)
        search = pd.DataFrame(search_rows)
        # One vote per dataset: average its splits, then a shifted geometric mean.
        mean_by_task = search.groupby(["configuration", "dataset"])["val_nrmse"].mean().reset_index()
        scores = mean_by_task.groupby("configuration")["val_nrmse"].apply(
            lambda s: float(np.exp(np.log(s.to_numpy() + 0.01).mean()) - 0.01))
        chosen = next(c for c in META_CANDIDATES if c.name == scores.idxmin())
        print(f"selected on meta-train validation only: {chosen.name}", flush=True)
        for spec in specs:
            if spec.role != "meta_test":
                continue
            x, y, cats = data[spec.slug]
            for seed in args.seeds:
                result = row_for(spec, seed, chosen, x, y, cats, splits[spec.slug, seed],
                                 args.iterations, args.threads, "transferred_default")
                rows.append(result)
                print(f"transfer {spec.slug:20} seed={seed} nRMSE={result['test_nrmse']:.4f}", flush=True)

        if not args.skip_hpo:
            for spec in specs:
                if spec.role != "meta_test":
                    continue
                x, y, cats = data[spec.slug]
                for seed in args.seeds:
                    train, val, test = splits[spec.slug, seed]
                    candidates = []
                    for config in hpo_candidates(seed + spec.uci_id, args.hpo_trials):
                        validation = validate_config(x, y, cats, train, val, config,
                                                     args.iterations, seed, args.threads)
                        candidates.append((validation["val_rmse"], config, validation))
                    _, best, best_validation = min(candidates, key=lambda item: item[0])
                    testing = refit_and_test(x, y, cats, train, val, test, best,
                                             best_validation["best_iteration"], seed, args.threads)
                    result = {"stage": "per_dataset_hpo", "dataset": spec.slug, "role": spec.role,
                              "seed": seed, "configuration": best.name, "rows": len(x),
                              "features": x.shape[1], **best_validation, **testing,
                              "hpo_trials": args.hpo_trials,
                              "search_seconds": sum(v[2]["validation_seconds"] for v in candidates),
                              "selected_params": json.dumps(best.params, sort_keys=True)}
                    rows.append(result)
                    print(f"HPO {spec.slug:20} seed={seed} nRMSE={result['test_nrmse']:.4f}", flush=True)

    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(args.results_dir / "metrics.csv", index=False)
    if search_rows:
        pd.DataFrame(search_rows).to_csv(args.results_dir / "meta_search.csv", index=False)
    metadata = {"iterations": args.iterations, "seeds": args.seeds, "meta_candidates": args.meta_candidates,
                "hpo_trials": 0 if args.skip_hpo else args.hpo_trials, "smoke": args.smoke,
                "chosen_meta_default": None if chosen is None else {"name": chosen.name, "params": chosen.params},
                "elapsed_seconds": time.perf_counter() - run_start,
                "versions": {"python": platform.python_version(), "catboost": catboost.__version__,
                             "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}}
    (args.results_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary = result_frame.groupby(["role", "dataset", "stage", "configuration"], dropna=False).agg(
        n=("test_nrmse", "size"), nrmse_mean=("test_nrmse", "mean"),
        rmse_mean=("test_rmse", "mean"),
        total_train_seconds=("validation_seconds", "sum"),
    ).reset_index()
    summary.to_csv(args.results_dir / "summary.csv", index=False)
    print("\nResults saved in", args.results_dir, flush=True)
    print(summary.to_string(index=False), flush=True)
    if not args.smoke:
        for label in ("library_default", "paper_td"):
            subset = result_frame[(result_frame.role == "meta_test") & (result_frame.configuration == label)]
            print(f"Held-out shifted geometric mean nRMSE, {label}: {aggregate_nrmse(subset):.4f}", flush=True)


if __name__ == "__main__":
    main()

