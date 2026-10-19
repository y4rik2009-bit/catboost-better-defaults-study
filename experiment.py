"""Leakage-free split, validation, refit and evaluation utilities."""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

from configs import Configuration


def split_indices(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(n_rows)
    train_val, test = train_test_split(all_idx, test_size=0.2, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.25, random_state=seed)
    assert len(set(train) & set(val)) == len(set(train) & set(test)) == len(set(val) & set(test)) == 0
    return train, val, test


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def normalized_rmse(actual: np.ndarray, predicted: np.ndarray, train_target_sd: float) -> float:
    if train_target_sd <= 0:
        raise ValueError("Training target has zero variance")
    return rmse(actual, predicted) / train_target_sd


def _model(config: Configuration, iterations: int, seed: int, threads: int, early_stop: bool) -> CatBoostRegressor:
    params = {
        "iterations": iterations,
        "loss_function": "RMSE",
        "random_seed": seed,
        "thread_count": threads,
        "task_type": "CPU",
        "allow_writing_files": False,
        "verbose": False,
        **config.params,
    }
    if early_stop:
        params.update(od_type="Iter", od_wait=min(300, max(20, iterations // 3)))
    return CatBoostRegressor(**params)


def validate_config(
    x: pd.DataFrame, y: pd.Series, cat_features: list[str],
    train_idx: np.ndarray, val_idx: np.ndarray, config: Configuration,
    iterations: int, seed: int, threads: int,
) -> dict:
    start = time.perf_counter()
    early_stop = config.name != "library_default"
    model = _model(config, iterations, seed, threads, early_stop)
    model.fit(
        x.iloc[train_idx], y.iloc[train_idx], cat_features=cat_features,
        eval_set=(x.iloc[val_idx], y.iloc[val_idx]) if early_stop else None,
    )
    prediction = model.predict(x.iloc[val_idx])
    best_iteration = (model.get_best_iteration() + 1) if early_stop else iterations
    return {"val_rmse": rmse(y.iloc[val_idx], prediction),
            "best_iteration": int(best_iteration), "validation_seconds": time.perf_counter() - start}


def refit_and_test(
    x: pd.DataFrame, y: pd.Series, cat_features: list[str],
    train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
    config: Configuration, fitted_iterations: int, seed: int, threads: int,
) -> dict:
    start = time.perf_counter()
    model = _model(config, fitted_iterations, seed, threads, early_stop=False)
    train_val = np.concatenate((train_idx, val_idx))
    model.fit(x.iloc[train_val], y.iloc[train_val], cat_features=cat_features)
    pred = model.predict(x.iloc[test_idx])
    target_sd = float(np.std(y.iloc[train_val], ddof=0))
    return {"test_rmse": rmse(y.iloc[test_idx], pred),
            "test_nrmse": normalized_rmse(y.iloc[test_idx], pred, target_sd),
            "refit_seconds": time.perf_counter() - start}


def aggregate_nrmse(frame: pd.DataFrame) -> float:
    """Mean per task, then shifted geometric mean across tasks."""
    values = frame.groupby("dataset", sort=True)["test_nrmse"].mean().to_numpy()
    return float(math.exp(np.log(values + 0.01).mean()) - 0.01)

