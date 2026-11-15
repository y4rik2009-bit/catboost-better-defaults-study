"""Paste this ENTIRE file into one Kaggle Python cell, then Save & Run All.

No GitHub upload. Training runs only when this cell/script is executed by you.
See RUN_GUIDE_RU.md for GPU, Internet, saved versions and cross-session resume.
"""


import subprocess
import sys

INSTALL_REQUIREMENTS = [
    "catboost==1.2.8", "optuna==4.5.0",
    "numpy>=1.26,<3", "pandas>=2.1,<3", "scikit-learn>=1.3,<2",
    "requests>=2.31,<3", "matplotlib>=3.7,<4", "joblib>=1.3,<2",
]
if __name__ == "__main__":
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--disable-pip-version-check", *INSTALL_REQUIREMENTS,
    ])


import gc
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import os
import platform
import shutil
import sqlite3
import time
import traceback
import types
import zipfile
from datetime import datetime, timezone
from contextlib import closing
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import requests
from scipy import sparse
from catboost import CatBoostRegressor, Pool
from catboost.utils import get_gpu_device_count
from optuna.trial import TrialState
from requests.adapters import HTTPAdapter
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from urllib3.util.retry import Retry




CONFIG = {
    "protocol_version": "gpu-regression-defaults-v1",
    "global_trials": 30,
    "local_trials": 30,
    "split_seed": 2026,
    "search_model_seed": 42,
    "final_model_seeds": [42, 137, 2026],
    "sampler_seed": 1907,
    "max_rows_per_dataset": 30000,
    "devices": "0:1",
    "gpu_ram_part": 0.80,
    "thread_count": 4,
    "session_hours": 7.5,
    "stop_margin_minutes": 20,
    "full_checkpoint_minutes": 15,
    "save_final_models": True,
}
BASE = Path("/kaggle/working/catboost_defaults_lab")



RESUME_FROM = ""

AUTO_RESUME_FROM_KAGGLE_INPUT = True
COMPATIBLE_HOTFIX_FINGERPRINT_PREFIXES = ("fadfcf6ce4eb",)




DEVELOPMENT = [
    {"id": 42225, "name": "diamonds", "target": "price"},
    {"id": 550, "name": "quake", "target": "col_4"},
    {"id": 546, "name": "sensory", "target": "Score"},
    {"id": 541, "name": "socmob", "target": "counts_for_sons_current_occupation"},
    {"id": 507, "name": "space_ga", "target": "ln(VOTES/POP)"},
    {"id": 42571, "name": "Allstate_Claims_Severity", "target": "loss"},
    {"id": 287, "name": "wine_quality", "target": "quality"},
    {"id": 216, "name": "elevators", "target": "Goal"},
    {"id": 42570, "name": "Mercedes_Benz_Greener_Manufacturing", "target": "y"},
    {"id": 42730, "name": "us_crime", "target": "ViolentCrimesPerPop"},
]
HELDOUT = [
    {"id": 42563, "name": "house_prices_nominal", "target": "SalePrice"},
    {"id": 42726, "name": "abalone", "target": "Class_number_of_rings"},
    {"id": 201, "name": "pol", "target": "foo"},
    {"id": 42724, "name": "OnlineNewsPopularity", "target": "shares"},
    {"id": 3050, "name": "QSAR-TID-11", "target": "MEDIAN_PXC50"},
]
SPACE = {
    "iterations": [500, 1000, 2000],
    "depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.20, "log"],
    "l2_leaf_reg": [1.0, 50.0, "log"],
    "border_count": [32, 64, 128],
    "bootstrap_type": ["Bayesian", "Bernoulli"],
    "bagging_temperature": [0.0, 5.0],
    "subsample": [0.6, 1.0],
    "one_hot_max_size": [2, 10, 25],
}




LOG = logging.getLogger("defaults-lab")
START = 0.0
LAST_CHECKPOINT = 0.0
RUN = BASE / "run"
CACHE = BASE / "cache"
ARCHIVES = BASE / "archives"
DB = RUN / "optuna.sqlite3"
STORAGE = None
FINGERPRINT = ""


class PauseRun(Exception):
    """Clean pause between GPU fits; next session continues the same protocol."""


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def code_signature(code):
    """Notebook/file-independent implementation ID (ignore paths/line numbers)."""
    def constant(value):
        if isinstance(value, types.CodeType):
            return code_signature(value)
        if isinstance(value, tuple):
            return [constant(v) for v in value]
        if isinstance(value, frozenset):
            return sorted(constant(v) for v in value)
        return repr(value)
    return {"bytecode": code.co_code.hex(), "constants": [constant(v) for v in code.co_consts],
            "names": code.co_names, "arguments": code.co_varnames, "freevars": code.co_freevars}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def write_status(state, **extra):
    atomic_json(RUN / "status.json", {
        "state": state, "updated_utc": now(), "fingerprint": FINGERPRINT, **extra,
    })


def check_time():
    available = CONFIG["session_hours"] * 3600 - CONFIG["stop_margin_minutes"] * 60
    if time.monotonic() - START >= available:
        raise PauseRun("Soft session budget reached; resume in another saved GPU run.")


def restore():
    resume_source = RESUME_FROM
    if not resume_source and AUTO_RESUME_FROM_KAGGLE_INPUT:
        input_root = Path("/kaggle/input")
        candidates = sorted(input_root.rglob("resume_checkpoint.zip")) if input_root.exists() else []
        if len(candidates) > 1:
            raise RuntimeError(f"Found multiple resume checkpoints; set RESUME_FROM explicitly: {candidates}")
        if len(candidates) == 1:
            resume_source = str(candidates[0])
            print(f"AUTO-RESUME checkpoint: {resume_source}")
    if not resume_source:
        return
    src = Path(resume_source)
    if (RUN / "manifest.json").exists():
        raise RuntimeError("Local state already exists. Clear RESUME_FROM to continue it.")
    if src.is_dir():
        if not (src / "run" / "manifest.json").exists():
            raise ValueError("Resume directory must contain run/manifest.json.")
        shutil.copytree(src / "run", RUN, dirs_exist_ok=True)
    elif src.is_file():
        with zipfile.ZipFile(src) as archive:
            names = archive.namelist()
            if "run/manifest.json" not in names or "run/optuna.sqlite3" not in names:
                raise ValueError("Not a resume checkpoint (do not use analysis_results.zip).")
            for item in archive.infolist():
                dest = (BASE / item.filename).resolve()
                if not dest.is_relative_to(BASE.resolve()):
                    raise ValueError("Unsafe archive path.")
            if "CHECKPOINT.json" not in names or not json.loads(archive.read("CHECKPOINT.json")).get("full_resumable"):
                raise ValueError("Analysis archive is not a full resume checkpoint.")
            if "SHA256.json" not in names:
                raise ValueError("Missing checkpoint integrity manifest.")
            for name, expected in json.loads(archive.read("SHA256.json")).items():
                h = hashlib.sha256()
                with archive.open(name) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(block)
                if h.hexdigest() != expected:
                    raise ValueError(f"Checkpoint hash mismatch: {name}")
            archive.extractall(BASE)
    else:
        raise FileNotFoundError(f"Resume input not found: {src}")


def zip_state(full):
    """SQLite backup API gives a consistent DB; never zip a live DB blindly."""
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    if not DB.exists():
        return
    backup = BASE / "db_backup.sqlite3"
    with closing(sqlite3.connect(DB, timeout=60)) as source, closing(sqlite3.connect(backup)) as dest:
        source.backup(dest)
    target = ARCHIVES / ("resume_checkpoint.zip" if full else "analysis_results.zip")
    tmp = target.with_suffix(".tmp")
    excluded = {"models", "snapshots"} if not full else set()
    hashes = {}
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(RUN.rglob("*")):
            if not path.is_file() or path.name.endswith((".tmp", "-wal", "-shm")):
                continue
            rel = path.relative_to(RUN)
            if any(part in excluded for part in rel.parts) or path == DB:
                continue
            name = "run/" + rel.as_posix()
            archive.write(path, name)
            hashes[name] = file_sha(path)
        archive.write(backup, "run/optuna.sqlite3")
        hashes["run/optuna.sqlite3"] = file_sha(backup)
        archive.writestr("SHA256.json", json.dumps(hashes, indent=2))
        archive.writestr("CHECKPOINT.json", json.dumps({"full_resumable": full, "fingerprint": FINGERPRINT}))
        archive.writestr("ARCHIVE_INFO.txt", (
            f"Created {now()}\nFull resumable state: {full}\n"
            "No raw feature tables or downloaded OpenML files included.\n"
            "Analysis archive is for interpretation; full checkpoint is for resume.\n"
            "Contains test labels/predictions: do not use them for retuning.\n"
        ))
    os.replace(tmp, target)
    backup.unlink(missing_ok=True)


def checkpoint(force=False):
    global LAST_CHECKPOINT
    zip_state(full=False)
    if force or time.monotonic() - LAST_CHECKPOINT > CONFIG["full_checkpoint_minutes"] * 60:
        zip_state(full=True)
        LAST_CHECKPOINT = time.monotonic()


def inspect_required_gpus():
    """Return diagnostics or raise a readable error before any data download."""
    gpu_count = get_gpu_device_count()
    if gpu_count < 2:
        raise RuntimeError(
            f"This notebook requires Kaggle T4 x2, but CatBoost sees {gpu_count} GPU(s). "
            "Open Notebook options / Session options, select Accelerator: GPU T4 x2, "
            "then create a NEW version with Save Version -> Save & Run All. "
            "CPU fallback is intentionally disabled."
        )
    smi_path = shutil.which("nvidia-smi")
    if not smi_path:
        return gpu_count, "nvidia-smi is not on PATH; CatBoost reports two GPUs."
    gpu_process = subprocess.run([smi_path], capture_output=True, text=True, check=False)
    return gpu_count, gpu_process.stdout or gpu_process.stderr


def is_compatible_hotfix(saved_manifest, new_protocol):
    old_protocol = dict(saved_manifest.get("protocol", {}))
    new_protocol = dict(new_protocol)
    old_protocol.pop("implementation_hash", None)
    new_protocol.pop("implementation_hash", None)
    return (
        saved_manifest.get("fingerprint", "").startswith(COMPATIBLE_HOTFIX_FINGERPRINT_PREFIXES)
        and old_protocol == new_protocol
    )


def setup():
    global START, LAST_CHECKPOINT, STORAGE, FINGERPRINT
    START = LAST_CHECKPOINT = time.monotonic()
    for path in [BASE, RUN, CACHE, ARCHIVES]:
        path.mkdir(parents=True, exist_ok=True)
    restore()
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(RUN / "run.log", encoding="utf-8")]:
        handler.setFormatter(formatter)
        LOG.addHandler(handler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    versions = {name: importlib.metadata.version(name) for name in [
        "catboost", "optuna", "numpy", "pandas", "scikit-learn", "requests", "joblib",
    ]}
    if versions["catboost"] != "1.2.8" or versions["optuna"] != "4.5.0":
        raise RuntimeError("Restart the Kaggle kernel after installing packages, then rerun.")
    protocol = {
        "python_major_minor": list(sys.version_info[:2]),
        "implementation_hash": digest({f.__name__: code_signature(f.__code__) for f in [
            prepare_features, split_groups, rmse, get_data, suggest, fit_gpu, search, make_selection, evaluate,
        ]}),
        "config": {k: v for k, v in CONFIG.items() if k not in {
            "session_hours", "stop_margin_minutes", "full_checkpoint_minutes", "devices", "thread_count",
        }},
        "development": DEVELOPMENT, "heldout": HELDOUT, "space": SPACE,
        "versions": versions,
        "objective": "mean(log(validation_RMSE / train_mean_dummy_validation_RMSE))",
        "split": "GroupShuffleSplit 60/20/20 by identical feature vector; fixed seed",
        "preprocessing": "metadata ID exclusion; finite y; X/y dedup; cap; train-only constant columns and y scaling",
        "early_stopping": False,
        "default": "CatBoost GPU RMSE; default 1000 iterations; no eval_set / best-model truncation",
    }
    FINGERPRINT = digest(protocol)
    manifest = RUN / "manifest.json"
    if manifest.exists():
        saved_manifest = read_json(manifest)
        if saved_manifest["fingerprint"] != FINGERPRINT:
            if not is_compatible_hotfix(saved_manifest, protocol):
                raise RuntimeError("Protocol/package versions changed. Restore matching versions or start a NEW run directory.")
            computed_fingerprint = FINGERPRINT
            FINGERPRINT = saved_manifest["fingerprint"]
            atomic_json(RUN / "compatible_hotfix.json", {
                "applied_utc": now(), "continued_fingerprint": FINGERPRINT,
                "new_implementation_fingerprint": computed_fingerprint,
                "change": "Load Sparse ARFF with as_frame='auto' and densify its numeric CSR matrix.",
                "scientific_protocol_changed": False,
            })
    else:
        atomic_json(manifest, {"fingerprint": FINGERPRINT, "created_utc": now(), "protocol": protocol})

    try:
        if "__file__" in globals() and Path(__file__).is_file():
            code_text = Path(__file__).read_text(encoding="utf-8")
        else:
            from IPython import get_ipython
            code_text = get_ipython().history_manager.input_hist_raw[-1]
        (RUN / "executed_code.py").write_text(code_text, encoding="utf-8")
    except Exception:

        code_text = "\n\n".join(inspect.getsource(f) for f in [prepare_features, split_groups, get_data, suggest, fit_gpu, search])
        (RUN / "pipeline_functions.py").write_text(code_text, encoding="utf-8")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (RUN / "pip-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    gpu_count, gpu_diagnostics = inspect_required_gpus()
    atomic_json(RUN / f"environment-{time.time_ns()}.json", {
        "utc": now(), "python": sys.version, "platform": platform.platform(),
        "versions": versions, "gpu": gpu_diagnostics, "gpu_count": gpu_count,
        "devices": CONFIG["devices"],
        "session_hours": CONFIG["session_hours"],
    })
    STORAGE = optuna.storages.RDBStorage(
        url="sqlite:///" + DB.as_posix(), engine_kwargs={"connect_args": {"timeout": 60}},
    )
    write_status("running")
    LOG.info("GPU-only run; protocol %s; max %d rows per dataset", FINGERPRINT[:12], CONFIG["max_rows_per_dataset"])


def metadata_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def prepare_features(frame):
    """No fitted statistics; native numeric NaNs, explicit categorical strings."""
    frame = frame.copy()
    frame.columns = [str(column) for column in frame.columns]
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names.")
    categorical = []
    for column in frame:
        if pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column]):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan)
        else:
            categorical.append(column)
            frame[column] = frame[column].astype("string").map(
                lambda value: "__MISSING__" if pd.isna(value) else "value:" + str(value)
            ).astype(str)
    return frame, categorical


def openml_result_to_frames(fetched):
    """Normalize dense OpenML frames and numeric Sparse ARFF to pandas objects."""
    if sparse.issparse(fetched.data):
        feature_names = [str(name) for name in fetched.feature_names]
        if fetched.data.shape[1] != len(feature_names):
            raise ValueError("Sparse OpenML feature-name count mismatch.")

        X = pd.DataFrame(fetched.data.toarray(), columns=feature_names)
    elif isinstance(fetched.data, pd.DataFrame):
        X = fetched.data.copy()
    else:
        X = pd.DataFrame(fetched.data, columns=[str(name) for name in fetched.feature_names])
    target = np.asarray(fetched.target)
    if target.ndim != 1:
        raise ValueError(f"Expected one regression target, got shape {target.shape}.")
    return X, pd.Series(target, name="target")


def split_groups(groups, seed):
    positions = np.arange(len(groups))
    train_val, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(positions, groups=groups))
    train_rel, val_rel = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1).split(
        train_val, groups=groups[train_val],
    ))
    splits = {"train": train_val[train_rel], "validation": train_val[val_rel], "test": test}
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if set(groups[splits[left]]) & set(groups[splits[right]]):
            raise AssertionError("Feature group leakage.")
    if sum(map(len, splits.values())) != len(groups):
        raise AssertionError("Incomplete split.")
    return splits


def rmse(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != prediction.shape or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Invalid metric arrays.")
    return float(np.sqrt(np.mean(np.square(y - prediction))))


def get_data(spec, group):
    check_time()
    cache_path = CACHE / f"prepared-{spec['id']}-{FINGERPRINT[:12]}.joblib"
    registry = RUN / "datasets" / f"{spec['id']}.json"
    if cache_path.exists() and registry.exists():

        data = joblib.load(cache_path)
        if data["fingerprint"] != FINGERPRINT:
            raise RuntimeError("Wrong prepared-data cache.")
        return data
    LOG.info("Download/prepare %s [%s]", spec["name"], group)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))
    response = session.get(f"https://www.openml.org/api/v1/json/data/{spec['id']}", timeout=90)
    response.raise_for_status()
    meta = response.json()["data_set_description"]
    if meta["name"] != spec["name"] or meta.get("default_target_attribute") != spec["target"]:
        raise ValueError(f"OpenML metadata mismatch for {spec}: {meta.get('name')}")
    fetched = fetch_openml(
        data_id=spec["id"], target_column=spec["target"], as_frame="auto",
        data_home=str(CACHE / "openml"), parser="auto", n_retries=3, delay=3,
    )
    X, raw_target = openml_result_to_frames(fetched)
    y = pd.to_numeric(raw_target, errors="coerce").astype(float)
    source_rows = np.arange(len(X), dtype=np.int64)
    excluded_columns = metadata_list(meta.get("ignore_attribute")) + metadata_list(meta.get("row_id_attribute"))
    X = X.drop(columns=[c for c in excluded_columns if c in X], errors="ignore")
    if spec["target"] in X:
        raise AssertionError("Target is present in predictors.")
    original_shape = list(X.shape)
    finite = np.isfinite(y.to_numpy())
    dropped_target = int((~finite).sum())
    X, y, source_rows = X.loc[finite].reset_index(drop=True), y.loc[finite].reset_index(drop=True), source_rows[finite]
    X, cats = prepare_features(X)
    joint = X.copy()
    joint["__target_for_dedup__"] = y.to_numpy()
    unique = ~joint.duplicated()
    duplicate_count = int((~unique).sum())
    X, y, source_rows = X.loc[unique].reset_index(drop=True), y.loc[unique].reset_index(drop=True), source_rows[unique]
    del joint
    before_cap = len(X)
    if len(X) > CONFIG["max_rows_per_dataset"]:
        rng = np.random.default_rng(CONFIG["split_seed"] + spec["id"])
        take = np.sort(rng.choice(len(X), CONFIG["max_rows_per_dataset"], replace=False))
        X, y, source_rows = X.iloc[take].reset_index(drop=True), y.iloc[take].reset_index(drop=True), source_rows[take]
    if len(X) < 100 or y.nunique() < 10:

        if len(X) < 100 or y.nunique() < 3:
            raise ValueError("Too few rows or distinct target values.")
    feature_groups = pd.util.hash_pandas_object(X, index=False).to_numpy(dtype=np.uint64)
    splits = split_groups(feature_groups, CONFIG["split_seed"] + spec["id"])
    keep = [c for c in X if X.iloc[splits["train"]][c].nunique(dropna=False) > 1]
    dropped_constant = [c for c in X if c not in keep]
    X = X[keep]
    cats = [c for c in cats if c in keep]
    if not keep:
        raise ValueError("No nonconstant training features.")

    data_hash = hashlib.sha256(pd.util.hash_pandas_object(X, index=False).to_numpy().tobytes() + y.to_numpy(dtype=np.float64).tobytes() + source_rows.tobytes()).hexdigest()
    split_hash = digest({name: indices.tolist() for name, indices in splits.items()})
    y_arr = y.to_numpy(dtype=np.float64)
    center = float(y_arr[splits["train"]].mean())
    scale = float(y_arr[splits["train"]].std())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Constant training target.")
    dummy_val = rmse(y_arr[splits["validation"]], np.full(len(splits["validation"]), center))
    if dummy_val <= 0:
        raise ValueError("Degenerate validation target.")
    info = {
        "id": spec["id"], "name": spec["name"], "group": group, "target": spec["target"],
        "url": f"https://www.openml.org/d/{spec['id']}", "version": meta.get("version"),
        "licence_as_reported_by_openml": meta.get("licence"), "source_md5": meta.get("md5_checksum"),
        "original_shape": original_shape, "missing_targets_removed": dropped_target,
        "exact_duplicate_rows_removed": duplicate_count, "rows_before_cap": before_cap,
        "rows_used": len(X), "features_used": len(keep), "categorical_features": cats,
        "excluded_metadata_columns": excluded_columns, "train_constant_columns_removed": dropped_constant,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "unique_feature_groups": int(len(np.unique(feature_groups))), "split_hash": split_hash,
        "data_hash": data_hash, "y_train_mean": center, "y_train_std": scale,
        "dummy_validation_rmse": dummy_val,
        "missing_numeric_cells": int(X.select_dtypes(include="number").isna().sum().sum()),
        "target_summary": {"min": float(y.min()), "max": float(y.max()), "distinct": int(y.nunique())},
    }
    if registry.exists():
        old = read_json(registry)
        if old["data_hash"] != data_hash or old["split_hash"] != split_hash:
            raise RuntimeError("Data changed on resume; refusing to mix results.")
    atomic_json(registry, info)
    atomic_npz(RUN / "splits" / f"{spec['id']}.npz", source_rows=source_rows, feature_groups=feature_groups, **splits)
    data = {"X": X, "y": y_arr, "splits": splits, "source_rows": source_rows, "cats": cats,
            "center": center, "scale": scale, "dummy_val": dummy_val, "info": info, "fingerprint": FINGERPRINT}
    tmp = cache_path.with_suffix(".tmp")
    joblib.dump(data, tmp, compress=1)
    os.replace(tmp, cache_path)
    return data


def suggest(trial):
    params = {
        "iterations": trial.suggest_categorical("iterations", SPACE["iterations"]),
        "depth": trial.suggest_categorical("depth", SPACE["depth"]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
        "border_count": trial.suggest_categorical("border_count", SPACE["border_count"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", SPACE["bootstrap_type"]),
        "one_hot_max_size": trial.suggest_categorical("one_hot_max_size", SPACE["one_hot_max_size"]),
    }
    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 5.0)
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
    return params


def fit_gpu(data, params, seed, evaluate_test=False, model_name=None):
    """Cache key excludes trial number: interrupted global trials reuse subfits."""
    key = digest({"protocol": FINGERPRINT, "data": data["info"]["data_hash"],
                  "split": data["info"]["split_hash"], "params": params, "seed": seed,
                  "evaluation_method": model_name if evaluate_test else None})
    result_file = RUN / "fits" / f"{key}.json"
    snapshot = RUN / "snapshots" / f"{key}.snapshot"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    model_path = RUN / "models" / f"{data['info']['id']}-{model_name}-{seed}.cbm" if model_name else None
    predictions_path = RUN / "predictions" / f"{data['info']['id']}-{model_name}-{seed}.npz" if model_name else None
    if result_file.exists():
        record = read_json(result_file)
        if not evaluate_test:
            return record
        if predictions_path.exists() and (not CONFIG["save_final_models"] or model_path.exists()):
            return record
    check_time()
    started = time.monotonic()
    cpu_started = time.process_time()
    splits, X, y = data["splits"], data["X"], data["y"]
    train = Pool(X.iloc[splits["train"]], (y[splits["train"]] - data["center"]) / data["scale"], cat_features=data["cats"])
    validation = Pool(X.iloc[splits["validation"]], cat_features=data["cats"])
    common = {
        "loss_function": "RMSE", "task_type": "GPU", "devices": CONFIG["devices"],
        "random_seed": seed, "verbose": False, "allow_writing_files": True,
        "train_dir": str(RUN / "catboost_temp"), "thread_count": CONFIG["thread_count"],
        "gpu_ram_part": CONFIG["gpu_ram_part"], "use_best_model": False,
        "save_snapshot": True, "snapshot_file": str(snapshot), "snapshot_interval": 60,
    }
    if set(common) & set(params):
        raise ValueError("Search cannot override protocol parameters.")
    model = CatBoostRegressor(**common, **params)
    model.fit(train)
    pred_val = model.predict(validation) * data["scale"] + data["center"]
    val_rmse = rmse(y[splits["validation"]], pred_val)
    pred_train = model.predict(train) * data["scale"] + data["center"]
    resolved = model.get_all_params()
    record = {
        "key": key, "dataset_id": data["info"]["id"], "dataset": data["info"]["name"],
        "seed": seed, "requested_params": params, "resolved_params": resolved,
        "validation_rmse": val_rmse, "validation_normalized_rmse": val_rmse / data["dummy_val"],
        "training_rmse": rmse(y[splits["train"]], pred_train),
        "learning_curve_standardized_target": model.get_evals_result(),
        "train_validation_seconds": time.monotonic() - started,
        "cpu_seconds": time.process_time() - cpu_started, "trees": int(model.tree_count_), "completed_utc": now(),
    }
    if evaluate_test:
        test = Pool(X.iloc[splits["test"]], cat_features=data["cats"])
        tick = time.monotonic()
        pred_test = model.predict(test) * data["scale"] + data["center"]
        record["prediction_seconds"] = time.monotonic() - tick
        record["test"] = {
            "rmse": rmse(y[splits["test"]], pred_test),
            "mae": float(mean_absolute_error(y[splits["test"]], pred_test)),
            "r2": float(r2_score(y[splits["test"]], pred_test)),
            "normalized_rmse": rmse(y[splits["test"]], pred_test) / rmse(
                y[splits["test"]], np.full(len(splits["test"]), data["center"])),
        }
        atomic_npz(predictions_path, y_true=y[splits["test"]], y_pred=pred_test,
                   source_rows=data["source_rows"][splits["test"]])
        if CONFIG["save_final_models"]:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_model = model_path.with_suffix(".tmp")
            model.save_model(str(tmp_model), format="cbm")
            os.replace(tmp_model, model_path)
        record["model_name"] = model_name
    atomic_json(result_file, record)
    snapshot.unlink(missing_ok=True)
    del model, train, validation
    gc.collect()
    LOG.info("%s | seed %d | val RMSE %.6g | %d trees | %.1fs", data["info"]["name"], seed, val_rmse, record["trees"], record["train_validation_seconds"])
    return record


def load_study(name):
    study = optuna.create_study(
        study_name=name, storage=STORAGE, direction="minimize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=CONFIG["sampler_seed"], n_startup_trials=10),
        pruner=optuna.pruners.NopPruner(),
    )


    for frozen in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(frozen.number, state=TrialState.FAIL)
        if frozen.params:
            study.enqueue_trial(frozen.params, user_attrs={"resumed_from_trial": frozen.number})
    return study


def export_study(study):
    directory = RUN / "studies"
    directory.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(directory / f"{study.study_name}.csv", index=False)
    complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    atomic_json(directory / f"{study.study_name}.json", {
        "name": study.study_name, "complete": len(complete), "attempts": len(study.trials),
        "failed": sum(t.state == TrialState.FAIL for t in study.trials),
        "best_trial": study.best_trial.number if complete else None,
        "best_value": study.best_value if complete else None,
        "best_params": study.best_params if complete else None,
    })


def search(name, specs, group, target_trials):
    study = load_study(name)
    while len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))) < target_trials:
        check_time()

        seed = int(digest([CONFIG["sampler_seed"], name, len(study.trials)])[:8], 16)
        study.sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)
        trial = study.ask()
        try:
            params = suggest(trial)
            trial.set_user_attr("model_params", params)
            losses, measurements = [], []
            for spec in specs:
                check_time()
                data = get_data(spec, group)
                measurement = fit_gpu(data, params, CONFIG["search_model_seed"])
                losses.append(math.log(max(measurement["validation_normalized_rmse"], 1e-15)))
                measurements.append({k: measurement[k] for k in [
                    "key", "dataset_id", "validation_rmse", "validation_normalized_rmse", "train_validation_seconds",
                ]})
                trial.set_user_attr("dataset_measurements", measurements)
                del data
                gc.collect()
            value = float(np.mean(losses))
            study.tell(trial, value)
            LOG.info("STUDY %s | completed %d/%d | mean log normalized RMSE %.6f", name,
                     len(study.get_trials(states=(TrialState.COMPLETE,))), target_trials, value)
        except PauseRun:
            study.tell(trial, state=TrialState.FAIL)
            study.enqueue_trial(trial.params, user_attrs={"resumed_from_trial": trial.number})
            export_study(study)
            raise
        except BaseException as exc:
            trial.set_user_attr("failure", repr(exc))
            study.tell(trial, state=TrialState.FAIL)
            if isinstance(exc, KeyboardInterrupt):
                study.enqueue_trial(trial.params, user_attrs={"resumed_from_trial": trial.number})
            export_study(study)
            raise
        export_study(study)
        checkpoint()
    export_study(study)
    return study


def make_selection():
    locked = RUN / "locked_selection.json"
    if locked.exists():
        selection = read_json(locked)
        if selection["fingerprint"] != FINGERPRINT:
            raise RuntimeError("Invalid selection lock.")
        return selection
    global_study = load_study("global_defaults")
    if len(global_study.get_trials(states=(TrialState.COMPLETE,))) != CONFIG["global_trials"]:
        raise RuntimeError("Global search incomplete.")
    common = global_study.best_trial.user_attrs["model_params"]
    selected = {}
    for spec in HELDOUT:
        study = load_study(f"local_{spec['id']}")
        if len(study.get_trials(states=(TrialState.COMPLETE,))) != CONFIG["local_trials"]:
            raise RuntimeError("Local search incomplete.")
        selected[str(spec["id"])] = {
            "params": study.best_trial.user_attrs["model_params"],
            "trial": study.best_trial.number, "validation_objective": study.best_value,
        }
    selection = {"fingerprint": FINGERPRINT, "locked_utc": now(),
                 "global_params": common, "global_trial": global_study.best_trial.number,
                 "local": selected, "test_was_used_for_selection": False}
    atomic_json(locked, selection)
    checkpoint(force=True)
    return selection


def evaluate(selection):

    results = RUN / "evaluation"
    results.mkdir(parents=True, exist_ok=True)
    for spec in HELDOUT:
        data = get_data(spec, "heldout")
        methods = {
            "catboost_default": {},
            "shared_defaults": selection["global_params"],
            "optuna_30": selection["local"][str(spec["id"])]["params"],
        }
        for name, params in methods.items():
            for seed in CONFIG["final_model_seeds"]:
                output = results / f"{spec['id']}-{name}-{seed}.json"
                if output.exists():
                    continue
                check_time()
                measurement = fit_gpu(data, params, seed, evaluate_test=True, model_name=name)
                atomic_json(output, measurement)
                checkpoint()
        del data
        gc.collect()


def report():
    """Descriptive tables only; do not declare scientific winners automatically."""
    report_dir = RUN / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted((RUN / "evaluation").glob("*.json")):
        m = read_json(path)
        records.append({"dataset_id": m["dataset_id"], "dataset": m["dataset"], "method": m["model_name"],
                        "seed": m["seed"], **m["test"], "validation_rmse": m["validation_rmse"],
                        "fit_seconds": m["train_validation_seconds"], "trees": m["trees"],
                        "prediction_seconds": m["prediction_seconds"]})
    if not records:
        return
    frame = pd.DataFrame(records)
    frame.to_csv(report_dir / "test_metrics_by_seed.csv", index=False)
    summary = frame.groupby(["dataset_id", "dataset", "method"]).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), mae_mean=("mae", "mean"),
        r2_mean=("r2", "mean"), normalized_rmse_mean=("normalized_rmse", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"), seed_count=("seed", "count"),
    ).reset_index()
    summary.to_csv(report_dir / "test_summary.csv", index=False)
    expected = len(HELDOUT) * 3 * len(CONFIG["final_model_seeds"])
    complete = len(frame) == expected
    atomic_json(report_dir / "completeness.json", {"complete": complete, "observed": len(frame), "expected": expected,
                                                   "datasets": int(frame.dataset_id.nunique())})
    costs = []
    for path in sorted((RUN / "studies").glob("*.json")):
        meta = read_json(path)
        study = optuna.load_study(study_name=meta["name"], storage=STORAGE)
        measurements = {}
        for trial in study.trials:
            for measurement in trial.user_attrs.get("dataset_measurements", []):
                measurements[measurement["key"]] = measurement
        costs.append({"study": meta["name"], "completed_trials": meta["complete"], "attempts": meta["attempts"],
                      "unique_search_fits": len(measurements),
                      "sum_search_fit_seconds": sum(m["train_validation_seconds"] for m in measurements.values())})
    pd.DataFrame(costs).to_csv(report_dir / "search_costs.csv", index=False)

    if not complete:
        LOG.info("Partial evaluation: %d/%d results. No final comparison yet.", len(frame), expected)
        return
    pivot = summary.pivot(index="dataset", columns="method", values="rmse_mean")
    ratios = pivot.div(pivot["catboost_default"], axis=0)
    ratios.to_csv(report_dir / "rmse_ratios_vs_default.csv")
    ranks = pivot.rank(axis=1, method="average", ascending=True)
    aggregate = pd.DataFrame({
        "geometric_mean_rmse_ratio": np.exp(np.log(ratios.clip(lower=1e-15)).mean()),
        "mean_rank": ranks.mean(),
        "strict_wins": pivot.eq(pivot.min(axis=1), axis=0).where(pivot.eq(pivot.min(axis=1), axis=0).sum(axis=1).eq(1), False).sum(),
        "datasets": len(pivot),
    })
    aggregate.to_csv(report_dir / "descriptive_aggregate.csv")

    paired = frame.pivot(index=["dataset", "seed"], columns="method", values="rmse")
    for name in ["shared_defaults", "optuna_30"]:
        paired[f"{name}_relative_change_pct"] = 100 * (paired[name] / paired["catboost_default"] - 1)
    paired.to_csv(report_dir / "paired_seed_results.csv")
    ax = ratios.plot.bar(figsize=(12, 5), width=0.8)
    ax.axhline(1, color="black", linewidth=1)
    ax.set_ylabel("Test RMSE / standard CatBoost RMSE (lower is better)")
    ax.set_title("5 held-out regression tasks; mean over 3 model seeds; one fixed split")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(report_dir / "heldout_rmse_ratios.png", dpi=170)
    plt.close()
    LOG.info("FULL TEST TABLE:\n%s", summary.to_string(index=False))
    LOG.info("Descriptive aggregate (NOT proof of universal superiority):\n%s", aggregate.to_string())


def main():
    try:
        setup()

        for spec in DEVELOPMENT:
            data = get_data(spec, "development")
            del data
            gc.collect()
        search("global_defaults", DEVELOPMENT, "development", CONFIG["global_trials"])

        global_study = load_study("global_defaults")
        atomic_json(RUN / "shared_defaults.json", {
            "params": global_study.best_trial.user_attrs["model_params"],
            "trial": global_study.best_trial.number, "objective": global_study.best_value,
            "fingerprint": FINGERPRINT, "created_utc": now(),
        })
        for spec in HELDOUT:
            search(f"local_{spec['id']}", [spec], "heldout", CONFIG["local_trials"])
        selection = make_selection()
        evaluate(selection)
        report()
        write_status("complete", evaluation_results=len(list((RUN / "evaluation").glob("*.json"))))
        LOG.info("COMPLETE. Download analysis_results.zip and send it for interpretation.")
    except PauseRun as exc:
        write_status("paused", reason=str(exc))
        LOG.info("PAUSED, not completed. Save notebook outputs and resume: %s", exc)
    except BaseException as exc:
        if RUN.exists():
            atomic_json(RUN / "last_error.json", {"utc": now(), "type": type(exc).__name__,
                                                   "message": str(exc), "traceback": traceback.format_exc()})
            write_status("failed", reason=str(exc))
        raise
    finally:
        if STORAGE is not None:
            atomic_json(RUN / f"session-{time.time_ns()}.json", {
                "ended_utc": now(), "elapsed_seconds": time.monotonic() - START,
                "state": read_json(RUN / "status.json").get("state"),
                "note": "Wall time includes preparation/search/reporting; hard-killed sessions may have no final ledger.",
            })
        if STORAGE is not None:
            try:
                report()
            except Exception:
                LOG.exception("Report failed; preserving the training state separately.")
            try:
                checkpoint(force=True)
            except Exception:
                LOG.exception("Could not finish checkpoint; run/ still contains atomic progress.")
        print("\nOUTPUTS:", str(ARCHIVES))
        print("Send analysis_results.zip for analysis. Keep resume_checkpoint.zip to continue.")
        print("A local checkpoint survives another Kaggle session ONLY if notebook outputs were saved.")


if __name__ == "__main__":
    main()