"""Download and cache six UCI regression datasets without publishing the data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    uci_id: int
    role: str
    target: str
    categorical: tuple[str, ...] = ()


SPECS = (
    DatasetSpec("concrete", 165, "meta_train", "Concrete compressive strength"),
    DatasetSpec("airfoil", 291, "meta_train", "scaled-sound-pressure"),
    DatasetSpec("wine_quality", 186, "meta_train", "quality"),
    DatasetSpec("abalone", 1, "meta_train", "Rings", ("Sex",)),
    DatasetSpec("power_plant", 294, "meta_test", "PE"),
    DatasetSpec("superconductivity", 464, "meta_test", "critical_temp"),
)


def load_dataset(spec: DatasetSpec, cache_dir: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Fetch from UCI once, then read the local ignored CSV cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{spec.slug}.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        fetched = fetch_ucirepo(id=spec.uci_id)
        x = fetched.data.features.reset_index(drop=True).copy()
        targets = fetched.data.targets.reset_index(drop=True)
        if spec.target not in targets:
            raise ValueError(f"UCI {spec.uci_id}: expected target {spec.target!r}, got {list(targets)}")
        if spec.target in x:
            raise ValueError(f"Target {spec.target!r} also occurs in features")
        frame = x.assign(**{spec.target: targets[spec.target]})
        frame.to_csv(path, index=False)

    if spec.target not in frame:
        raise ValueError(f"Cached {path} is missing target {spec.target!r}")
    x = frame.drop(columns=spec.target).copy()
    y = pd.to_numeric(frame[spec.target], errors="coerce")
    for column in spec.categorical:
        if column not in x:
            raise ValueError(f"Missing categorical column {column!r} in {spec.slug}")
        x[column] = x[column].fillna("__MISSING__").astype(str)
    for column in x.columns.difference(spec.categorical):
        x[column] = pd.to_numeric(x[column], errors="coerce")
    valid = y.notna() & np.isfinite(y)
    x, y = x.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
    if x.isna().any().any():
        raise ValueError(f"{spec.slug} contains missing numerical features; handle them before benchmarking")
    if len(x) < 100 or y.nunique() < 2:
        raise ValueError(f"{spec.slug} is empty, too small, or has a constant target")
    return x, y, list(spec.categorical)

