"""CatBoost configurations for the partial reproduction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Configuration:
    name: str
    params: dict


# Holzmüller et al., NeurIPS 2024, Appendix C, Table C.3, regression column.
# CatBoost Python aliases: max_depth -> depth; n_estimators -> iterations;
# max_bin -> border_count. od_type/od_wait are supplied during validation.
PAPER_TD = Configuration(
    "paper_td",
    {
        "boosting_type": "Plain",
        "bootstrap_type": "Bernoulli",
        "depth": 9,
        "learning_rate": 0.09,
        "subsample": 0.9,
        "l2_leaf_reg": 1e-5,
        "random_strength": 0.0,
        "one_hot_max_size": 20,
        "leaf_estimation_iterations": 20,
        "border_count": 254,
    },
)

# A small, transparent search space for a new dataset-independent default.
# It is our experiment, not the authors' optimization procedure.
_GRID = [
    (depth, lr, l2)
    for depth in (5, 7, 9)
    for lr in (0.03, 0.09)
    for l2 in (1.0, 3.0)
]
# The first six cover every depth and learning rate for an affordable pilot.
_PILOT_FIRST = [
    (5, 0.03, 3.0), (5, 0.09, 1.0),
    (7, 0.03, 1.0), (7, 0.09, 3.0),
    (9, 0.03, 3.0), (9, 0.09, 1.0),
]
META_CANDIDATES = [
    Configuration(f"meta_{depth}_{lr}_{l2}", {
        "boosting_type": "Plain", "bootstrap_type": "Bernoulli",
        "subsample": 0.9, "depth": depth, "learning_rate": lr,
        "l2_leaf_reg": l2, "random_strength": 0.0,
        "one_hot_max_size": 20, "border_count": 254,
    })
    for depth, lr, l2 in _PILOT_FIRST + [item for item in _GRID if item not in _PILOT_FIRST]
]


def library_default() -> Configuration:
    # Shared iteration budget and CPU are set by the experiment runner.
    # Actual library defaults depend on the installed CatBoost version.
    return Configuration("library_default", {})

