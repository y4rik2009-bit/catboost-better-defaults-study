"""Create a compact held-out dataset chart from a completed run."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["svg.fonttype"] = "none"


def main() -> None:
    path = Path("results/raw/metrics.csv")
    if not path.exists():
        raise SystemExit("Run run_experiment.py first")
    frame = pd.read_csv(path)
    frame = frame.loc[frame["role"] == "meta_test"].copy()
    labels = {
        "library_default": "Library default",
        "paper_td": "Paper tuned default",
        "transferred_default": "Our transferred default",
        "per_dataset_hpo": "Per-dataset HPO",
    }
    frame["method"] = frame.apply(
        lambda row: row["stage"] if row["stage"] in ("transferred_default", "per_dataset_hpo")
        else row["configuration"], axis=1)
    values = frame.groupby(["dataset", "method"])["test_nrmse"].mean().unstack()
    methods = [m for m in labels if m in values.columns]
    ax = values[methods].rename(columns=labels).plot.bar(
        figsize=(9, 5), rot=0, width=0.78,
        color=["#9CA3AF", "#3B82F6", "#10B981", "#F59E0B"][:len(methods)],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Test nRMSE (lower is better)")
    n_splits = frame["seed"].nunique()
    ax.set_title(f"Held-out OpenML datasets · {n_splits} split{'s' if n_splits != 1 else ''}")
    ax.legend(title="", loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    plt.tight_layout()
    output = Path("figures/heldout_nrmse.svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, format="svg")
    plt.savefig(output.with_suffix(".png"), dpi=180)
    print(output)


if __name__ == "__main__":
    main()

