# analysis/plots.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def distribution_box_jitter(df: pd.DataFrame, value_col: str, title: str, out_path: Path) -> None:
    if df.empty:
        return
    vals = df[value_col].astype(float).values

    plt.figure(figsize=(8, 4))
    plt.boxplot(vals, vert=True, showfliers=False)
    # jitter points
    x = np.ones_like(vals, dtype=float) + (np.random.rand(len(vals)) - 0.5) * 0.08
    plt.plot(x, vals, "o", markersize=3, alpha=0.7)

    plt.title(title)
    plt.ylabel(value_col)
    plt.xticks([1], [""])
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=config.FIG_DPI)
    plt.close()


def heatmap_rep_time(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: Path,
    time_bin_s: int = 3,
) -> None:
    if df.empty:
        return

    d = df.copy()
    d["rep"] = d["rep"].astype(int)
    d["tbin"] = (d["rel_t_s"].astype(float) // time_bin_s).astype(int)

    # pivot to matrix: rows=rep, cols=timebin
    mat = d.pivot_table(index="rep", columns="tbin", values=value_col, aggfunc="mean").sort_index()

    plt.figure(figsize=(10, 6))
    plt.imshow(mat.values, aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel(f"time bin ({time_bin_s}s)")
    plt.ylabel("rep")
    plt.colorbar(label=value_col)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=config.FIG_DPI)
    plt.close()


def line_idle_envelope(
    df_total: pd.DataFrame,
    df_idle: pd.DataFrame,
    title: str,
    out_path: Path,
    ylabel: str = "power (mW)",
) -> None:
    """
    Plot bucket power time-series for:
      - observed total power (from total energy counter)
      - estimated idle power (from idle energy counter)

    Expects dataframes with columns: rel_t_s, p_mW (and ideally sorted by rel_t_s).
    """
    if df_total.empty or df_idle.empty:
        return

    a = df_total.sort_values("rel_t_s").copy()
    b = df_idle.sort_values("rel_t_s").copy()

    # Merge on nearest rel_t_s to avoid assuming perfect timestamp alignment.
    # Prom scrape timestamps typically align, but this keeps it robust.
    a = a[["rel_t_s", "p_mW"]].rename(columns={"p_mW": "p_total_mW"})
    b = b[["rel_t_s", "p_mW"]].rename(columns={"p_mW": "p_idle_mW"})

    # asof merge requires sorted keys
    a = a.sort_values("rel_t_s")
    b = b.sort_values("rel_t_s")

    merged = pd.merge_asof(a, b, on="rel_t_s", direction="nearest", tolerance=1.5)
    merged = merged.dropna(subset=["p_total_mW", "p_idle_mW"])
    if merged.empty:
        return

    plt.figure(figsize=(10, 4))
    plt.plot(merged["rel_t_s"], merged["p_total_mW"], label="observed total")
    plt.plot(merged["rel_t_s"], merged["p_idle_mW"], label="estimated idle")

    plt.title(title)
    plt.xlabel("time in idle phase (s)")
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=config.FIG_DPI)
    plt.close()
