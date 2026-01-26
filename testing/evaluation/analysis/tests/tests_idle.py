# analysis/tests/tests_idle.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config


@dataclass
class IdleTest:
    name: str = "idle"

    # -----------------------------
    # Metric wiring
    # -----------------------------
    @staticmethod
    def required_metrics() -> Dict[str, str]:
        # Energy counters only (+ residual usability flag).
        needed = [
            "system_total_energy_mj",
            # RAPL pkg (total + split)
            "rapl_pkg_total_energy_mj",
            "rapl_pkg_dynamic_energy_mj",
            "rapl_pkg_idle_energy_mj",
            # Residual
            "residual_total_energy_mj",
            "residual_window_usable",
            # GPU (total + split)
            "gpu_total_energy_mj",
            "gpu_dynamic_energy_mj",
            "gpu_idle_energy_mj",
        ]

        missing = [k for k in needed if k not in config.TYCHO_METRICS]
        if missing:
            raise RuntimeError(
                "Idle test metrics missing from config.TYCHO_METRICS: " + ", ".join(missing)
            )
        return {k: config.TYCHO_METRICS[k] for k in needed}

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _labels_key(x) -> str:
        # Stable identity for a Prometheus series (labelset).
        try:
            return json.dumps(x, sort_keys=True)
        except Exception:
            return str(x)

    @staticmethod
    def _filter_instance(df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce a single Prometheus instance if config exposes PROM_INSTANCE.
        This is critical: otherwise metrics from multiple servers appear and break diffs/rates.
        """
        inst = getattr(config, "PROM_INSTANCE", None)
        if not inst:
            return df

        if df.empty:
            return df

        # labels is expected to be dict-like (set by extract.py)
        def _get_inst(lbl) -> str:
            try:
                return (lbl or {}).get("instance", "")
            except Exception:
                return ""

        out = df.copy()
        out["_instance"] = out["labels"].apply(_get_inst)
        out = out[out["_instance"] == inst].drop(columns=["_instance"])
        return out

    @staticmethod
    def _counter_to_rate_power_W(df: pd.DataFrame, window_s: float = 10.0) -> pd.DataFrame:
        """
        Grafana-like: power(W) ≈ (E(t) - E(t-window)) / (t - (t-window)) / 1000.
        Uses merge_asof to find a sample at ~window_s earlier.
        Assumes df is one monotonic counter series (one labelset).
        """
        if df.empty:
            return df

        d = df.sort_values("ts").copy()
        d["ts"] = pd.to_datetime(d["ts"], utc=True)

        left = d[["ts", "value"]].rename(columns={"value": "E_now_mJ"})
        right = d[["ts", "value"]].rename(columns={"ts": "ts_prev", "value": "E_prev_mJ"})

        left["ts_target"] = left["ts"] - pd.to_timedelta(window_s, unit="s")
        right = right.sort_values("ts_prev")

        merged = pd.merge_asof(
            left.sort_values("ts_target"),
            right,
            left_on="ts_target",
            right_on="ts_prev",
            direction="backward",
            tolerance=pd.to_timedelta(window_s * 2, unit="s"),
        )

        merged = merged.dropna(subset=["E_prev_mJ", "ts_prev"]).copy()
        merged["dt_s"] = (merged["ts"] - merged["ts_prev"]).dt.total_seconds()

        # Guard counter resets/glitches
        merged["dE_mJ"] = (merged["E_now_mJ"] - merged["E_prev_mJ"]).clip(lower=0.0)

        # mJ/s = mW; divide by 1000 -> W
        merged["p_W"] = np.where(
            merged["dt_s"] > 0.0,
            (merged["dE_mJ"] / merged["dt_s"]) / 1000.0,
            0.0,
        )

        return merged[["ts", "p_W"]].sort_values("ts")

    @staticmethod
    def _aggregate_metric_power(
        s: pd.DataFrame,
        metric: str,
        t0: pd.Timestamp,
        window_s: float = 10.0,
    ) -> pd.DataFrame:
        """
        For a given metric name in the sliced series df:
        - split by labelset
        - compute rate-like power per series
        - sum across series at each timestamp
        Returns columns: ts, rel_t_s, p_W
        """
        g = s[s["metric"] == metric].copy()
        if g.empty:
            return pd.DataFrame(columns=["ts", "rel_t_s", "p_W"])

        g["labels_key"] = g["labels"].apply(IdleTest._labels_key)

        per_series: List[pd.DataFrame] = []
        for _, gs in g.groupby("labels_key"):
            gs = gs.sort_values("ts").copy()
            pw = IdleTest._counter_to_rate_power_W(gs, window_s=window_s)
            if pw.empty:
                continue
            per_series.append(pw)

        if not per_series:
            return pd.DataFrame(columns=["ts", "rel_t_s", "p_W"])

        pw_all = pd.concat(per_series, ignore_index=True)

        pw_sum = (
            pw_all.groupby("ts", as_index=False)["p_W"]
            .sum()
            .sort_values("ts")
        )
        pw_sum["rel_t_s"] = (pd.to_datetime(pw_sum["ts"], utc=True) - t0).dt.total_seconds()
        return pw_sum[["ts", "rel_t_s", "p_W"]]

    @staticmethod
    def _choose_rep_median_idle(reps: pd.DataFrame, idle_metric: str) -> int:
        """
        Choose a representative rep: closest to median idle baseline (p10).
        Uses p10 of the idle power distribution, which aligns with "lower envelope".
        """
        d = reps[reps["metric"] == idle_metric].copy()
        if d.empty:
            return int(reps["rep"].min())
        d = d.dropna(subset=["p10_p_W"])
        if d.empty:
            return int(reps["rep"].min())

        med = float(d["p10_p_W"].median())
        d["dist"] = (d["p10_p_W"] - med).abs()
        return int(d.sort_values(["dist", "rep"], kind="stable").iloc[0]["rep"])

    # -----------------------------
    # Plotting (self-contained, no dependency on plots.py)
    # -----------------------------
    @staticmethod
    def _plot_idle_dynamic_total(
        run_key: str,
        rep: int,
        component_title: str,
        idle_pw: pd.DataFrame,
        dyn_pw: pd.DataFrame,
        total_pw: pd.DataFrame,
        out_path: Path,
    ) -> None:
        """
        Grafana-like plot:
        - idle area
        - dynamic area stacked on idle
        - total line
        """
        import matplotlib.pyplot as plt

        if idle_pw.empty or dyn_pw.empty or total_pw.empty:
            return

        # Align on time using merge_asof on rel_t_s (robust to slight timestamp differences)
        a = total_pw.sort_values("rel_t_s")[["rel_t_s", "p_W"]].rename(columns={"p_W": "p_total_W"})
        b = idle_pw.sort_values("rel_t_s")[["rel_t_s", "p_W"]].rename(columns={"p_W": "p_idle_W"})
        c = dyn_pw.sort_values("rel_t_s")[["rel_t_s", "p_W"]].rename(columns={"p_W": "p_dyn_W"})

        m = pd.merge_asof(a, b, on="rel_t_s", direction="nearest", tolerance=2.0)
        m = pd.merge_asof(m, c, on="rel_t_s", direction="nearest", tolerance=2.0)
        m = m.dropna(subset=["p_total_W", "p_idle_W", "p_dyn_W"])
        if m.empty:
            return

        x = m["rel_t_s"].astype(float).to_numpy()
        p_idle = m["p_idle_W"].astype(float).to_numpy()
        p_dyn = m["p_dyn_W"].astype(float).to_numpy()
        p_total = m["p_total_W"].astype(float).to_numpy()

        plt.figure(figsize=(10, 4))
        # Areas (no explicit colors set; matplotlib defaults)
        plt.fill_between(x, 0.0, p_idle, alpha=0.25, label="idle")
        plt.fill_between(x, p_idle, p_idle + p_dyn, alpha=0.25, label="dynamic")
        plt.plot(x, p_total, linewidth=1.5, label="total")

        plt.title(f"{run_key} {component_title} (rep {rep}): idle + dynamic = total")
        plt.xlabel("time in idle phase (s)")
        plt.ylabel("power (W)")
        plt.legend(loc="best")
        plt.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=getattr(config, "FIG_DPI", 150))
        plt.close()

    # -----------------------------
    # Main entrypoint
    # -----------------------------
    @staticmethod
    def run(run_key: str, phases_df: pd.DataFrame, series_df: pd.DataFrame, out_dir: Path) -> None:
        idle_phases = phases_df[phases_df["phase"] == "idle_baseline_start"].copy()
        if idle_phases.empty:
            raise RuntimeError("No idle_baseline_start phases found in phases_df.")

        figs_dir = out_dir / "figs" / "idle"
        figs_dir.mkdir(parents=True, exist_ok=True)

        # Enforce instance restriction early (critical)
        series_df = IdleTest._filter_instance(series_df)

        # Flag metrics (gauges)
        FLAG_METRICS = {"residual_window_usable"}

        # Output rows
        bucket_rows: List[dict] = []
        rep_rows: List[dict] = []

        # Use the same smoothing window as Grafana.
        RATE_WINDOW_S = 10.0

        for _, ph in idle_phases.iterrows():
            rep = int(ph["rep"])
            t_start = ph["start_utc"] + pd.Timedelta(seconds=config.IDLE_SETTLE_SEC)
            t_end = ph["end_utc"] - pd.Timedelta(seconds=config.IDLE_GUARD_SEC)
            if t_end <= t_start:
                continue

            # Slice to rep + phase + time window
            s = series_df[
                (series_df["rep"] == rep)
                & (series_df["phase"] == "idle_baseline_start")
                & (series_df["ts"] >= t_start)
                & (series_df["ts"] <= t_end)
            ].copy()
            if s.empty:
                continue

            t0 = pd.to_datetime(t_start, utc=True)

            # Residual usability flag sanity
            flag = s[s["metric"].isin(FLAG_METRICS)]
            usable_frac = float(flag["value"].astype(float).mean()) if not flag.empty else float("nan")

            # Counter metrics only
            counters = s[~s["metric"].isin(FLAG_METRICS)].copy()
            if counters.empty:
                continue

            # Compute and store per-metric aggregated power time series
            for metric in sorted(counters["metric"].unique()):
                pw = IdleTest._aggregate_metric_power(
                    counters,
                    metric=metric,
                    t0=t0,
                    window_s=RATE_WINDOW_S,
                )
                if pw.empty:
                    continue

                # Persist per-sample rows
                for _, r in pw.iterrows():
                    bucket_rows.append({
                        "run_key": run_key,
                        "rep": rep,
                        "metric": metric,
                        "ts": r["ts"],
                        "rel_t_s": float(r["rel_t_s"]),
                        "p_W": float(r["p_W"]),
                        "usable_frac_mean": usable_frac,
                    })

                # Per-rep summary stats on power distribution
                p = pw["p_W"].astype(float)
                p = p[p > 0.0]  # drop zeros if any
                if len(p) == 0:
                    mean_p = std_p = p10 = p50 = p90 = 0.0
                else:
                    mean_p = float(p.mean())
                    std_p = float(p.std(ddof=1)) if len(p) > 1 else 0.0
                    p10 = float(p.quantile(0.10))
                    p50 = float(p.quantile(0.50))
                    p90 = float(p.quantile(0.90))

                rep_rows.append({
                    "run_key": run_key,
                    "rep": rep,
                    "metric": metric,
                    "mean_p_W": mean_p,
                    "std_p_W": std_p,
                    "p10_p_W": p10,
                    "p50_p_W": p50,
                    "p90_p_W": p90,
                    "n_samples": int(len(pw)),
                    "usable_frac_mean": usable_frac,
                    "rate_window_s": RATE_WINDOW_S,
                })

        buckets = pd.DataFrame(bucket_rows)
        reps = pd.DataFrame(rep_rows)

        # Persist
        (out_dir / "idle_buckets.parquet").parent.mkdir(parents=True, exist_ok=True)
        buckets.to_parquet(out_dir / "idle_buckets.parquet", index=False)
        reps.to_parquet(out_dir / "results_idle.parquet", index=False)

        if reps.empty or buckets.empty:
            raise RuntimeError("Idle test produced no results. Check extraction / filters / metric mapping.")

        # -----------------------------
        # Representative line diagrams (RAPL pkg + GPU), Grafana-like
        # -----------------------------
        rep_sel = IdleTest._choose_rep_median_idle(reps, "rapl_pkg_idle_energy_mj")

        def _get(rep: int, metric: str) -> pd.DataFrame:
            return buckets[(buckets["rep"] == rep) & (buckets["metric"] == metric)].copy()

        # RAPL pkg: idle + dynamic = total
        IdleTest._plot_idle_dynamic_total(
            run_key=run_key,
            rep=rep_sel,
            component_title="RAPL PKG Power",
            idle_pw=_get(rep_sel, "rapl_pkg_idle_energy_mj"),
            dyn_pw=_get(rep_sel, "rapl_pkg_dynamic_energy_mj"),
            total_pw=_get(rep_sel, "rapl_pkg_total_energy_mj"),
            out_path=figs_dir / "line_rapl_pkg_idle_dynamic_total.png",
        )

        # GPU (summed across GPUs): idle + dynamic = total
        IdleTest._plot_idle_dynamic_total(
            run_key=run_key,
            rep=rep_sel,
            component_title="GPU Power",
            idle_pw=_get(rep_sel, "gpu_idle_energy_mj"),
            dyn_pw=_get(rep_sel, "gpu_dynamic_energy_mj"),
            total_pw=_get(rep_sel, "gpu_total_energy_mj"),
            out_path=figs_dir / "line_gpu_idle_dynamic_total.png",
        )


        # -----------------------------
        # Most important statistical test for idle:
        # baseline stability across repetitions (lower envelope)
        # -----------------------------
        import matplotlib.pyplot as plt

        def _baseline_boxplot(metric: str, title: str, out_name: str) -> None:
            d = reps[reps["metric"] == metric].copy()
            if d.empty:
                return
            d = d.sort_values("rep", kind="stable")
            y = d["p10_p_W"].astype(float).to_numpy()
            y = y[~np.isnan(y)]
            if y.size == 0:
                return

            plt.figure(figsize=(6.5, 3.5))
            # boxplot (single category) + per-rep dots for transparency
            plt.boxplot(y, vert=True, widths=0.4, showfliers=False)

            # overlay dots (one per rep)
            x = np.ones_like(y, dtype=float)
            # small deterministic jitter based on index (no RNG)
            jitter = (np.arange(len(y)) % 11 - 5) * 0.01
            plt.plot(x + jitter, y, marker="o", linestyle="none", markersize=3, alpha=0.7)

            plt.title(title)
            plt.ylabel("idle baseline power (W)  [p10 over time]")
            plt.xticks([1], ["50 reps"])
            plt.tight_layout()

            out_path = figs_dir / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=getattr(config, "FIG_DPI", 150))
            plt.close()

        # RAPL pkg idle baseline stability
        _baseline_boxplot(
            metric="rapl_pkg_idle_energy_mj",
            title=f"{run_key} idle baseline stability: RAPL pkg",
            out_name="dist_idle_baseline_p10_rapl_pkg.png",
        )

        # GPU idle baseline stability
        _baseline_boxplot(
            metric="gpu_idle_energy_mj",
            title=f"{run_key} idle baseline stability: GPU",
            out_name="dist_idle_baseline_p10_gpu.png",
        )



run_idle_test = IdleTest()
