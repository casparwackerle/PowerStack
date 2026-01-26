from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

import config


@dataclass
class CpuDiscriminationTest:
    name: str = "cpu_discrimination"

    # -----------------------------
    # Metric wiring
    # -----------------------------
    @staticmethod
    def required_metrics() -> Dict[str, str]:
        needed = [
            "workload_rapl_core_idle_energy_mj",
            "workload_rapl_core_dynamic_energy_mj",
        ]
        missing = [k for k in needed if k not in config.TYCHO_METRICS]
        if missing:
            raise RuntimeError(
                "CPU discrimination test metrics missing from config.TYCHO_METRICS: " + ", ".join(missing)
            )
        return {k: config.TYCHO_METRICS[k] for k in needed}

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _labels_key(x) -> str:
        try:
            return json.dumps(x, sort_keys=True)
        except Exception:
            return str(x)

    @staticmethod
    def _filter_instance(df: pd.DataFrame) -> pd.DataFrame:
        inst = getattr(config, "PROM_INSTANCE", None)
        if not inst or df.empty:
            return df

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
    def _get_label(lbl, key: str) -> str:
        try:
            return (lbl or {}).get(key, "")
        except Exception:
            return ""

    @staticmethod
    def _counter_to_interval_power_W(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a monotonic energy counter (mJ) to per-interval power samples (W),
        using consecutive diffs (no 10s window smoothing).
        """
        if df.empty:
            return pd.DataFrame(columns=["ts", "p_W"])

        d = df.sort_values("ts").copy()
        d["ts"] = pd.to_datetime(d["ts"], utc=True)

        # Consecutive diffs
        d["E_prev_mJ"] = d["value"].shift(1)
        d["ts_prev"] = d["ts"].shift(1)
        d = d.dropna(subset=["E_prev_mJ", "ts_prev"]).copy()

        d["dt_s"] = (d["ts"] - d["ts_prev"]).dt.total_seconds()
        d = d[d["dt_s"] > 0].copy()

        # Counter reset/glitch guard
        d["dE_mJ"] = (d["value"] - d["E_prev_mJ"]).clip(lower=0.0)

        # mJ/s = mW, divide by 1000 => W
        d["p_W"] = (d["dE_mJ"] / d["dt_s"]) / 1000.0

        return d[["ts", "p_W"]].sort_values("ts")

    @staticmethod
    def _sum_pod_counter_series(
        s: pd.DataFrame,
        metric: str,
        pod: str,
    ) -> pd.DataFrame:
        """
        Filter extracted long-form series to:
          - one logical metric (idle or dynamic)
          - one pod label
        Then sum counters across all remaining label splits per timestamp.
        Returns columns: ts, value (summed counter in mJ)
        """
        if s.empty:
            return pd.DataFrame(columns=["ts", "value"])

        g = s[s["metric"] == metric].copy()
        if g.empty:
            return pd.DataFrame(columns=["ts", "value"])

        g["_pod"] = g["labels"].apply(lambda x: CpuDiscriminationTest._get_label(x, "pod"))
        g = g[g["_pod"] == pod].drop(columns=["_pod"])
        if g.empty:
            return pd.DataFrame(columns=["ts", "value"])

        # Sum across series at each scrape timestamp
        out = (
            g.groupby("ts", as_index=False)["value"]
            .sum()
            .sort_values("ts")
        )
        out["ts"] = pd.to_datetime(out["ts"], utc=True)
        return out

    @staticmethod
    def _power_stats(total_pw: pd.DataFrame) -> Tuple[float, float, int]:
        """
        Return mean, std (ddof=1), n for total power samples.
        """
        if total_pw.empty:
            return 0.0, 0.0, 0
        p = total_pw["p_W"].astype(float)
        p = p[np.isfinite(p)]
        if len(p) == 0:
            return 0.0, 0.0, 0
        mean_p = float(p.mean())
        std_p = float(p.std(ddof=1)) if len(p) > 1 else 0.0
        return mean_p, std_p, int(len(p))

    # -----------------------------
    # Plot
    # -----------------------------
    @staticmethod
    def _plot_scatter_two_rows(df: pd.DataFrame, out_path: Path, title: str) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        if df.empty:
            return

        d = df.sort_values(["rep", "workload"], kind="stable").copy()
        reps = sorted(d["rep"].unique().tolist())
        workloads = sorted(d["workload"].unique().tolist())

        # Small horizontal dodge so overlapping workloads become visible.
        # Keep offsets within +/- max_span around the integer rep position.
        max_span = 0.30
        n = len(workloads)
        if n <= 1:
            offsets = {workloads[0]: 0.0} if n == 1 else {}
        else:
            grid = np.linspace(-max_span, max_span, n)
            offsets = {w: float(grid[i]) for i, w in enumerate(workloads)}

        plt.figure(figsize=(10, 4))

        for workload, g in d.groupby("workload", sort=True):
            off = offsets.get(str(workload), 0.0)

            x = g["rep"].astype(float).to_numpy() + off
            y = g["mean_p_W"].astype(float).to_numpy()
            e = g["std_p_W"].astype(float).to_numpy()

            plt.errorbar(
                x, y, yerr=e,
                fmt="o", markersize=4, capsize=2,
                label=str(workload),
            )

        plt.title(title)
        plt.xlabel("Test Repetition")
        plt.ylabel("mean total CPU power (W) derived from energy counters")

        # Tick labels stay on integer reps (centered), even though points are dodged.
        plt.xticks(reps)

        plt.legend(loc="best")
        plt.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=getattr(config, "FIG_DPI", 150))
        plt.close()

    @staticmethod
    def _server_state_from_run_key(run_key: str) -> str:
        rk = run_key.lower()
        if "idle" in rk:
            return "idle"
        if "busy" in rk:
            return "busy"
        return "unknown"
    
    # -----------------------------
    # Main entrypoint
    # -----------------------------
    
    @staticmethod
    def run(
        run_key: str,
        phases_df: pd.DataFrame,
        series_df: pd.DataFrame,
        out_dir: Path,
    ) -> None:
        # Minimal logging hook (only a couple lines)
        from main import log

        # CPU discrimination windows were written by main.py as cpu_discrimination_windows.parquet
        win_path = out_dir / "cpu_discrimination_windows.parquet"
        if not win_path.exists():
            raise RuntimeError(f"Missing windows file: {win_path}. Did main.py build subrun windows?")

        windows = pd.read_parquet(win_path)
        if windows.empty:
            raise RuntimeError("cpu_discrimination_windows.parquet is empty.")

        # Enforce instance restriction early
        series_df = CpuDiscriminationTest._filter_instance(series_df)

        # Trim policy
        settle = getattr(config, "CPU_DISCRIM_SETTLE_SEC", 10)
        guard = getattr(config, "CPU_DISCRIM_GUARD_SEC", 10)

        idle_metric = "workload_rapl_core_idle_energy_mj"
        dyn_metric = "workload_rapl_core_dynamic_energy_mj"

        rows: List[dict] = []

        # Workload name is stored in windows["workload"], phase name encodes it too.
        for _, w in windows.iterrows():
            rep = int(w["rep"])
            workload = str(w["workload"])
            pod = str(w["pod"])
            phase = str(w["phase"])

            t_start = pd.to_datetime(w["start_utc"], utc=True) + pd.Timedelta(seconds=settle)
            t_end = pd.to_datetime(w["end_utc"], utc=True) - pd.Timedelta(seconds=guard)
            if t_end <= t_start:
                continue

            # Slice extracted series to this rep+phase and trimmed time window
            s = series_df[
                (series_df["rep"] == rep)
                & (series_df["phase"] == phase)
                & (series_df["ts"] >= t_start)
                & (series_df["ts"] <= t_end)
            ].copy()

            if s.empty:
                continue

            # Build summed counter series per pod for idle/dynamic
            idle_ctr = CpuDiscriminationTest._sum_pod_counter_series(s, idle_metric, pod)
            dyn_ctr = CpuDiscriminationTest._sum_pod_counter_series(s, dyn_metric, pod)

            if idle_ctr.empty or dyn_ctr.empty:
                continue

            # Convert to per-interval power samples
            idle_pw = CpuDiscriminationTest._counter_to_interval_power_W(idle_ctr)
            dyn_pw = CpuDiscriminationTest._counter_to_interval_power_W(dyn_ctr)

            if idle_pw.empty or dyn_pw.empty:
                continue

            # Align by nearest timestamps and sum to total
            a = idle_pw.sort_values("ts").rename(columns={"p_W": "p_idle_W"})
            b = dyn_pw.sort_values("ts").rename(columns={"p_W": "p_dyn_W"})

            merged = pd.merge_asof(
                a,
                b,
                on="ts",
                direction="nearest",
                tolerance=pd.to_timedelta(2.0, unit="s"),
            ).dropna(subset=["p_idle_W", "p_dyn_W"])

            if merged.empty:
                continue

            merged["p_W"] = merged["p_idle_W"].astype(float) + merged["p_dyn_W"].astype(float)
            total_pw = merged[["ts", "p_W"]].copy()

            mean_p, std_p, n = CpuDiscriminationTest._power_stats(total_pw)

            rows.append(
                {
                    "run_key": run_key,
                    "rep": rep,
                    "workload": workload,
                    "pod": pod,
                    "phase": phase,
                    "start_utc": t_start,
                    "end_utc": t_end,
                    "mean_p_W": mean_p,
                    "std_p_W": std_p,
                    "n_samples": n,
                }
            )

        res = pd.DataFrame(rows)
        out_parquet = out_dir / "results_cpu_discrimination.parquet"
        res.to_parquet(out_parquet, index=False)

        if res.empty:
            raise RuntimeError(
                "CPU discrimination produced no results. "
                "Likely causes: pod label mismatch, missing workload metrics, or extraction windows misbuilt."
            )

        # Minimal sanity: require 2 workloads per rep
        g = res.groupby("rep")["workload"].nunique()
        bad = g[g < 2]
        if not bad.empty:
            log(f"[cpu_discrimination] WARNING: some reps have <2 workloads after filtering: {bad.to_dict()}")

        figs_dir = out_dir / "figs" / "cpu_discrimination"
        figs_dir.mkdir(parents=True, exist_ok=True)

        # One plot per parent master phase if available, otherwise a single combined plot.
        if "parent_phase" in res.columns:
            groups = res.groupby("parent_phase", sort=True)
        else:
            groups = [("all", res)]

        for parent_phase, g in groups:
            safe = str(parent_phase).replace("/", "_")
            server_state = CpuDiscriminationTest._server_state_from_run_key(run_key)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            CpuDiscriminationTest._plot_scatter_two_rows(
                g,
                out_path=figs_dir / f"scatter_mean_total_power_core__{safe}__{ts}.png",
                title = (
                    f"RAPL core energy consumption of different workload types "
                    f"on an {server_state} server"
                )
            )


        log(f"[cpu_discrimination] wrote {out_parquet}")
        log(f"[cpu_discrimination] figures in: {figs_dir}")

            
run_cpu_discrimination_test = CpuDiscriminationTest()
