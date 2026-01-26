from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config


@dataclass
class WorkloadEnergyTest:
    """
    Generic test:
      - builds per-workload mean power (W) from workload energy counters (mJ)
      - keeps idle and dynamic separate (no totals)
      - one figure per (test phase) and per CPU domain if configured
    """
    name: str
    target_phase: str

    # For CPU tests we may loop over domains. For GPU tests: domains=None.
    cpu_domains: Optional[List[str]] = None

    # Metric logical names in config.TYCHO_METRICS.
    # For CPU, these will be formatted with domain (see _metric_key()).
    metric_prefix_idle: str = ""     # e.g. "workload_rapl_{domain}_idle_energy_mj"
    metric_prefix_dyn: str = ""      # e.g. "workload_rapl_{domain}_dynamic_energy_mj"

    # For GPU tests (no domain formatting):
    metric_idle: str = ""            # e.g. "workload_gpu_idle_energy_mj"
    metric_dyn: str = ""             # e.g. "workload_gpu_dynamic_energy_mj"

    # Minimum number of workloads per rep (2 or 3)
    min_workloads_per_rep: int = 2

    # -----------------------------
    # Metric wiring
    # -----------------------------
    def required_metrics(self) -> Dict[str, str]:
        if self.cpu_domains:
            needed = []
            for d in self.cpu_domains:
                needed.append(self.metric_prefix_idle.format(domain=d))
                needed.append(self.metric_prefix_dyn.format(domain=d))
        else:
            needed = [self.metric_idle, self.metric_dyn]

        missing = [k for k in needed if k not in config.TYCHO_METRICS]
        if missing:
            raise RuntimeError(
                f"{self.name} metrics missing from config.TYCHO_METRICS: " + ", ".join(missing)
            )
        return {k: config.TYCHO_METRICS[k] for k in needed}

    # -----------------------------
    # Helpers
    # -----------------------------
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
        using consecutive diffs.
        """
        if df.empty:
            return pd.DataFrame(columns=["ts", "p_W"])

        d = df.sort_values("ts").copy()
        d["ts"] = pd.to_datetime(d["ts"], utc=True)

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
          - one logical metric
          - one pod label
        Then sum counters across all remaining label splits per timestamp.
        Returns: ts, value (summed counter in mJ)
        """
        if s.empty:
            return pd.DataFrame(columns=["ts", "value"])

        g = s[s["metric"] == metric].copy()
        if g.empty:
            return pd.DataFrame(columns=["ts", "value"])

        g["_pod"] = g["labels"].apply(lambda x: WorkloadEnergyTest._get_label(x, "pod"))
        g = g[g["_pod"] == pod].drop(columns=["_pod"])
        if g.empty:
            return pd.DataFrame(columns=["ts", "value"])

        out = (
            g.groupby("ts", as_index=False)["value"]
            .sum()
            .sort_values("ts")
        )
        out["ts"] = pd.to_datetime(out["ts"], utc=True)
        return out

    @staticmethod
    def _power_stats(pw: pd.DataFrame) -> Tuple[float, float, int]:
        if pw.empty:
            return 0.0, 0.0, 0
        p = pw["p_W"].astype(float)
        p = p[np.isfinite(p)]
        if len(p) == 0:
            return 0.0, 0.0, 0
        mean_p = float(p.mean())
        std_p = float(p.std(ddof=1)) if len(p) > 1 else 0.0
        return mean_p, std_p, int(len(p))

    @staticmethod
    def _plot_scatter_multi_series(df: pd.DataFrame, out_path: Path, title: str, ylabel: str) -> None:
        import matplotlib.pyplot as plt

        if df.empty:
            return

        d = df.sort_values(["rep", "workload", "kind"], kind="stable").copy()
        reps = sorted(d["rep"].unique().tolist())

        # series key = (workload, kind)
        series_keys = (
            d[["workload", "kind"]]
            .drop_duplicates()
            .sort_values(["workload", "kind"], kind="stable")
        )
        keys = [(str(r["workload"]), str(r["kind"])) for _, r in series_keys.iterrows()]

        # Horizontal dodge: one offset per series
        max_span = 0.32
        n = len(keys)
        if n <= 1:
            offsets = {keys[0]: 0.0}
        else:
            grid = np.linspace(-max_span, max_span, n)
            offsets = {keys[i]: float(grid[i]) for i in range(n)}

        plt.figure(figsize=(10, 4))

        for (workload, kind), g in d.groupby(["workload", "kind"], sort=True):
            k = (str(workload), str(kind))
            off = offsets.get(k, 0.0)

            x = g["rep"].astype(float).to_numpy() + off
            y = g["mean_p_W"].astype(float).to_numpy()
            e = g["std_p_W"].astype(float).to_numpy()

            plt.errorbar(
                x, y, yerr=e,
                fmt="o", markersize=4, capsize=2,
                label=f"{workload} ({kind})",
            )

        plt.title(title)
        plt.xlabel("Test Repetition")
        plt.ylabel(ylabel)
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

    def _metric_key(self, kind: str, domain: Optional[str]) -> str:
        if domain is None:
            return self.metric_idle if kind == "idle" else self.metric_dyn
        else:
            if kind == "idle":
                return self.metric_prefix_idle.format(domain=domain)
            else:
                return self.metric_prefix_dyn.format(domain=domain)

    # -----------------------------
    # Main entrypoint
    # -----------------------------
    def run(
        self,
        run_key: str,
        phases_df: pd.DataFrame,
        series_df: pd.DataFrame,
        out_dir: Path,
    ) -> None:
        from main import log

        win_path = out_dir / "workload_set_windows.parquet"
        if not win_path.exists():
            raise RuntimeError(
                f"Missing windows file: {win_path}. "
                f"Did main.py build workload_set windows?"
            )

        windows = pd.read_parquet(win_path)
        if windows.empty:
            raise RuntimeError("workload_set_windows.parquet is empty.")

        # Filter to this test's phase
        windows = windows[windows["parent_phase"] == self.target_phase].copy()
        if windows.empty:
            raise RuntimeError(f"No windows for target_phase='{self.target_phase}' found in workload_set_windows.parquet")

        # Enforce instance restriction early
        series_df = WorkloadEnergyTest._filter_instance(series_df)

        settle = int(getattr(config, "CPU_DISCRIM_SETTLE_SEC", 10))
        guard = int(getattr(config, "CPU_DISCRIM_GUARD_SEC", 10))

        domains = self.cpu_domains[:] if self.cpu_domains else [None]

        figs_dir = out_dir / "figs" / self.name
        figs_dir.mkdir(parents=True, exist_ok=True)

        all_rows: List[dict] = []

        for domain in domains:
            rows: List[dict] = []

            for _, w in windows.iterrows():
                rep = int(w["rep"])
                workload = str(w["workload"])
                pod = str(w["pod"])
                phase = str(w["phase"])

                t_start = pd.to_datetime(w["start_utc"], utc=True) + pd.Timedelta(seconds=settle)
                t_end = pd.to_datetime(w["end_utc"], utc=True) - pd.Timedelta(seconds=guard)
                if t_end <= t_start:
                    continue

                s = series_df[
                    (series_df["rep"] == rep)
                    & (series_df["phase"] == phase)
                    & (series_df["ts"] >= t_start)
                    & (series_df["ts"] <= t_end)
                ].copy()
                if s.empty:
                    continue

                for kind in ("idle", "dynamic"):
                    metric = self._metric_key(kind, domain)

                    ctr = WorkloadEnergyTest._sum_pod_counter_series(s, metric, pod)
                    if ctr.empty:
                        continue

                    pw = WorkloadEnergyTest._counter_to_interval_power_W(ctr)
                    if pw.empty:
                        continue

                    mean_p, std_p, n = WorkloadEnergyTest._power_stats(pw)

                    rows.append(
                        {
                            "run_key": run_key,
                            "rep": rep,
                            "parent_phase": self.target_phase,
                            "phase": phase,
                            "workload": workload,
                            "pod": pod,
                            "kind": kind,
                            "domain": "" if domain is None else str(domain),
                            "start_utc": t_start,
                            "end_utc": t_end,
                            "mean_p_W": mean_p,
                            "std_p_W": std_p,
                            "n_samples": n,
                        }
                    )

            res = pd.DataFrame(rows)

            # Validate min workload count per rep (after filtering)
            if not res.empty:
                g = res.groupby("rep")["workload"].nunique()
                bad = g[g < int(self.min_workloads_per_rep)]
                if not bad.empty:
                    log(f"[{self.name}] WARNING: some reps have <{self.min_workloads_per_rep} workloads after filtering: {bad.to_dict()}")

            # Write results per domain (or single file for GPU)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            if domain is None:
                out_parquet = out_dir / f"results_{self.name}.parquet"
            else:
                out_parquet = out_dir / f"results_{self.name}__{domain}.parquet"

            res.to_parquet(out_parquet, index=False)
            all_rows.append(res)

            if res.empty:
                continue

            server_state = WorkloadEnergyTest._server_state_from_run_key(run_key)

            # Titles/labels: simple and editable
            if domain is None:
                title = f"{self.target_phase}: workload power (idle vs dynamic) on {server_state} server"
                fname = f"scatter_{self.target_phase}__{ts}.png"
                ylabel = "mean workload power (W)"
            else:
                title = f"{self.target_phase}: RAPL {domain} workload power (idle vs dynamic) on {server_state} server"
                fname = f"scatter_{self.target_phase}__{domain}__{ts}.png"
                ylabel = f"mean workload power (W) from RAPL {domain}"

            WorkloadEnergyTest._plot_scatter_multi_series(
                res,
                out_path=figs_dir / fname,
                title=title,
                ylabel=ylabel,
            )

            log(f"[{self.name}] wrote {out_parquet}")

        log(f"[{self.name}] figures in: {figs_dir}")


# -----------------------------
# Concrete test instances
# -----------------------------

CPU_DOMAINS = getattr(config, "CPU_DOMAINS", ["core"])

run_cpu_busy_vs_noop_idle_share_test = WorkloadEnergyTest(
    name="cpu_busy_vs_noop_idle_share",
    target_phase="cpu_busy_vs_noop_idle_share",
    cpu_domains=CPU_DOMAINS,
    metric_prefix_idle="workload_rapl_{domain}_idle_energy_mj",
    metric_prefix_dyn="workload_rapl_{domain}_dynamic_energy_mj",
    min_workloads_per_rep=2,
)

run_gpu_concurrent_2pods_test = WorkloadEnergyTest(
    name="gpu_concurrent_2pods",
    target_phase="gpu_concurrent_2pods",
    cpu_domains=None,
    metric_idle="workload_gpu_idle_energy_mj",
    metric_dyn="workload_gpu_dynamic_energy_mj",
    min_workloads_per_rep=2,
)

run_gpu_concurrent_3pods_test = WorkloadEnergyTest(
    name="gpu_concurrent_3pods",
    target_phase="gpu_concurrent_3pods",
    cpu_domains=None,
    metric_idle="workload_gpu_idle_energy_mj",
    metric_dyn="workload_gpu_dynamic_energy_mj",
    min_workloads_per_rep=3,
)
