from __future__ import annotations

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
      - builds per-series mean power (W) from workload energy counters (mJ)
      - CPU tests: per-pod series (workload) and optional per-domain loop
      - GPU tests: aggregate by gpu_uuid, and for 3pods split into (gpu_uuid, pod) series when a GPU multitasks
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
            needed: List[str] = []
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
    def _counter_to_window_mean_power_W(df: pd.DataFrame):
        """
        Robust window-level mean power estimate from a monotonic energy counter (mJ).

        Returns:
          (mean_power_W, n_samples_used)

        mean_power_W = (E_last - E_first) / (t_last - t_first) / 1000
        """
        if df.empty:
            return None, 0

        d = df.sort_values("ts").copy()
        d["ts"] = pd.to_datetime(d["ts"], utc=True)

        if len(d) < 2:
            return None, int(len(d))

        t0 = d["ts"].iloc[0]
        t1 = d["ts"].iloc[-1]
        dt_s = (t1 - t0).total_seconds()
        if dt_s <= 0:
            return None, int(len(d))

        E0 = float(d["value"].iloc[0])
        E1 = float(d["value"].iloc[-1])
        dE_mJ = max(0.0, E1 - E0)  # guard counter reset/glitch

        p_W = (dE_mJ / dt_s) / 1000.0
        return p_W, int(len(d))


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
    def _relabel_gpu3_split_single_roles(res: pd.DataFrame) -> pd.DataFrame:
        """
        For gpu_concurrent_3pods: rewrite 'series' to stable role-based labels.

        Per rep:
          - one GPU runs 2 pods (split GPU) -> emits two series: (split task 1/2)
          - the other GPU runs 1 pod (single GPU) -> emits one series: (single)

        Requires columns: rep, gpu_uuid, pod, series (existing), kind.
        """
        if res.empty:
            return res

        needed = {"rep", "gpu_uuid", "pod", "kind", "series"}
        missing = [c for c in needed if c not in res.columns]
        if missing:
            raise RuntimeError(f"_relabel_gpu3_split_single_roles: missing columns: {missing}")

        out = res.copy()

        new_series: List[str] = []

        # iterate in row order to keep alignment
        for rep, g_rep in out.groupby("rep", sort=True):
            # count rows per gpu_uuid for this rep
            counts = g_rep["gpu_uuid"].astype(str).value_counts()

            # identify split vs single uuid
            split_uuids = counts[counts == 2].index.tolist()
            single_uuids = counts[counts == 1].index.tolist()

            # If the rep is malformed (dead pods etc.), fall back to existing labels.
            # This keeps the plot generating instead of exploding.
            if len(split_uuids) != 1 or len(single_uuids) != 1:
                for _ in range(len(g_rep)):
                    new_series.append("")  # placeholder, will be filled below
                continue

            split_uuid = str(split_uuids[0])
            single_uuid = str(single_uuids[0])

            # Map uuid -> friendly GPU name (model)
            split_name = WorkloadEnergyTest._gpu_friendly_name(split_uuid)
            single_name = WorkloadEnergyTest._gpu_friendly_name(single_uuid)

            # Determine stable task ordering for the split GPU by pod name
            split_pods = (
                g_rep[g_rep["gpu_uuid"].astype(str) == split_uuid]["pod"]
                .astype(str)
                .unique()
                .tolist()
            )
            split_pods = sorted(split_pods)
            pod_to_slot = {p: i + 1 for i, p in enumerate(split_pods)}  # 1..2

            # Assign new series per row in this rep (preserve row order of g_rep)
            for _, r in g_rep.iterrows():
                uuid = str(r["gpu_uuid"])
                pod = str(r["pod"])

                if uuid == split_uuid:
                    slot = pod_to_slot.get(pod, 1)
                    new_series.append(f"{split_name} (split task {slot})")
                elif uuid == single_uuid:
                    new_series.append(f"{single_name} (single)")
                else:
                    # should not happen, but keep safe
                    new_series.append(str(r["series"]))

        # Fill placeholders for malformed reps using original series
        # (we inserted "" placeholders above)
        orig_series = out["series"].astype(str).tolist()
        fixed_series: List[str] = []
        j = 0
        for s in new_series:
            if s == "":
                fixed_series.append(orig_series[j])
            else:
                fixed_series.append(s)
            j += 1

        out["series"] = fixed_series
        return out


    @staticmethod
    def _plot_scatter_multi_series(df: pd.DataFrame, out_path: Path, title: str, ylabel: str) -> None:
        import matplotlib.pyplot as plt

        if df.empty:
            return

        d = df.sort_values(["rep", "series", "kind"], kind="stable").copy()
        reps = sorted(d["rep"].unique().tolist())

        # series key = (series, kind)
        series_keys = (
            d[["series", "kind"]]
            .drop_duplicates()
            .sort_values(["series", "kind"], kind="stable")
        )
        keys = [(str(r["series"]), str(r["kind"])) for _, r in series_keys.iterrows()]

        # Horizontal dodge: one offset per series
        max_span = 0.32
        n = len(keys)
        if n <= 1:
            offsets = {keys[0]: 0.0}
        else:
            grid = np.linspace(-max_span, max_span, n)
            offsets = {keys[i]: float(grid[i]) for i in range(n)}

        plt.figure(figsize=(10, 4))

        for (series, kind), g in d.groupby(["series", "kind"], sort=True):
            k = (str(series), str(kind))
            off = offsets.get(k, 0.0)

            x = g["rep"].astype(float).to_numpy() + off
            y = g["mean_p_W"].astype(float).to_numpy()
            e = g["std_p_W"].astype(float).to_numpy()

            plt.errorbar(
                x, y, yerr=e,
                fmt="o", markersize=4, capsize=2,
                label=f"{series} ({kind})",
            )

        plt.title(title)
        plt.xlabel("Test Repetition")
        plt.ylabel(ylabel)
        plt.xticks(reps)
        # plt.legend(loc="best")
        plt.legend(
            loc="upper right",
            fontsize=8,
            markerscale=0.8,
            handlelength=1.5,
            handletextpad=0.4,
            borderpad=0.3,
            labelspacing=0.3,
        )

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

    @staticmethod
    def _gpu_friendly_name(uuid: str) -> str:
        m = getattr(config, "GPU_UUID_NAMES", {}) or {}
        return str(m.get(uuid, uuid))
    
    @staticmethod
    def _gpu_aggregate_series_stats(
        s: pd.DataFrame,
        metric: str,
        t_start: pd.Timestamp,
        t_end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Build per-(gpu_uuid, pod) stats:
          - filter metric
          - sum counters per timestamp within each (gpu_uuid, pod)
          - convert to power samples
          - mean/std per (gpu_uuid, pod)

        Returns rows: gpu_uuid, pod, mean_p_W, std_p_W, n_samples
        """
        if s.empty:
            return pd.DataFrame(columns=["gpu_uuid", "pod", "mean_p_W", "std_p_W", "n_samples"])

        g = s[s["metric"] == metric].copy()
        if g.empty:
            return pd.DataFrame(columns=["gpu_uuid", "pod", "mean_p_W", "std_p_W", "n_samples"])

        # Extract labels we need
        g["_gpu_uuid"] = g["labels"].apply(lambda x: WorkloadEnergyTest._get_label(x, "gpu_uuid"))
        g["_pod"] = g["labels"].apply(lambda x: WorkloadEnergyTest._get_label(x, "pod"))

        # Keep only rows with uuid+pod
        g = g[(g["_gpu_uuid"] != "") & (g["_pod"] != "")]
        if g.empty:
            return pd.DataFrame(columns=["gpu_uuid", "pod", "mean_p_W", "std_p_W", "n_samples"])

        rows: List[dict] = []

        for (uuid, pod), gg in g.groupby(["_gpu_uuid", "_pod"], sort=True):
            # Sum counters across remaining label splits at each timestamp
            ctr = (
                gg.groupby("ts", as_index=False)["value"]
                .sum()
                .sort_values("ts")
            )
            ctr["ts"] = pd.to_datetime(ctr["ts"], utc=True)

            # Also enforce window bounds here (in case phases overlap oddly)
            ctr = ctr[(ctr["ts"] >= t_start) & (ctr["ts"] <= t_end)]
            if ctr.empty:
                continue

            pw = WorkloadEnergyTest._counter_to_interval_power_W(ctr)
            if pw.empty:
                continue

            mean_p, std_p, n = WorkloadEnergyTest._power_stats(pw)
            rows.append(
                {
                    "gpu_uuid": str(uuid),
                    "pod": str(pod),
                    "mean_p_W": mean_p,
                    "std_p_W": std_p,
                    "n_samples": n,
                }
            )

        return pd.DataFrame(rows)

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

        server_state = WorkloadEnergyTest._server_state_from_run_key(run_key)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # ------------------------------------------------------------------
        # GPU mode
        #   - gpu_concurrent_2pods: use per-pod windows (old semantics) -> clean bands
        #   - gpu_concurrent_3pods: keep rep-level intersection (we will revisit later)
        # ------------------------------------------------------------------
        if self.cpu_domains is None:
            rep_rows: List[dict] = []

            # -------------------------------
            # Case 1: gpu_concurrent_2pods
            # Restore old behavior: treat each window row (pod) independently.
            # This avoids artificial drops caused by rep-level window intersection.
            # -------------------------------
            if self.min_workloads_per_rep == 2:
                for _, w in windows.iterrows():
                    rep = int(w["rep"])
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

                    # Dead-pod guard (belt and suspenders): enforce exact pod match.
                    def _pod_is(lbl) -> bool:
                        try:
                            return (lbl or {}).get("pod", "") == pod
                        except Exception:
                            return False

                    s = s[s["labels"].apply(_pod_is)].copy()
                    if s.empty:
                        continue

                    # dynamic-only for now
                    for kind in ("dynamic",):
                        metric = self._metric_key(kind, None)

                        stats = WorkloadEnergyTest._gpu_aggregate_series_stats(
                            s=s,
                            metric=metric,
                            t_start=t_start,
                            t_end=t_end,
                        )
                        if stats.empty:
                            continue

                        # In this per-pod mode, stats should contain exactly this pod.
                        # But we still iterate defensively.
                        for _, r in stats.iterrows():
                            uuid = str(r["gpu_uuid"])
                            gpu_name = WorkloadEnergyTest._gpu_friendly_name(uuid)

                            rep_rows.append(
                                {
                                    "run_key": run_key,
                                    "rep": rep,
                                    "parent_phase": self.target_phase,
                                    "series": gpu_name,
                                    "kind": kind,
                                    "domain": "",
                                    "start_utc": t_start,
                                    "end_utc": t_end,
                                    "mean_p_W": float(r["mean_p_W"]),
                                    "std_p_W": float(r["std_p_W"]),
                                    "n_samples": int(r["n_samples"]),
                                    "gpu_uuid": uuid,
                                    "pod": str(r.get("pod", pod)),
                                }
                            )

            # -------------------------------
            # Case 2: gpu_concurrent_3pods
            # Use original semantics: per-pod window (no rep-level intersection).
            # One line per pod, labeled by GPU + workload.
            # -------------------------------
            else:
                for _, w in windows.iterrows():
                    rep = int(w["rep"])
                    workload = str(w.get("workload", "") or "")
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

                    # Dead-pod guard: enforce exact pod match.
                    def _pod_is(lbl) -> bool:
                        try:
                            return (lbl or {}).get("pod", "") == pod
                        except Exception:
                            return False

                    s = s[s["labels"].apply(_pod_is)].copy()
                    if s.empty:
                        continue

                    for kind in ("dynamic",):
                        metric = self._metric_key(kind, None)

                        stats = WorkloadEnergyTest._gpu_aggregate_series_stats(
                            s=s,
                            metric=metric,
                            t_start=t_start,
                            t_end=t_end,
                        )
                        if stats.empty:
                            continue

                        # Should be exactly one (gpu_uuid, pod) row, but iterate defensively.
                        for _, r in stats.iterrows():
                            uuid = str(r["gpu_uuid"])
                            gpu_name = WorkloadEnergyTest._gpu_friendly_name(uuid)

                            # One line per pod. Prefer workload name if present; fall back to pod.
                            suffix = workload if workload else pod
                            series_label = f"{gpu_name} / {suffix}"

                            rep_rows.append(
                                {
                                    "run_key": run_key,
                                    "rep": rep,
                                    "parent_phase": self.target_phase,
                                    "series": series_label,
                                    "kind": kind,
                                    "domain": "",
                                    "start_utc": t_start,
                                    "end_utc": t_end,
                                    "mean_p_W": float(r["mean_p_W"]),
                                    "std_p_W": float(r["std_p_W"]),
                                    "n_samples": int(r["n_samples"]),
                                    "gpu_uuid": uuid,
                                    "pod": str(r.get("pod", pod)),
                                }
                            )


            res = pd.DataFrame(rep_rows)
            out_parquet = out_dir / f"results_{self.name}.parquet"
            res.to_parquet(out_parquet, index=False)

            if res.empty:
                raise RuntimeError(
                    f"{self.name} produced no GPU results. "
                    "Likely: missing gpu_uuid label, missing metric, or filtering too strict."
                )
            
            # Role-based relabeling for gpu_concurrent_3pods only
            if self.min_workloads_per_rep == 3:
                res = WorkloadEnergyTest._relabel_gpu3_split_single_roles(res)

            title = f"{self.target_phase}: workload power by GPU (idle vs dynamic) on {server_state} server"
            fname = f"scatter_{self.target_phase}__{ts}.png"
            ylabel = "mean workload power (W)"

            WorkloadEnergyTest._plot_scatter_multi_series(
                res,
                out_path=figs_dir / fname,
                title=title,
                ylabel=ylabel,
            )

            log(f"[{self.name}] wrote {out_parquet}")
            log(f"[{self.name}] figures in: {figs_dir}")
            return



        # ------------------------------------------------------------------
        # CPU mode (unchanged behavior): per-window per-workload series,
        # idle/dynamic separated, optional domain loop.
        # ------------------------------------------------------------------
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

                    g = s[s["metric"] == metric].copy()
                    if g.empty:
                        continue

                    g["_pod"] = g["labels"].apply(lambda x: WorkloadEnergyTest._get_label(x, "pod"))
                    g = g[g["_pod"] == pod].drop(columns=["_pod"])
                    if g.empty:
                        continue

                    ctr = (
                        g.groupby("ts", as_index=False)["value"]
                        .sum()
                        .sort_values("ts")
                    )
                    ctr["ts"] = pd.to_datetime(ctr["ts"], utc=True)

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
                            "series": workload,
                            "kind": kind,
                            "domain": "" if domain is None else str(domain),
                            "start_utc": t_start,
                            "end_utc": t_end,
                            "mean_p_W": mean_p,
                            "std_p_W": std_p,
                            "n_samples": n,
                            "pod": pod,
                        }
                    )

            res = pd.DataFrame(rows)

            # Write results per domain
            if domain is None:
                out_parquet = out_dir / f"results_{self.name}.parquet"
            else:
                out_parquet = out_dir / f"results_{self.name}__{domain}.parquet"

            res.to_parquet(out_parquet, index=False)

            if res.empty:
                continue

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
