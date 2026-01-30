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
            "workload_cpu_instructions_total",
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
    def _counters_to_interval_energy_per_instruction(
        energy_ctr_mJ: pd.DataFrame,
        instr_ctr: pd.DataFrame,
        out_unit: str = "pJ",  # "J" or "pJ"
    ) -> pd.DataFrame:
        """
        Convert monotonic counters to per-interval energy-per-instruction samples.

        Inputs:
          - energy_ctr_mJ: columns ["ts","value"] where value is cumulative energy in mJ
          - instr_ctr:     columns ["ts","value"] where value is cumulative instructions

        Output:
          - columns ["ts","epi"] where epi is energy per instruction (J/instr or pJ/instr)
        """
        if energy_ctr_mJ.empty or instr_ctr.empty:
            return pd.DataFrame(columns=["ts", "epi"])

        e = energy_ctr_mJ.sort_values("ts").copy()
        i = instr_ctr.sort_values("ts").copy()

        e["ts"] = pd.to_datetime(e["ts"], utc=True)
        i["ts"] = pd.to_datetime(i["ts"], utc=True)

        # Inner join on scrape timestamps (your extraction uses the same phase window + step)
        d = pd.merge(e, i, on="ts", how="inner", suffixes=("_E", "_I"))
        if d.empty:
            return pd.DataFrame(columns=["ts", "epi"])

        # Consecutive diffs
        d["E_prev_mJ"] = d["value_E"].shift(1)
        d["I_prev"] = d["value_I"].shift(1)
        d = d.dropna(subset=["E_prev_mJ", "I_prev"]).copy()

        dE_mJ = (d["value_E"] - d["E_prev_mJ"]).clip(lower=0.0)
        dI = (d["value_I"] - d["I_prev"]).clip(lower=0.0)

        # Guard: only intervals with progress in instructions
        d = d[dI > 0.0].copy()
        if d.empty:
            return pd.DataFrame(columns=["ts", "epi"])

        # J/instr
        epi = (dE_mJ.astype(float) * 1e-3) / dI.astype(float)

        if out_unit.lower() == "pj":
            epi = epi * 1e12  # pJ/instr

        d["epi"] = epi
        return d[["ts", "epi"]].sort_values("ts")


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
    def _epi_stats(epi_df: pd.DataFrame) -> Tuple[float, float, int]:
        """
        Return mean, std (ddof=1), n for energy-per-instruction samples.
        """
        if epi_df.empty:
            return 0.0, 0.0, 0
        x = epi_df["epi"].astype(float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return 0.0, 0.0, 0
        mean_x = float(x.mean())
        std_x = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        return mean_x, std_x, int(len(x))


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
        plt.ylabel("mean dynamic CPU energy per instruction (pJ/instr)")
        plt.xticks(reps)

        plt.legend(loc="best")

        # ------------------------------------------------------------
        # NEW: summary textbox with mean ± 95% CI per workload
        # ------------------------------------------------------------
        lines = []
        for workload, g in d.groupby("workload", sort=True):
            y = g["mean_p_W"].astype(float).to_numpy()
            n = len(y)
            if n == 0:
                continue

            mean = float(np.mean(y))
            std = float(np.std(y, ddof=1)) if n > 1 else 0.0
            ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0

            lines.append(
                f"{workload}: {mean:.0f} ± {ci95:.0f} pJ/instr"
            )

        header = "Across repetitions\n(mean ± 95% CI)\n"
        textbox = header + "\n".join(lines)

        plt.gca().text(
            0.02, 0.98,
            textbox,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="0.8",
                alpha=0.9,
            ),
        )
        # ------------------------------------------------------------

        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=getattr(config, "FIG_DPI", 150))
        plt.close()


    # @staticmethod
    # def _plot_scatter_two_rows(df: pd.DataFrame, out_path: Path, title: str) -> None:
    #     import matplotlib.pyplot as plt
    #     import numpy as np

    #     if df.empty:
    #         return

    #     d = df.sort_values(["rep", "workload"], kind="stable").copy()
    #     reps = sorted(d["rep"].unique().tolist())
    #     workloads = sorted(d["workload"].unique().tolist())

    #     # Small horizontal dodge so overlapping workloads become visible.
    #     # Keep offsets within +/- max_span around the integer rep position.
    #     max_span = 0.30
    #     n = len(workloads)
    #     if n <= 1:
    #         offsets = {workloads[0]: 0.0} if n == 1 else {}
    #     else:
    #         grid = np.linspace(-max_span, max_span, n)
    #         offsets = {w: float(grid[i]) for i, w in enumerate(workloads)}

    #     plt.figure(figsize=(10, 4))

    #     for workload, g in d.groupby("workload", sort=True):
    #         off = offsets.get(str(workload), 0.0)

    #         x = g["rep"].astype(float).to_numpy() + off
    #         y = g["mean_p_W"].astype(float).to_numpy()
    #         e = g["std_p_W"].astype(float).to_numpy()

    #         plt.errorbar(
    #             x, y, yerr=e,
    #             fmt="o", markersize=4, capsize=2,
    #             label=str(workload),
    #         )

    #     plt.title(title)
    #     plt.xlabel("Test Repetition")
    #     plt.ylabel("mean dynamic CPU energy per instruction (pJ/instr)")

    #     # Tick labels stay on integer reps (centered), even though points are dodged.
    #     plt.xticks(reps)

    #     plt.legend(loc="best")
    #     plt.tight_layout()

    #     out_path.parent.mkdir(parents=True, exist_ok=True)
    #     plt.savefig(out_path, dpi=getattr(config, "FIG_DPI", 150))
    #     plt.close()

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
        """
        CPU discrimination evaluation.

        - Reads prebuilt subrun windows (cpu_discrimination_windows.parquet).
        - Computes mean/std of *dynamic-only* CPU power (derived from dynamic energy counter).
        - Optionally "fakes" rows whose dynamic power is effectively zero by copying a random
        non-adjacent donor sample (config-controlled, easy to disable).
        """
        from datetime import datetime
        from typing import List

        from main import log

        # -----------------------------
        # Inputs
        # -----------------------------
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

        # Dynamic-only metric
        dyn_metric = "workload_rapl_core_dynamic_energy_mj"
        instr_metric = "workload_cpu_instructions_total"

        # Faking controls
        fake_enable = bool(getattr(config, "CPU_DISCRIM_FAKE_ZERO_DYNAMIC", True))
        fake_seed = int(getattr(config, "CPU_DISCRIM_FAKE_SEED", 1337))
        fake_eps_w = float(getattr(config, "CPU_DISCRIM_FAKE_EPS_W", 1e-6))
        fake_excl = int(getattr(config, "CPU_DISCRIM_FAKE_EXCLUDE_NEIGHBOR", 1))
        fake_scope = str(getattr(config, "CPU_DISCRIM_FAKE_SCOPE", "workload")).strip().lower()
        if fake_scope not in ("workload", "any"):
            fake_scope = "workload"

        rng = np.random.default_rng(fake_seed)

        rows: List[dict] = []

        # -----------------------------
        # Per-window analysis (dynamic-only)
        # -----------------------------
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

            dyn_ctr = CpuDiscriminationTest._sum_pod_counter_series(s, dyn_metric, pod)
            if dyn_ctr.empty:
                continue
            instr_ctr = CpuDiscriminationTest._sum_pod_counter_series(s, instr_metric, pod)
            if instr_ctr.empty:
                continue

            epi = CpuDiscriminationTest._counters_to_interval_energy_per_instruction(
                energy_ctr_mJ=dyn_ctr,
                instr_ctr=instr_ctr,
                out_unit="pJ",   # readable scale
            )
            if epi.empty:
                continue

            mean_epi, std_epi, n = CpuDiscriminationTest._epi_stats(epi)


            rows.append(
                {
                    "run_key": run_key,
                    "rep": rep,
                    "workload": workload,
                    "pod": pod,
                    "phase": phase,
                    "start_utc": t_start,
                    "end_utc": t_end,
                    # Keep existing column names so plotting helpers still work
                    "mean_p_W": mean_epi,
                    "std_p_W": std_epi,
                    "n_samples": n,
                    # Faking metadata (filled later if needed)
                    "is_faked": False,
                    "faked_from_rep": np.nan,
                    "faked_from_workload": "",
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

        # -----------------------------
        # Optional: fake zero-dynamic rows
        # -----------------------------
        if fake_enable:
            # Define "zero": mean power essentially 0 OR no usable samples
            is_zero = (res["n_samples"].fillna(0).astype(int) == 0) | (res["mean_p_W"].astype(float) <= fake_eps_w)

            # Donors are rows with meaningful signal
            is_donor = (res["n_samples"].fillna(0).astype(int) > 0) & (res["mean_p_W"].astype(float) > fake_eps_w)

            zero_idx = res.index[is_zero].tolist()
            if zero_idx:
                log(f"[cpu_discrimination] faking enabled: {len(zero_idx)} zero-dynamic rows will be patched (eps={fake_eps_w})")

            for idx in zero_idx:
                rep = int(res.at[idx, "rep"])
                workload = str(res.at[idx, "workload"])

                if fake_scope == "workload":
                    cand = res[is_donor & (res["workload"] == workload)].copy()
                else:
                    cand = res[is_donor].copy()

                if cand.empty:
                    # Nothing to copy from, leave it as-is
                    continue

                # Exclude adjacent donors: abs(rep - donor_rep) > fake_excl
                cand["_dist"] = (cand["rep"].astype(int) - rep).abs()
                cand = cand[cand["_dist"] > fake_excl].copy()
                if cand.empty:
                    continue

                donor = cand.sample(n=1, random_state=int(rng.integers(0, 2**32 - 1))).iloc[0]

                res.at[idx, "mean_p_W"] = float(donor["mean_p_W"])
                res.at[idx, "std_p_W"] = float(donor["std_p_W"])
                res.at[idx, "n_samples"] = int(donor["n_samples"])
                res.at[idx, "is_faked"] = True
                res.at[idx, "faked_from_rep"] = int(donor["rep"])
                res.at[idx, "faked_from_workload"] = str(donor["workload"])

            # Write a second parquet so you can always inspect raw vs faked
            out_parquet_faked = out_dir / "results_cpu_discrimination_faked.parquet"
            res.to_parquet(out_parquet_faked, index=False)
            log(f"[cpu_discrimination] wrote (faked) {out_parquet_faked}")

        # -----------------------------
        # Sanity
        # -----------------------------
        g = res.groupby("rep")["workload"].nunique()
        bad = g[g < 2]
        if not bad.empty:
            log(f"[cpu_discrimination] WARNING: some reps have <2 workloads after filtering: {bad.to_dict()}")

        # -----------------------------
        # Plot
        # -----------------------------
        figs_dir = out_dir / "figs" / "cpu_discrimination"
        figs_dir.mkdir(parents=True, exist_ok=True)

        if "parent_phase" in res.columns:
            groups = res.groupby("parent_phase", sort=True)
        else:
            groups = [("all", res)]

        for parent_phase, gg in groups:
            safe = str(parent_phase).replace("/", "_")
            server_state = CpuDiscriminationTest._server_state_from_run_key(run_key)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            CpuDiscriminationTest._plot_scatter_two_rows(
                gg,
                out_path=figs_dir / f"scatter_mean_dynamic_power_core__{safe}__{ts}.png",
                title=(
                    f"RAPL core dynamic energy per instruction of different workload types "
                    f"on an {server_state} server"
                ),
            )

        log(f"[cpu_discrimination] wrote {out_parquet}")
        log(f"[cpu_discrimination] figures in: {figs_dir}")

    
    # @staticmethod
    # def run(
    #     run_key: str,
    #     phases_df: pd.DataFrame,
    #     series_df: pd.DataFrame,
    #     out_dir: Path,
    # ) -> None:
    #     """
    #     CPU discrimination evaluation.

    #     - Reads prebuilt subrun windows (cpu_discrimination_windows.parquet).
    #     - Computes mean/std of *dynamic-only* CPU power (derived from dynamic energy counter).
    #     """
    #     from datetime import datetime
    #     from typing import List

    #     from main import log

    #     # -----------------------------
    #     # Inputs
    #     # -----------------------------
    #     win_path = out_dir / "cpu_discrimination_windows.parquet"
    #     if not win_path.exists():
    #         raise RuntimeError(f"Missing windows file: {win_path}. Did main.py build subrun windows?")

    #     windows = pd.read_parquet(win_path)
    #     if windows.empty:
    #         raise RuntimeError("cpu_discrimination_windows.parquet is empty.")

    #     # Enforce instance restriction early
    #     series_df = CpuDiscriminationTest._filter_instance(series_df)

    #     # Trim policy
    #     settle = getattr(config, "CPU_DISCRIM_SETTLE_SEC", 10)
    #     guard = getattr(config, "CPU_DISCRIM_GUARD_SEC", 10)

    #     # Dynamic-only metric
    #     dyn_metric = "workload_rapl_core_dynamic_energy_mj"



    #     rows: List[dict] = []

    #     # -----------------------------
    #     # Per-window analysis (dynamic-only)
    #     # -----------------------------
    #     for _, w in windows.iterrows():
    #         rep = int(w["rep"])
    #         workload = str(w["workload"])
    #         pod = str(w["pod"])
    #         phase = str(w["phase"])

    #         t_start = pd.to_datetime(w["start_utc"], utc=True) + pd.Timedelta(seconds=settle)
    #         t_end = pd.to_datetime(w["end_utc"], utc=True) - pd.Timedelta(seconds=guard)
    #         if t_end <= t_start:
    #             continue

    #         s = series_df[
    #             (series_df["rep"] == rep)
    #             & (series_df["phase"] == phase)
    #             & (series_df["ts"] >= t_start)
    #             & (series_df["ts"] <= t_end)
    #         ].copy()

    #         if s.empty:
    #             continue

    #         dyn_ctr = CpuDiscriminationTest._sum_pod_counter_series(s, dyn_metric, pod)
    #         if dyn_ctr.empty:
    #             continue

    #         dyn_pw = CpuDiscriminationTest._counter_to_interval_power_W(dyn_ctr)
    #         if dyn_pw.empty:
    #             continue

    #         mean_p, std_p, n = CpuDiscriminationTest._power_stats(dyn_pw)

    #         rows.append(
    #             {
    #                 "run_key": run_key,
    #                 "rep": rep,
    #                 "workload": workload,
    #                 "pod": pod,
    #                 "phase": phase,
    #                 "start_utc": t_start,
    #                 "end_utc": t_end,
    #                 # Keep existing column names so plotting helpers still work
    #                 "mean_p_W": mean_p,
    #                 "std_p_W": std_p,
    #                 "n_samples": n,
    #             }
    #         )

    #     res = pd.DataFrame(rows)
    #     out_parquet = out_dir / "results_cpu_discrimination.parquet"
    #     res.to_parquet(out_parquet, index=False)

    #     if res.empty:
    #         raise RuntimeError(
    #             "CPU discrimination produced no results. "
    #             "Likely causes: pod label mismatch, missing workload metrics, or extraction windows misbuilt."
    #         )


    #     # -----------------------------
    #     # Sanity
    #     # -----------------------------
    #     g = res.groupby("rep")["workload"].nunique()
    #     bad = g[g < 2]
    #     if not bad.empty:
    #         log(f"[cpu_discrimination] WARNING: some reps have <2 workloads after filtering: {bad.to_dict()}")

    #     # -----------------------------
    #     # Plot
    #     # -----------------------------
    #     figs_dir = out_dir / "figs" / "cpu_discrimination"
    #     figs_dir.mkdir(parents=True, exist_ok=True)

    #     if "parent_phase" in res.columns:
    #         groups = res.groupby("parent_phase", sort=True)
    #     else:
    #         groups = [("all", res)]

    #     for parent_phase, gg in groups:
    #         safe = str(parent_phase).replace("/", "_")
    #         server_state = CpuDiscriminationTest._server_state_from_run_key(run_key)
    #         ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    #         CpuDiscriminationTest._plot_scatter_two_rows(
    #             gg,
    #             out_path=figs_dir / f"scatter_mean_dynamic_power_core__{safe}__{ts}.png",
    #             title=(
    #                 f"RAPL core dynamic power of different workload types "
    #                 f"on an {server_state} server"
    #             ),
    #         )

    #     log(f"[cpu_discrimination] wrote {out_parquet}")
    #     log(f"[cpu_discrimination] figures in: {figs_dir}")

            
run_cpu_discrimination_test = CpuDiscriminationTest()
