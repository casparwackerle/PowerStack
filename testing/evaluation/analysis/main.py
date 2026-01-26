from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

import config
import extract
import parse_events
import parse_subruns
from tests.tests_idle import run_idle_test
from tests.tests_cpu_discrimination import run_cpu_discrimination_test


def log(msg: str) -> None:
    now = datetime.utcnow().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def ensure_dirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs(config.OUT_DIR)

    tests = []
    if "idle" in config.TESTS_TO_RUN:
        tests.append(run_idle_test)
    if "cpu_discrimination" in config.TESTS_TO_RUN:
        tests.append(run_cpu_discrimination_test)

    if not tests:
        log("No tests selected. Set config.TESTS_TO_RUN.")
        return

    for run_key in config.SELECTED_RUN_KEYS:
        run_dir = config.RUNS[run_key]
        out_run_dir = config.OUT_DIR / run_key
        ensure_dirs(out_run_dir)
        ensure_dirs(out_run_dir / "figs")

        events_path = run_dir / "events.log"
        if not events_path.exists():
            raise FileNotFoundError(f"Missing events.log: {events_path}")

        # 1) Parse master phases
        log(f"=== RUN {run_key} ===")
        log(f"Prometheus URL: {config.PROM_URL}")
        log(f"Parsing master events: {events_path}")

        phases_df = parse_events.parse_events_log(events_path, run_key=run_key)
        log(f"Parsed events.log → {len(phases_df)} phase instances")
        if len(phases_df) == 0:
            raise RuntimeError("events.log parsed to 0 phase instances.")

        for col in ("run_key", "rep", "phase", "start_utc", "end_utc"):
            if col not in phases_df.columns:
                raise RuntimeError(f"phases_df missing '{col}'.")

        (out_run_dir / "phases.parquet").write_bytes(b"")  # ensure path exists
        phases_out = out_run_dir / "phases.parquet"
        phases_df.to_parquet(phases_out, index=False)
        log(f"Wrote: {phases_out}")

        # 1b) Build CPU discrimination windows anchored to master phase
        windows_df = None
        if "cpu_discrimination" in config.TESTS_TO_RUN:
            targets = getattr(config, "CPU_DISCRIM_TARGET_PHASES", None)
            if not targets:
                targets = [getattr(config, "CPU_DISCRIM_TARGET_PHASE", "cpu_discrimination_concurrent_alt")]

            all_windows = []
            for target in targets:
                log(f"Building CPU discrimination windows (target phase='{target}')...")
                wdf = parse_subruns.build_cpu_discrimination_windows(
                    run_key=run_key,
                    run_dir=run_dir,
                    phases_df=phases_df,
                    target_phase=target,
                )
                all_windows.append(wdf)

            windows_df = pd.concat(all_windows, ignore_index=True)
            win_out = out_run_dir / "cpu_discrimination_windows.parquet"
            windows_df.to_parquet(win_out, index=False)
            log(f"Wrote: {win_out} ({len(windows_df)} windows, phases={len(targets)})")


        # 2) Determine required metrics
        required: dict[str, str] = {}
        for t in tests:
            req = t.required_metrics()
            for k, v in req.items():
                if k in required and required[k] != v:
                    raise RuntimeError(f"Metric key '{k}' has conflicting promql.")
                required[k] = v

        log(f"Enabled tests: {[t.name for t in tests]}")
        log(f"Total metrics to extract: {len(required)}")

        # 3) Select extraction windows
        frames = []
        if "idle" in config.TESTS_TO_RUN:
            frames.append(phases_df[phases_df["phase"] == "idle_baseline_start"].copy())

        if "cpu_discrimination" in config.TESTS_TO_RUN:
            if windows_df is None:
                raise RuntimeError("cpu_discrimination selected but windows_df is None")
            frames.append(windows_df[["run_key", "rep", "phase", "start_utc", "end_utc"]].copy())

        if not frames:
            raise RuntimeError("No extraction windows selected.")

        phases_used = pd.concat(frames, ignore_index=True)
        phases_used = phases_used.sort_values(["rep", "start_utc"], kind="stable").reset_index(drop=True)
        log(f"Extraction windows: {len(phases_used)}")

        # 4) Extract
        log("Starting Prometheus extraction...")
        t0 = datetime.utcnow()

        series_df = extract.extract_for_phases(
            run_key=run_key,
            phases_df=phases_used,
            prom_url=config.PROM_URL,
            timeout_sec=config.PROM_TIMEOUT_SEC,
            max_points=config.PROM_MAX_POINTS,
            default_step_sec=config.DEFAULT_STEP_SEC,
            metrics=required,
        )

        dt = (datetime.utcnow() - t0).total_seconds()
        log(f"Extraction complete in {dt:.1f}s → {len(series_df)} samples")

        series_out = out_run_dir / "series.parquet"
        series_df.to_parquet(series_out, index=False)
        log(f"Wrote: {series_out}")

        # 5) Run tests
        for t in tests:
            log(f"Running test: {t.name}")
            t.run(run_key=run_key, phases_df=phases_df, series_df=series_df, out_dir=out_run_dir)
            log(f"Finished test: {t.name}")

        log(f"=== DONE {run_key} ===")


if __name__ == "__main__":
    main()
