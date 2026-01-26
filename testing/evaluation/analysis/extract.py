# analysis/extract.py
from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import pandas as pd

import prom


def choose_step(start_dt: datetime, end_dt: datetime, default_step_sec: int, max_points: int) -> int:
    dur = (end_dt - start_dt).total_seconds()
    if dur <= 0:
        return default_step_sec
    min_step = max(1, int(dur // max_points))
    return max(default_step_sec, min_step)


def extract_for_phases(
    run_key: str,
    phases_df: pd.DataFrame,
    prom_url: str,
    timeout_sec: int,
    max_points: int,
    default_step_sec: int,
    metrics: Dict[str, str],   # logical_name -> promql
) -> pd.DataFrame:
    """
    Pull Prometheus query_range data for each (rep, phase) window and each requested metric.
    Output rows are "long": one row per (timestamp, series).
    """

    # Local import to avoid circular import issues with analysis.main importing extract.
    from main import log  # main.py is in the same folder when executed as `python analysis/main.py`

    if not metrics:
        log("No metrics requested for extraction. Returning empty series_df.")
        return pd.DataFrame(columns=["run_key", "rep", "phase", "ts", "metric", "value", "labels"])

    # Quick sanity: ensure required columns exist
    for col in ("rep", "phase", "start_utc", "end_utc"):
        if col not in phases_df.columns:
            raise RuntimeError(f"phases_df missing '{col}'. Got columns: {list(phases_df.columns)}")

    out_rows: List[dict] = []
    metric_items = list(metrics.items())

    n_phases = len(phases_df)
    log(f"Extraction: phases={n_phases}, metrics={len(metric_items)} (requests ≈ {n_phases * len(metric_items)})")

    for i, row in enumerate(phases_df.itertuples(index=False), start=1):
        rep = int(getattr(row, "rep"))
        phase = str(getattr(row, "phase"))
        start_dt = getattr(row, "start_utc").to_pydatetime()
        end_dt = getattr(row, "end_utc").to_pydatetime()

        step = choose_step(start_dt, end_dt, default_step_sec, max_points)

        # Coarse progress log (every N phases + first one)
        if i == 1 or i % 25 == 0:
            log(f"[{i}/{n_phases}] phase='{phase}' rep={rep} window={start_dt.isoformat()}..{end_dt.isoformat()} step={step}s")

        for logical, promql in metric_items:
            log(f"  → query metric='{logical}' phase='{phase}' rep={rep} step={step}s")

            try:
                results = prom.query_range(
                    prom_url=prom_url,
                    promql=promql,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    step_sec=step,
                    timeout_sec=timeout_sec,
                )
            except Exception as e:
                # Fail fast with context. Under time pressure, this is better than silent skipping.
                raise RuntimeError(
                    f"Prometheus query failed.\n"
                    f"  run_key={run_key}\n"
                    f"  rep={rep}\n"
                    f"  phase={phase}\n"
                    f"  metric={logical}\n"
                    f"  promql={promql}\n"
                    f"  window={start_dt.isoformat()}..{end_dt.isoformat()} step={step}s\n"
                    f"  error={repr(e)}"
                ) from e

            log(f"  ← metric='{logical}' got {len(results)} series")

            for series in results:
                labels = series.get("metric", {}) or {}
                for ts_unix, val_str in series.get("values", []) or []:
                    try:
                        v = float(val_str)
                    except Exception:
                        # Keep going; but don't die on a single bad sample.
                        continue
                    out_rows.append(
                        {
                            "run_key": run_key,
                            "rep": rep,
                            "phase": phase,
                            "metric": logical,
                            "ts": datetime.utcfromtimestamp(float(ts_unix)),
                            "value": v,
                            "labels": labels,  # dict; parquet can store as object
                        }
                    )

    df = pd.DataFrame(out_rows)
    if df.empty:
        log("Extraction produced 0 rows (no Prometheus results). Check promql/labels/time windows.")
        return df

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    log(f"Extraction produced {len(df)} rows total.")
    return df
