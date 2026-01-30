# analysis/adhoc_line_metrics.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------
# Configuration (edit as needed)
# ---------------------------------------------------------------------
DEFAULT_START_UTC = "2026-01-28T04:35:00Z"
DEFAULT_END_UTC = "2026-01-28T04:38:00Z"

DEFAULT_STEP_SEC = getattr(config, "DEFAULT_STEP_SEC", 3)
DEFAULT_RATE_WINDOW_S = 10.0  # same as tests_idle


def _parse_ts_z(s: str) -> pd.Timestamp:
    # Example: 2026-01-27T20:05:21Z
    if not s.endswith("Z"):
        raise ValueError(f"Expected Zulu timestamp ending with 'Z', got: {s}")
    return pd.to_datetime(s.replace("Z", "+00:00"), utc=True)


def _labels_key(x: Any) -> str:
    try:
        return json.dumps(x, sort_keys=True)
    except Exception:
        return str(x)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    promql: str


# Your requested default series
DEFAULT_METRICS: List[MetricSpec] = [
    MetricSpec(
        name="rapl_pkg_total",
        promql='tycho_rapl_energy_mj{domain="pkg", instance="irvine.zhaw.ch:9102", job="kepler-metrics", kind="total", source="rapl"}',
    ),
    MetricSpec(
        name="residual_total",
        promql='tycho_residual_energy_mj{chassis="Self", instance="irvine.zhaw.ch:9102", job="kepler-metrics", kind="total", source="redfish_corrected"}',
    ),
    MetricSpec(
        name="system_total",
        promql='tycho_system_energy_mj{chassis="Self", instance="irvine.zhaw.ch:9102", job="kepler-metrics", kind="total", source="redfish_corrected"}',
    ),
    # Sum of the two GPU counters as one plotted series (single PromQL)
    MetricSpec(
        name="gpu0_total",
        promql='tycho_gpu_energy_mj{gpu_uuid="GPU-6f16a00c-b60b-a0c0-abab-a1beed8807f2", instance="irvine.zhaw.ch:9102", job="kepler-metrics", kind="total", source="nvml_corrected"}',
    ),
    MetricSpec(
        name="gpu1_total",
        promql='tycho_gpu_energy_mj{gpu_uuid="GPU-1c3975de-070b-1827-b372-fab61b25635d", instance="irvine.zhaw.ch:9102", job="kepler-metrics", kind="total", source="nvml_corrected"}',
    ),
]


# ---------------------------------------------------------------------
# Prometheus client (query_range)
# ---------------------------------------------------------------------
def _requests_session():
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": "tycho-adhoc-analysis/1.0"})
    return s


def prom_query_range(
    prom_url: str,
    promql: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    step_sec: int,
    timeout_sec: int,
) -> List[Dict[str, Any]]:
    """
    Returns Prometheus "matrix" result list; each element is a series:
      {"metric": {...labels...}, "values": [[ts, "val"], ...]}
    """
    import requests

    url = prom_url.rstrip("/") + "/api/v1/query_range"
    params = {
        "query": promql,
        "start": float(start_utc.timestamp()),
        "end": float(end_utc.timestamp()),
        "step": str(int(step_sec)),
    }

    proxies = None
    if getattr(config, "PROM_USE_SOCKS", False):
        proxy = getattr(config, "PROM_SOCKS_PROXY", "")
        if proxy:
            proxies = {"http": proxy, "https": proxy}

    sess = _requests_session()
    r = sess.get(url, params=params, timeout=timeout_sec, proxies=proxies)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query_range failed: {payload}")

    data = payload.get("data", {})
    if data.get("resultType") != "matrix":
        raise RuntimeError(f"Expected matrix resultType, got: {data.get('resultType')}")

    return data.get("result", [])


def prom_matrix_to_df(result: List[Dict[str, Any]], metric_name: str) -> pd.DataFrame:
    """
    Flatten matrix results to:
      metric=<metric_name>, ts=<pd.Timestamp>, value=<float>, labels=<dict>
    """
    rows: List[Dict[str, Any]] = []
    for s in result:
        labels = s.get("metric", {}) or {}
        values = s.get("values", []) or []
        for ts_unix, v_str in values:
            try:
                v = float(v_str)
            except Exception:
                continue
            ts = pd.to_datetime(float(ts_unix), unit="s", utc=True)
            rows.append({"metric": metric_name, "ts": ts, "value": v, "labels": labels})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["metric", "ts"], kind="stable").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Counter → power (W), same logic as tests_idle
# ---------------------------------------------------------------------
def counter_series_to_power_W(df: pd.DataFrame, window_s: float) -> pd.DataFrame:
    """
    Convert ONE monotonic counter series (single labelset) into power (W) via a Grafana-like windowed delta.
    Returns columns: ts, p_W
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


def aggregate_metric_power_W(
    df: pd.DataFrame,
    metric_name: str,
    t0: pd.Timestamp,
    window_s: float,
) -> pd.DataFrame:
    """
    Split by labelset, compute power per series, then sum across series by timestamp.
    Output: ts, rel_t_s, p_W
    """
    g = df[df["metric"] == metric_name].copy()
    if g.empty:
        return pd.DataFrame(columns=["ts", "rel_t_s", "p_W"])

    # Enforce instance restriction (extra safety; your promql already pins instance)
    inst = getattr(config, "PROM_INSTANCE", None)
    if inst:
        def _get_inst(lbl) -> str:
            try:
                return (lbl or {}).get("instance", "")
            except Exception:
                return ""
        g["_instance"] = g["labels"].apply(_get_inst)
        g = g[g["_instance"] == inst].drop(columns=["_instance"])
        if g.empty:
            return pd.DataFrame(columns=["ts", "rel_t_s", "p_W"])

    g["labels_key"] = g["labels"].apply(_labels_key)

    per_series: List[pd.DataFrame] = []
    for _, gs in g.groupby("labels_key"):
        pw = counter_series_to_power_W(gs, window_s=window_s)
        if not pw.empty:
            per_series.append(pw)

    if not per_series:
        return pd.DataFrame(columns=["ts", "rel_t_s", "p_W"])

    pw_all = pd.concat(per_series, ignore_index=True)
    pw_sum = pw_all.groupby("ts", as_index=False)["p_W"].sum().sort_values("ts")
    pw_sum["rel_t_s"] = (pd.to_datetime(pw_sum["ts"], utc=True) - t0).dt.total_seconds()
    return pw_sum[["ts", "rel_t_s", "p_W"]]


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
def plot_lines(series: List[Tuple[str, pd.DataFrame]], title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))

    for name, df in series:
        if df.empty:
            continue
        x = df["rel_t_s"].astype(float).to_numpy()
        y = df["p_W"].astype(float).to_numpy()
        plt.plot(x, y, label=name)

    plt.title(title)
    plt.xlabel("time (s) relative to start")
    plt.ylabel("power (W)")
    plt.legend(loc="best")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=getattr(config, "FIG_DPI", 200))
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def run(
    start_utc: str = DEFAULT_START_UTC,
    end_utc: str = DEFAULT_END_UTC,
    metrics: Optional[List[MetricSpec]] = None,
    step_sec: int = DEFAULT_STEP_SEC,
    rate_window_s: float = DEFAULT_RATE_WINDOW_S,
    out_dir: Optional[Path] = None,
) -> Path:
    if metrics is None:
        metrics = DEFAULT_METRICS

    t_start = _parse_ts_z(start_utc)
    t_end = _parse_ts_z(end_utc)
    if t_end <= t_start:
        raise ValueError("end_utc must be after start_utc")

    prom_url = getattr(config, "PROM_URL", None)
    if not prom_url:
        raise RuntimeError("config.PROM_URL is not set")

    timeout_sec = int(getattr(config, "PROM_TIMEOUT_SEC", 60))

    # Fetch & flatten
    frames: List[pd.DataFrame] = []
    for ms in metrics:
        result = prom_query_range(
            prom_url=prom_url,
            promql=ms.promql,
            start_utc=t_start,
            end_utc=t_end,
            step_sec=step_sec,
            timeout_sec=timeout_sec,
        )
        df = prom_matrix_to_df(result, metric_name=ms.name)
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw.empty:
        raise RuntimeError("No samples returned. Check promql, time range, and connectivity.")

    # Convert to power and prepare plot series
    t0 = pd.to_datetime(t_start, utc=True)
    pw_by_name: Dict[str, pd.DataFrame] = {}
    for ms in metrics:
        pw_by_name[ms.name] = aggregate_metric_power_W(
            raw, metric_name=ms.name, t0=t0, window_s=rate_window_s
        )

    # Only keep non-GPU or already-aggregated series
    plotted: List[Tuple[str, pd.DataFrame]] = [
        (name, df)
        for name, df in pw_by_name.items()
        if name not in {"gpu0_total", "gpu1_total"}
    ]

    # Add summed GPU power (Watts-sum, not counter-sum)
    gpu0 = pw_by_name.get("gpu0_total", pd.DataFrame())
    gpu1 = pw_by_name.get("gpu1_total", pd.DataFrame())
    gpu_sum = sum_power_series(gpu0, gpu1, t0=t0)
    plotted.append(("gpu_total_sum", gpu_sum))


    # Output
    if out_dir is None:
        out_dir = getattr(config, "OUT_DIR", Path(".")) / "adhoc"
    out_dir = Path(out_dir)

    title = f"Total component power consumption during sudden GPU stress on an idle node"
    out_path = out_dir / "figs" / "adhoc" / "line_selected_metrics_W.png"
    plot_lines(plotted, title=title, out_path=out_path)

    # Optional: also persist raw/power for debugging
    raw_out = out_dir / "adhoc_raw_series.parquet"
    raw.to_parquet(raw_out, index=False)

    return out_path

def sum_power_series(a: pd.DataFrame, b: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """
    Sum two power series (ts, rel_t_s, p_W) by timestamp.
    Uses outer join on ts and treats missing points as 0 W.
    """
    if a.empty and b.empty:
        return pd.DataFrame(columns=["ts", "rel_t_s", "p_W"])

    aa = a[["ts", "p_W"]].rename(columns={"p_W": "pA"})
    bb = b[["ts", "p_W"]].rename(columns={"p_W": "pB"})

    m = pd.merge(aa, bb, on="ts", how="outer").sort_values("ts")
    m["pA"] = m["pA"].fillna(0.0)
    m["pB"] = m["pB"].fillna(0.0)
    m["p_W"] = m["pA"] + m["pB"]

    m["ts"] = pd.to_datetime(m["ts"], utc=True)
    m["rel_t_s"] = (m["ts"] - t0).dt.total_seconds()

    return m[["ts", "rel_t_s", "p_W"]]


if __name__ == "__main__":
    out = run()
    print(f"Wrote: {out}")
