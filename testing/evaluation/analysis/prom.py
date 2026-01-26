# analysis/prom.py
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, List

import requests

import config


# One session for the whole analysis run.
# If configured, this session routes ONLY this script's Prometheus traffic via SOCKS.
_session = requests.Session()

if getattr(config, "PROM_USE_SOCKS", False):
    proxy = getattr(config, "PROM_SOCKS_PROXY", "").strip()
    if not proxy:
        raise RuntimeError("PROM_USE_SOCKS=True but config.PROM_SOCKS_PROXY is empty.")
    _session.proxies.update({"http": proxy, "https": proxy})


def _to_unix(dt: datetime) -> int:
    return int(dt.timestamp())


def _shorten(s: str, n: int = 180) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def query_range(
    prom_url: str,
    promql: str,
    start_dt: datetime,
    end_dt: datetime,
    step_sec: int,
    timeout_sec: int = 60,
) -> List[Dict[str, Any]]:
    """
    Query Prometheus /api/v1/query_range and return the raw 'result' list.

    Adds:
    - request start/end log lines (duration, series count)
    - rich error context on failures (HTTP status, payload status, etc.)
    """
    # Local import to avoid circular imports (main imports extract -> prom).
    try:
        from main import log  # when run as `python analysis/main.py`
    except Exception:
        def log(_: str) -> None:
            return

    url = prom_url.rstrip("/") + "/api/v1/query_range"
    params = {
        "query": promql,
        "start": _to_unix(start_dt),
        "end": _to_unix(end_dt),
        "step": f"{step_sec}s",
    }

    log(
        f"[prom] GET query_range start={start_dt.isoformat()} end={end_dt.isoformat()} "
        f"step={step_sec}s timeout={timeout_sec}s promql='{_shorten(promql)}'"
    )

    t0 = time.time()
    try:
        r = _session.get(url, params=params, timeout=timeout_sec)
    except requests.exceptions.RequestException as e:
        dt_s = time.time() - t0
        raise RuntimeError(
            "Prometheus request failed.\n"
            f"  url={url}\n"
            f"  window={start_dt.isoformat()}..{end_dt.isoformat()} step={step_sec}s\n"
            f"  promql={promql}\n"
            f"  after={dt_s:.2f}s\n"
            f"  error={repr(e)}"
        ) from e

    dt_s = time.time() - t0
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # Try to capture response body for debugging.
        body = ""
        try:
            body = r.text
        except Exception:
            body = "<unavailable>"
        raise RuntimeError(
            "Prometheus HTTP error.\n"
            f"  url={url}\n"
            f"  status={r.status_code}\n"
            f"  window={start_dt.isoformat()}..{end_dt.isoformat()} step={step_sec}s\n"
            f"  promql={promql}\n"
            f"  after={dt_s:.2f}s\n"
            f"  body={_shorten(body, 500)}"
        ) from e

    # Parse JSON payload with context on failure
    try:
        payload = r.json()
    except Exception as e:
        raise RuntimeError(
            "Prometheus response was not valid JSON.\n"
            f"  url={url}\n"
            f"  status={r.status_code}\n"
            f"  after={dt_s:.2f}s\n"
            f"  text={_shorten(r.text, 500)}"
        ) from e

    if payload.get("status") != "success":
        raise RuntimeError(
            "Prometheus query returned non-success status.\n"
            f"  url={url}\n"
            f"  window={start_dt.isoformat()}..{end_dt.isoformat()} step={step_sec}s\n"
            f"  promql={promql}\n"
            f"  after={dt_s:.2f}s\n"
            f"  payload={json.dumps(payload)[:800]}"
        )

    result = payload.get("data", {}).get("result", []) or []
    log(f"[prom] OK after={dt_s:.2f}s series={len(result)}")
    return result
