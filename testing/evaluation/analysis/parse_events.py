# analysis/parse_events.py
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd


# Examples:
# 2026-01-19T21:51:48Z rep=1 start
_RE_REP_START = re.compile(r"^(?P<ts>\S+)\s+rep=(?P<rep>\d+)\s+start$")

# 2026-01-19T21:51:48Z phase 'idle_baseline_start' start_utc=2026-01-19T21:51:48Z type=sleep template=noop-sleep
_RE_PHASE_START = re.compile(
    r"^(?P<ts>\S+)\s+phase\s+'(?P<phase>[^']+)'\s+start_utc=(?P<start>\S+)\s+type=(?P<type>\S+)"
    r"(?:\s+template=(?P<template>\S+))?(?P<rest>.*)$"
)

# 2026-01-19T21:54:48Z phase 'idle_baseline_start' end_utc=2026-01-19T21:54:48Z status=ok reason=
_RE_PHASE_END = re.compile(
    r"^(?P<ts>\S+)\s+phase\s+'(?P<phase>[^']+)'\s+end_utc=(?P<end>\S+)\s+status=(?P<status>\S+)(?P<rest>.*)$"
)

# Optional: extract pod=... when present in "rest" lines (workload / ramp lines)
_RE_POD = re.compile(r"\bpod=(?P<pod>\S+)\b")


def _parse_ts_z(ts: str) -> datetime:
    # Example: 2026-01-19T21:51:48Z
    if not ts.endswith("Z"):
        raise ValueError(f"Expected Zulu timestamp, got: {ts}")
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def parse_events_log(path: Path, run_key: str) -> pd.DataFrame:
    """
    Parse the master events.log and produce one row per phase instance:
    (run_key, rep, phase, type, template, pod, start_utc, end_utc, status, duration_s).

    Important: pod names often appear on separate "workload ..." lines, not in the phase start line.
    We therefore keep a small rolling "last pod seen" per rep, and attach it to the next phase start
    if the phase start itself has no pod=... field.
    """
    rows: List[Dict[str, Any]] = []

    cur_rep: int | None = None

    # Track open phases per rep by name.
    open_phase: Dict[str, Dict[str, Any]] = {}

    # Best-effort pod inference: when we see a pod=... line, remember it briefly.
    last_pod_seen: str = ""

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")

            m = _RE_REP_START.match(line)
            if m:
                cur_rep = int(m.group("rep"))
                last_pod_seen = ""
                continue

            # Many lines include pod=... (workload/ramp lines). Capture it opportunistically.
            mp = _RE_POD.search(line)
            if mp:
                last_pod_seen = mp.group("pod")

            m = _RE_PHASE_START.match(line)
            if m and cur_rep is not None:
                phase = m.group("phase")
                start_dt = _parse_ts_z(m.group("start"))
                ptype = m.group("type")
                template = m.group("template") or ""
                rest = m.group("rest") or ""

                # Prefer explicit pod in rest, otherwise fall back to last pod seen.
                pod = ""
                mp2 = _RE_POD.search(rest)
                if mp2:
                    pod = mp2.group("pod")
                elif last_pod_seen:
                    pod = last_pod_seen

                # If a phase with same name is already open (should not happen, but can),
                # close it with a parse note and keep going.
                if phase in open_phase:
                    prev = open_phase.pop(phase)
                    rows.append(
                        {
                            "run_key": run_key,
                            "rep": cur_rep,
                            "phase": phase,
                            "type": prev.get("type", ""),
                            "template": prev.get("template", ""),
                            "pod": prev.get("pod", ""),
                            "start_utc": prev.get("start_utc", pd.NaT),
                            "end_utc": start_dt,
                            "status": "parse_forced_close",
                            "parse_note": f"duplicate_start line={line_no}",
                        }
                    )

                open_phase[phase] = {
                    "start_utc": start_dt,
                    "type": ptype,
                    "template": template,
                    "pod": pod,
                }
                continue

            m = _RE_PHASE_END.match(line)
            if m and cur_rep is not None:
                phase = m.group("phase")
                end_dt = _parse_ts_z(m.group("end"))
                status = m.group("status")

                if phase not in open_phase:
                    # End without start: keep a record; better than crashing
                    rows.append(
                        {
                            "run_key": run_key,
                            "rep": cur_rep,
                            "phase": phase,
                            "type": "",
                            "template": "",
                            "pod": "",
                            "start_utc": pd.NaT,
                            "end_utc": end_dt,
                            "status": status,
                            "parse_note": f"end_without_start line={line_no}",
                        }
                    )
                    continue

                st = open_phase.pop(phase)
                rows.append(
                    {
                        "run_key": run_key,
                        "rep": cur_rep,
                        "phase": phase,
                        "type": st.get("type", ""),
                        "template": st.get("template", ""),
                        "pod": st.get("pod", ""),
                        "start_utc": st.get("start_utc", pd.NaT),
                        "end_utc": end_dt,
                        "status": status,
                        "parse_note": "",
                    }
                )
                continue

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No phases parsed from {path}")

    # Convert time columns
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True, errors="coerce")
    df["end_utc"] = pd.to_datetime(df["end_utc"], utc=True, errors="coerce")

    # Basic validation: we require start/end for all rows (we keep end_without_start rows above,
    # but they are useless for extraction; fail loudly so you notice)
    missing = df[df["start_utc"].isna() | df["end_utc"].isna()]
    if not missing.empty:
        example = missing.head(5).to_string(index=False)
        raise RuntimeError(f"Parsed phases contain missing start/end times. Example:\n{example}")

    # Sort & duration
    df = df.sort_values(["rep", "start_utc"], kind="stable").reset_index(drop=True)
    df["duration_s"] = (df["end_utc"] - df["start_utc"]).dt.total_seconds()

    return df
