from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import config


# ----------------------------
# Regex for subrun events.log
# ----------------------------
_RE_SUB_PHASE_START = re.compile(
    r"^(?P<ts>\S+)\s+phase\s+'(?P<phase>[^']+)'\s+start_utc=(?P<start>\S+)\s+type=(?P<type>\S+)"
)
_RE_SUB_PHASE_END = re.compile(
    r"^(?P<ts>\S+)\s+phase\s+'(?P<phase>[^']+)'\s+end_utc=(?P<end>\S+)\s+status=(?P<status>\S+)"
)
_RE_SUB_POD = re.compile(r"\bpod=(?P<pod>\S+)\b")


# ----------------------------
# Regex for master events.log
# ----------------------------
_RE_REP_START = re.compile(r"^(?P<ts>\S+)\s+rep=(?P<rep>\d+)\s+start$")
_RE_MASTER_PHASE_START = re.compile(
    r"^(?P<ts>\S+)\s+phase\s+'(?P<phase>[^']+)'\s+start_utc=(?P<start>\S+)\s+type=(?P<type>\S+)"
)
_RE_MASTER_PHASE_END = re.compile(
    r"^(?P<ts>\S+)\s+phase\s+'(?P<phase>[^']+)'\s+end_utc=(?P<end>\S+)\s+status=(?P<status>\S+)"
)
_RE_LAUNCH = re.compile(r"\blaunch\s+subrunner\b.*\bworkload=(?P<workload>\S+)\b")


def _parse_ts_z(ts: str) -> datetime:
    if not ts.endswith("Z"):
        raise ValueError(f"Expected Zulu timestamp, got: {ts}")
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def _find_subrun_dirs(run_dir: Path) -> List[Path]:
    if not run_dir.exists():
        return []
    return sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("subrun_")])


def _parse_single_subrun_events(events_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse one subrun's events.log and extract:
      - workload (from workload phase name)
      - pod name (from workload line)
      - workload phase start/end
    """
    if not events_path.exists():
        return None

    workload: str = ""
    pod: str = ""
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None
    status: str = ""

    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            mpod = _RE_SUB_POD.search(line)
            if mpod:
                pod = mpod.group("pod")

            ms = _RE_SUB_PHASE_START.match(line)
            if ms:
                workload = ms.group("phase")
                start_dt = _parse_ts_z(ms.group("start"))
                continue

            me = _RE_SUB_PHASE_END.match(line)
            if me:
                end_dt = _parse_ts_z(me.group("end"))
                status = me.group("status")
                continue

    if not (workload and pod and start_dt and end_dt):
        return None

    return {
        "workload": workload,
        "pod": pod,
        "start_utc": start_dt,
        "end_utc": end_dt,
        "status": status or "ok",
        "events_path": str(events_path),
    }


def _parse_master_launches_for_phase(master_events_path: Path, target_phase: str) -> List[Dict[str, Any]]:
    """
    Scan master events.log and return a list of launches inside target_phase:
      {rep, workload, launch_ts}
    """
    launches: List[Dict[str, Any]] = []
    cur_rep: Optional[int] = None
    in_target: bool = False

    with master_events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            mrep = _RE_REP_START.match(line)
            if mrep:
                cur_rep = int(mrep.group("rep"))
                in_target = False
                continue

            ms = _RE_MASTER_PHASE_START.match(line)
            if ms and cur_rep is not None:
                if ms.group("phase") == target_phase:
                    in_target = True
                continue

            me = _RE_MASTER_PHASE_END.match(line)
            if me and cur_rep is not None:
                if me.group("phase") == target_phase:
                    in_target = False
                continue

            if in_target and cur_rep is not None:
                ml = _RE_LAUNCH.search(line)
                if ml:
                    # use the line timestamp (leftmost token) as launch time reference
                    ts = line.split(" ", 1)[0]
                    launches.append(
                        {
                            "rep": cur_rep,
                            "workload": ml.group("workload"),
                            "launch_ts": _parse_ts_z(ts),
                        }
                    )

    return launches


def build_cpu_discrimination_windows(
    run_key: str,
    run_dir: Path,
    phases_df: pd.DataFrame,
    target_phase: str,
) -> pd.DataFrame:
    """
    Build extraction windows ONLY for the selected master phase (target_phase).

    Robust matching:
      - take (rep, workload, launch_ts) from master events.log
      - take (workload, start_utc, pod, end_utc) from subrun logs
      - for each launch, choose the subrun of same workload with nearest start_utc to launch_ts
        within CPU_DISCRIM_MATCH_TOL_SEC

    This avoids relying on rep parsing from pod names.
    """
    from main import log

    master_events = run_dir / "events.log"
    if not master_events.exists():
        raise FileNotFoundError(f"Missing master events.log: {master_events}")

    launches = _parse_master_launches_for_phase(master_events, target_phase)
    if not launches:
        raise RuntimeError(f"Found 0 subrunner launches for '{target_phase}' in {master_events}")

    # Parse all subruns
    subruns: List[Dict[str, Any]] = []
    for d in _find_subrun_dirs(run_dir):
        ev = d / "events.log"
        parsed = _parse_single_subrun_events(ev)
        if parsed:
            subruns.append(parsed)

    if not subruns:
        raise RuntimeError(f"No parseable subrun_*/events.log found under {run_dir}")

    sub = pd.DataFrame(subruns)
    sub["start_utc"] = pd.to_datetime(sub["start_utc"], utc=True)
    sub["end_utc"] = pd.to_datetime(sub["end_utc"], utc=True)

    tol_s = int(getattr(config, "CPU_DISCRIM_MATCH_TOL_SEC", 6))

    rows: List[Dict[str, Any]] = []
    used_events: set[str] = set()

    # Minimal summary logs
    reps_in_master = sorted({int(x["rep"]) for x in launches})
    log(f"[cpu_discrimination] target_phase='{target_phase}' reps_in_master={reps_in_master[:6]}{' ...' if len(reps_in_master)>6 else ''}")

    for L in launches:
        rep = int(L["rep"])
        w = str(L["workload"])
        launch_ts = pd.to_datetime(L["launch_ts"], utc=True)

        cand = sub[sub["workload"] == w].copy()
        if cand.empty:
            continue

        # nearest by absolute time difference
        cand["dt_s"] = (cand["start_utc"] - launch_ts).abs().dt.total_seconds()
        cand = cand.sort_values(["dt_s", "start_utc"], kind="stable")

        best = cand.iloc[0]
        if float(best["dt_s"]) > tol_s:
            continue

        ev_path = str(best["events_path"])
        # Prevent reusing the same subrun for multiple launches
        if ev_path in used_events:
            continue
        used_events.add(ev_path)

        phase = f"{target_phase}__{w}".lower()
        rows.append(
            {
                "run_key": run_key,
                "rep": rep,
                "phase": phase,
                "start_utc": best["start_utc"],
                "end_utc": best["end_utc"],
                "workload": w,
                "pod": str(best["pod"]),
                "status": str(best.get("status", "ok")),
                "events_path": ev_path,
                "parent_phase": target_phase,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"Matched 0 subruns for '{target_phase}'. "
            f"Try increasing CPU_DISCRIM_MATCH_TOL_SEC (currently {tol_s})."
        )

    # Validate: 2 workloads per rep
    g = df.groupby("rep")["workload"].nunique()
    bad = g[g < 2]
    if not bad.empty:
        ex = df[df["rep"].isin(bad.index)].sort_values(["rep", "workload"]).head(20).to_string(index=False)
        raise RuntimeError(
            f"Some reps have <2 matched workloads for '{target_phase}'.\nExamples:\n{ex}"
        )

    log(f"[cpu_discrimination] matched reps={df['rep'].nunique()} rows={len(df)}")
    return df.sort_values(["rep", "workload"], kind="stable").reset_index(drop=True)


def build_workload_set_windows(
    run_key: str,
    run_dir: Path,
    target_phase: str,
    min_workloads_per_rep: int = 2,
) -> pd.DataFrame:
    """
    Generic window builder for workload_set phases (cpu_busy_vs_noop_idle_share, gpu_concurrent_2pods, gpu_concurrent_3pods, ...)

    Same matching strategy as CPU discrimination:
      - scan master events.log for launches within target_phase, capturing (rep, workload, launch_ts)
      - parse all subrun_*/events.log to get (workload, pod, start_utc, end_utc)
      - for each launch, match the nearest subrun with same workload by start time within tolerance

    Output rows:
      (run_key, rep, phase, start_utc, end_utc, workload, pod, parent_phase, events_path)
    where phase is normalized as: "{target_phase}__{workload}" (lowercase).
    """
    from main import log

    master_events = run_dir / "events.log"
    if not master_events.exists():
        raise FileNotFoundError(f"Missing master events.log: {master_events}")

    launches = _parse_master_launches_for_phase(master_events, target_phase)
    if not launches:
        raise RuntimeError(f"Found 0 subrunner launches for '{target_phase}' in {master_events}")

    # Parse all subruns
    subruns: List[Dict[str, Any]] = []
    for d in _find_subrun_dirs(run_dir):
        ev = d / "events.log"
        parsed = _parse_single_subrun_events(ev)
        if parsed:
            subruns.append(parsed)

    if not subruns:
        raise RuntimeError(f"No parseable subrun_*/events.log found under {run_dir}")

    sub = pd.DataFrame(subruns)
    sub["start_utc"] = pd.to_datetime(sub["start_utc"], utc=True)
    sub["end_utc"] = pd.to_datetime(sub["end_utc"], utc=True)

    tol_s = int(getattr(config, "CPU_DISCRIM_MATCH_TOL_SEC", 6))

    rows: List[Dict[str, Any]] = []
    used_events: set[str] = set()

    reps_in_master = sorted({int(x["rep"]) for x in launches})
    log(f"[workload_set] target_phase='{target_phase}' reps_in_master={reps_in_master[:6]}{' ...' if len(reps_in_master)>6 else ''}")

    for L in launches:
        rep = int(L["rep"])
        w = str(L["workload"])
        launch_ts = pd.to_datetime(L["launch_ts"], utc=True)

        cand = sub[sub["workload"] == w].copy()
        if cand.empty:
            continue

        cand["dt_s"] = (cand["start_utc"] - launch_ts).abs().dt.total_seconds()
        cand = cand.sort_values(["dt_s", "start_utc"], kind="stable")

        best = cand.iloc[0]
        if float(best["dt_s"]) > tol_s:
            continue

        ev_path = str(best["events_path"])
        if ev_path in used_events:
            continue
        used_events.add(ev_path)

        phase = f"{target_phase}__{w}".lower()
        rows.append(
            {
                "run_key": run_key,
                "rep": rep,
                "phase": phase,
                "start_utc": best["start_utc"],
                "end_utc": best["end_utc"],
                "workload": w,
                "pod": str(best["pod"]),
                "status": str(best.get("status", "ok")),
                "events_path": ev_path,
                "parent_phase": target_phase,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"Matched 0 subruns for '{target_phase}'. "
            f"Try increasing CPU_DISCRIM_MATCH_TOL_SEC (currently {tol_s})."
        )

    # Validate minimum workloads per rep (2 for cpu_busy_vs_noop and gpu_2pods, 3 for gpu_3pods)
    g = df.groupby("rep")["workload"].nunique()
    bad = g[g < int(min_workloads_per_rep)]
    if not bad.empty:
        ex = df[df["rep"].isin(bad.index)].sort_values(["rep", "workload"]).head(20).to_string(index=False)
        raise RuntimeError(
            f"Some reps have <{min_workloads_per_rep} matched workloads for '{target_phase}'.\nExamples:\n{ex}"
        )

    log(f"[workload_set] matched reps={df['rep'].nunique()} rows={len(df)}")
    return df.sort_values(["rep", "workload"], kind="stable").reset_index(drop=True)
