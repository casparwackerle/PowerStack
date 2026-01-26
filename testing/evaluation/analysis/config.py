from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # .../testing/evaluation/analysis
EVAL_DIR = BASE_DIR.parent                          # .../testing/evaluation
LOGS_DIR = EVAL_DIR / "logs"
OUT_DIR = BASE_DIR / "out"


# ---------------------------------------------------------------------
# Which runs to process (local folders under logs/)
# ---------------------------------------------------------------------
RUNS = {
    "TYCHO_IDLE": LOGS_DIR / "tycho-testing-TYCHO-IDLE",
    "TYCHO_BUSY": LOGS_DIR / "tycho-testing-TYCHO-BUSY",
    # "KEPLER_IDLE": LOGS_DIR / "tycho-testing-KEPLER-IDLE",  # unused for now
}

# Select which run(s) to analyze when pressing play.
SELECTED_RUN_KEYS = [
    "TYCHO_IDLE",
    "TYCHO_BUSY",
]


# ---------------------------------------------------------------------
# Which tests to execute
# ---------------------------------------------------------------------
TESTS_TO_RUN = [
    # "idle",
    "cpu_discrimination",
]


# ---------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------
PROM_URL = "http://newnan.zhaw.ch:30002"   # change if needed
PROM_TIMEOUT_SEC = 60
PROM_MAX_POINTS = 11000
DEFAULT_STEP_SEC = 3

PROM_USE_SOCKS = True
PROM_SOCKS_PROXY = "socks5h://127.0.0.1:12345"

# Prometheus: ALWAYS restrict to the node under test
PROM_INSTANCE = 'irvine.zhaw.ch:9102'


# ---------------------------------------------------------------------
# Time window policy
# ---------------------------------------------------------------------
# For idle baseline: ignore first/last seconds of the phase to avoid startup/teardown effects.
IDLE_SETTLE_SEC = 10
IDLE_GUARD_SEC = 10

# For CPU discrimination: ignore first/last seconds of the pod lifetime window.
CPU_DISCRIM_SETTLE_SEC = 10
CPU_DISCRIM_GUARD_SEC = 10

# CPU discrimination: analyze these master phases (one plot per phase)
CPU_DISCRIM_TARGET_PHASES = [
    "cpu_discrimination_concurrent_alt",
    "cpu_discrimination_concurrent",
]

# How far subrun timestamps may drift relative to master phase window
CPU_DISCRIM_MATCH_TOL_SEC = 6

# ---------------------------------------------------------------------
# Tycho metric mapping
# ---------------------------------------------------------------------
# Keep this small: start with the few metrics needed for idle + cpu_discrimination.
TYCHO_METRICS = {
    # ------------------------
    # Idle test (existing)
    # ------------------------
    "rapl_pkg_total_energy_mj": f'tycho_rapl_energy_mj{{instance="{PROM_INSTANCE}",domain="pkg",kind="total",source="rapl"}}',
    "rapl_pkg_dynamic_energy_mj": f'tycho_rapl_energy_mj{{instance="{PROM_INSTANCE}",domain="pkg",kind="dynamic",source="rapl"}}',
    "rapl_pkg_idle_energy_mj": f'tycho_rapl_energy_mj{{instance="{PROM_INSTANCE}",domain="pkg",kind="idle",source="rapl"}}',

    "gpu_total_energy_mj": f'tycho_gpu_energy_mj{{instance="{PROM_INSTANCE}",kind="total",source="nvml_corrected"}}',
    "gpu_dynamic_energy_mj": f'tycho_gpu_energy_mj{{instance="{PROM_INSTANCE}",kind="dynamic",source="nvml_corrected"}}',
    "gpu_idle_energy_mj": f'tycho_gpu_energy_mj{{instance="{PROM_INSTANCE}",kind="idle",source="nvml_corrected"}}',

    "residual_total_energy_mj": f'tycho_residual_energy_mj{{instance="{PROM_INSTANCE}",chassis="Self",kind="total",source="redfish_corrected"}}',
    "residual_idle_energy_mj": f'tycho_residual_energy_mj{{instance="{PROM_INSTANCE}",chassis="Self",kind="idle",source="redfish_corrected"}}',
    "residual_dynamic_energy_mj": f'tycho_residual_energy_mj{{instance="{PROM_INSTANCE}",chassis="Self",kind="dynamic",source="redfish_corrected"}}',
    "system_total_energy_mj": f'tycho_system_energy_mj{{instance="{PROM_INSTANCE}",chassis="Self",kind="total",source="redfish_corrected"}}',

    "residual_window_usable": f'tycho_residual_window_usable{{instance="{PROM_INSTANCE}",chassis="Self"}}',

    # ------------------------
    # CPU discrimination (new)
    # core domain only, workload-attributed counters
    # NOTE: no pod filter here. We filter by pod label in the test.
    # ------------------------
    "workload_rapl_core_idle_energy_mj": (
        f'tycho_workload_rapl_energy_mj{{instance="{PROM_INSTANCE}",domain="core",kind="idle",source="rapl"}}'
    ),
    "workload_rapl_core_dynamic_energy_mj": (
        f'tycho_workload_rapl_energy_mj{{instance="{PROM_INSTANCE}",domain="core",kind="dynamic",source="rapl"}}'
    ),
}


# ---------------------------------------------------------------------
# Figure configuration (keep global to stay consistent)
# ---------------------------------------------------------------------
FIG_DPI = 200
