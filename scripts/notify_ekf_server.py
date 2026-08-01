"""Notifies the EKF/MPC piloting server (github.com/mantfish/EKF_MPC_profiling_float_control)
whenever a float it pilots has genuinely surfaced since the last pipeline run, and keeps its
EKF state fresh for every float it pilots -- every time this project's own leaderboard cron runs
(see .github/workflows/run-pipeline.yml).

The EKF server is a separate, deliberately decoupled project (see CLAUDE.md's "What this is").
This script talks to it only over its existing public HTTP API (/update_state, /return_action)
-- never importing its code -- so nothing here creates a code-level coupling between the repos.

"New surfacing" is determined entirely from this project's own git history: data/store/
floats_meta.parquet is committed every pipeline run, so the previous run's last_lat/last_lon/
last_time per float is one `git show HEAD:...` away. No need to ask the server what it already
knows -- if a float's last_time moved forward since the last commit, it surfaced for real this
run.

Run from the repo root (matches how run-pipeline.yml invokes it): `uv run scripts/notify_ekf_server.py`
"""
from __future__ import annotations

import io
import json
import logging
import os
import subprocess
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
FLOATS_META_RELPATH = "data/store/floats_meta.parquet"
FLOATS_META_PATH = REPO_ROOT / FLOATS_META_RELPATH

logger = logging.getLogger(__name__)


def _load_before_floats_meta() -> pd.DataFrame:
    """The last *committed* floats_meta.parquet -- i.e. before this run's changes.

    Returns an empty DataFrame (rather than raising) if there's no prior commit touching
    this file, so a bootstrap run just treats every float as new instead of failing.
    """
    try:
        raw = subprocess.run(
            ["git", "show", f"HEAD:{FLOATS_META_RELPATH}"],
            cwd=REPO_ROOT, capture_output=True, check=True,
        ).stdout
        return pd.read_parquet(io.BytesIO(raw))
    except Exception:
        logger.warning("No prior committed floats_meta.parquet found; treating every float as new.")
        return pd.DataFrame(columns=["float_id", "last_lat", "last_lon", "last_time"])


def _is_new_surfacing(float_id: str, last_time, before: pd.DataFrame) -> bool:
    prior_rows = before[before["float_id"] == float_id]
    if prior_rows.empty:
        return True
    prior_last_time = pd.Timestamp(prior_rows.iloc[0]["last_time"])
    return pd.Timestamp(last_time) > prior_last_time


def _notify_one(base_url: str, float_id: str, last_lat: float, last_lon: float, last_time, is_new: bool) -> None:
    update_resp = requests.post(f"{base_url}/update_state", params={"float_id": float_id}, timeout=300)
    if update_resp.status_code == 404:
        logger.info("Float %s is not piloted by the EKF server, skipping.", float_id)
        return
    update_resp.raise_for_status()
    logger.info("Refreshed EKF state for float %s.", float_id)

    if not is_new:
        logger.info("Float %s: no new surfacing since the last pipeline run.", float_id)
        return

    payload = {
        "float_id": int(float_id),
        "location": {"latitude": float(last_lat), "longitude": float(last_lon)},
        "time_of_transmission": pd.Timestamp(last_time).isoformat(),
    }
    files = {"file": ("surfacing.json", json.dumps(payload), "application/json")}
    trigger_resp = requests.post(f"{base_url}/return_action", files=files, timeout=300)
    trigger_resp.raise_for_status()
    logger.info(
        "Float %s surfaced at %s -- EKF server selected action: %s",
        float_id, payload["time_of_transmission"], trigger_resp.json().get("action"),
    )


def notify_all() -> None:
    base_url = os.environ["EKF_API_BASE_URL"].rstrip("/")

    if not FLOATS_META_PATH.exists():
        logger.warning("No floats_meta.parquet found at %s, nothing to notify.", FLOATS_META_PATH)
        return

    after = pd.read_parquet(FLOATS_META_PATH)
    before = _load_before_floats_meta()

    for _, row in after.iterrows():
        float_id = str(row.float_id)
        try:
            is_new = _is_new_surfacing(float_id, row.last_time, before)
            _notify_one(base_url, float_id, row.last_lat, row.last_lon, row.last_time, is_new)
        except Exception:
            logger.exception("Failed to notify EKF server about float %s, continuing.", float_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    notify_all()
