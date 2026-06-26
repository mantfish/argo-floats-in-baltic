"""
run.py
=======
Orchestrates one full leaderboard cycle:

    1. Extend every (float, model) trajectory using each model's latest
       forecast-only data (freeze in place if there's nothing new beyond
       where that row already is).
    2. Reconcile against the current real Argo population: score confirmed
       surfacings, bump missed_pulls for absent floats, kill floats past
       DEAD_THRESHOLD.
    3. Register any float seen in the Argo pull that isn't in floats_db yet.
    4. Persist, display.

This module owns the policy constants (DEAD_THRESHOLD, OVERDUE_DAYS, REGION)
-- everything it calls (data_handler, float_store, cycle_extractor, simulate)
is policy-free, so changing a threshold here never requires touching those.

The add_new_float-equivalent logic (_build_new_float_row) lives here, not in
float_store.py, because it fetches (download_float_history) and computes
(extract_cycles) -- float_store.py's whole point is to stay pure load/save
with no side effects of its own.
"""

from __future__ import annotations

import logging
import tomllib
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from . import cycle_extractor, data_handler, float_store, simulate, web_export
from .data_handler import ArgoPull, Region
from .float_store import FloatRow, ModelTrack

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def _load_config(config_path: Path | None = None) -> dict:
    """Load config.toml from the project root (two levels up from this file)."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.toml"
    with open(config_path, "rb") as fh:
        return tomllib.load(fh)


def _build_globals(cfg: dict) -> None:
    """Populate module-level constants from a loaded config dict."""
    global MODELS, DEAD_THRESHOLD, OVERDUE_DAYS, STORE_DIR, REGION
    global ARGO_CACHE_DIR, FCOO_CACHE_DIR

    MODELS          = data_handler.MODELS
    DEAD_THRESHOLD  = cfg["thresholds"]["dead_after_missed_pulls"]
    OVERDUE_DAYS    = cfg["thresholds"]["overdue_days"]
    STORE_DIR       = Path(cfg["paths"]["store_dir"])
    ARGO_CACHE_DIR  = Path(cfg["paths"]["argo_cache_dir"])
    FCOO_CACHE_DIR  = Path(cfg["paths"]["fcoo_cache_dir"])
    REGION          = Region(
        lat_min=cfg["region"]["lat_min"],
        lat_max=cfg["region"]["lat_max"],
        lon_min=cfg["region"]["lon_min"],
        lon_max=cfg["region"]["lon_max"],
    )

    # Push data_handler constants that live in config
    data_handler.CMEMS_DATASET_ID = cfg["cmems"]["dataset_id"]
    data_handler.CMEMS_DEPTH_MAX  = cfg["cmems"]["max_depth_m"]
    data_handler.ARGO_CACHE_DIR   = ARGO_CACHE_DIR


# Defaults (overwritten by run() via config)
MODELS         = data_handler.MODELS
DEAD_THRESHOLD = 5
OVERDUE_DAYS   = 10.0
STORE_DIR      = Path("data/store")
ARGO_CACHE_DIR = Path("data/argo_cache")
FCOO_CACHE_DIR = Path("data/fcoo_cache")
REGION         = Region(lat_min=53.5, lat_max=60.0, lon_min=9.0, lon_max=23.0)


def run(config_path: Path | None = None) -> None:
    _build_globals(_load_config(config_path))

    floats_db = float_store.load_floats_db(STORE_DIR)
    error_db = float_store.load_error_db(STORE_DIR)

    _backfill_surfacing_history(floats_db)
    _extend_trajectories(floats_db)

    argo_now = data_handler.download_argo_floats_in_domain(REGION)
    error_db = _reconcile_with_argo(floats_db, argo_now, error_db)
    _register_new_floats(floats_db, argo_now)

    float_store.save_floats_db(STORE_DIR, floats_db)
    float_store.save_error_db(STORE_DIR, error_db)

    now = datetime.utcnow()
    web_export.export_floats(floats_db, now)
    web_export.export_leaderboard(error_db)


# --------------------------------------------------------------------------- #
# Loop 1: extend simulated trajectories
# --------------------------------------------------------------------------- #

def _extend_trajectories(floats_db: dict[str, FloatRow]) -> None:
    """
    One model_data fetch per model -- not per float. trim_to_forecast_only
    is what runs per (float, model) pair, since each row's own trajectory
    tip defines "what this row has already consumed." That's a cheap
    in-memory filter on already-downloaded data, not a second network call,
    so this stays efficient despite the nested loop shape.
    """
    now        = datetime.utcnow()
    fetch_start = now - timedelta(days=1)   # 1-day lookback so no float is gapped
    fetch_end   = now + timedelta(days=5)   # one full max-cycle horizon

    for model in MODELS:
        try:
            raw = data_handler.download_model_data(model, REGION,
                                                   issue_time=fetch_start,
                                                   end_time=fetch_end)
        except Exception:
            logger.warning(
                "download_model_data failed for %s -- freezing all rows this round",
                model, exc_info=True,
            )
            for row in floats_db.values():
                if not row.is_dead and model in row.models:
                    row.models[model].missed_model_pulls += 1
            continue

        lat_min = float(raw.lat.min())
        lat_max = float(raw.lat.max())
        lon_min = float(raw.lon.min())
        lon_max = float(raw.lon.max())

        for row in floats_db.values():
            if row.is_dead:
                continue

            anchor_lat, anchor_lon, anchor_time = row.last_real_position
            in_domain = lat_min <= anchor_lat <= lat_max and lon_min <= anchor_lon <= lon_max
            if not in_domain:
                row.models.pop(model, None)
                continue
            if model not in row.models:
                row.models[model] = ModelTrack(trajectory=[(anchor_time, anchor_lat, anchor_lon)])

            track = row.models[model]
            issue_time = track.trajectory[-1][0]  # invariant: trajectory always has >= 1 point
            model_data = data_handler.trim_to_forecast_only(raw, issue_time)

            if _is_empty(model_data):
                # Nothing beyond where this row already is -- freeze-on-gap.
                track.missed_model_pulls += 1
                continue

            track.missed_model_pulls = 0
            tip_time, tip_lat, tip_lon = track.trajectory[-1]
            anchor_time, anchor_lat, anchor_lon = track.trajectory[0]

            try:
                new_points = simulate.simulate_cycle(
                    model_data=model_data,
                    control_action=row.cycle_action,
                    anchor_lat=anchor_lat,
                    anchor_lon=anchor_lon,
                    anchor_time=anchor_time,
                    tip_lat=tip_lat,
                    tip_lon=tip_lon,
                    tip_time=tip_time,
                    until_time=_last_timestamp(model_data),
                )
            except Exception:
                logger.warning(
                    "simulate_cycle failed for float %s model %s -- skipping this float",
                    row.float_id, model, exc_info=True,
                )
                continue
            track.trajectory.extend(new_points)


# --------------------------------------------------------------------------- #
# Loop 2: reconcile against real Argo data
# --------------------------------------------------------------------------- #

def _reconcile_with_argo(
    floats_db: dict[str, FloatRow],
    argo_now: dict[str, ArgoPull],
    error_db: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns the updated error_db. pd.concat does not mutate in place, so the
    caller must reassign -- run() does `error_db = _reconcile_with_argo(...)`.
    """
    new_rows: list[dict] = []

    for float_id, row in floats_db.items():
        if row.is_dead:
            continue

        if float_id not in argo_now:
            row.missed_pulls += 1
            if row.missed_pulls >= DEAD_THRESHOLD:
                row.is_dead = True
            continue

        row.missed_pulls = 0
        pull = argo_now[float_id]
        real_lat, real_lon = pull.last_position
        real_time = pull.last_time

        # NOTE(Claude): exact tuple equality, matching your original sketch.
        # Fine if Argo positions are quantized consistently pull-to-pull;
        # flag if you'd rather key off real_time instead.
        if (real_lat, real_lon) == row.last_real_position[:2]:
            continue  # no new surfacing since last run

        # Overdue is judged against THIS ping's timestamp, not wall-clock
        # "now" -- pipeline lag between an actual surfacing and it appearing
        # in the Argo feed shouldn't affect whether the gap counts as overdue.
        was_overdue = row.is_overdue(real_time, threshold_days=OVERDUE_DAYS)

        if not was_overdue:
            for model in MODELS:
                if model not in row.models:
                    continue
                predicted = _lookup(row.models[model].trajectory, real_time)
                if predicted is not None:
                    error_m = _haversine_error_m(real_lat, real_lon, *predicted)
                    new_rows.append(
                        {"float_id": float_id, "model": model, "t": real_time, "error_m": error_m}
                    )
        # else: excluded from error_db per the 10-day rule -- still reset
        # below regardless: "if we get a new ping, great, we start from 0 again."

        row.surfacing_history.append((real_lat, real_lon, real_time))
        row.last_real_position = (real_lat, real_lon, real_time)
        for model in MODELS:
            if model not in row.models:
                continue
            row.models[model].trajectory = [(real_time, real_lat, real_lon)]
            row.models[model].missed_model_pulls = 0

    if new_rows:
        error_db = pd.concat([error_db, pd.DataFrame(new_rows)], ignore_index=True)
    return error_db


# --------------------------------------------------------------------------- #
# Loop 3: register new floats
# --------------------------------------------------------------------------- #

def _register_new_floats(floats_db: dict[str, FloatRow], argo_now: dict[str, ArgoPull]) -> None:
    for float_id, pull in argo_now.items():
        if float_id in floats_db:
            continue
        try:
            floats_db[float_id] = _build_new_float_row(float_id, pull)
        except Exception:
            logger.warning("Could not register float %s -- will retry next run", float_id, exc_info=True)


def _build_new_float_row(float_id: str, pull: ArgoPull) -> FloatRow:
    try:
        rtraj_path = data_handler.download_float_history(float_id)
        cycles = cycle_extractor.extract_cycles(rtraj_path, bathy_interp=_bathy_interp)
    except Exception:
        logger.warning("History download failed for %s, using default action", float_id, exc_info=True)
        cycles = []

    surfacing_history = [
        (float(c["last_lat"]), float(c["last_lon"]), datetime.fromisoformat(c["end_time"]))
        for c in cycles
    ]

    if len(cycles) >= 2:
        actions = cycle_extractor.build_actions(cycles)
        cycle_action = cycle_extractor.mode_vote_action(actions)
        last_lat, last_lon = cycles[-1]["last_lat"], cycles[-1]["last_lon"]
        last_time = datetime.fromisoformat(cycles[-1]["end_time"])
    else:
        cycle_action = cycle_extractor.default_action()
        last_lat, last_lon = pull.last_position
        last_time = pull.last_time

    return FloatRow(
        float_id=float_id,
        cycle_action=cycle_action,
        last_real_position=(last_lat, last_lon, last_time),
        missed_pulls=0,
        is_dead=False,
        models={m: ModelTrack(trajectory=[(last_time, last_lat, last_lon)]) for m in MODELS},
        surfacing_history=surfacing_history,
    )


def _backfill_surfacing_history(floats_db: dict[str, FloatRow]) -> None:
    """One-time backfill for floats loaded from store before surfacing_history was added."""
    for row in floats_db.values():
        if row.surfacing_history:
            continue
        rtraj_path = ARGO_CACHE_DIR / f"{row.float_id}_Rtraj.nc"
        if not rtraj_path.exists():
            continue
        try:
            cycles = cycle_extractor.extract_cycles(rtraj_path, bathy_interp=_bathy_interp)
            row.surfacing_history = [
                (float(c["last_lat"]), float(c["last_lon"]), datetime.fromisoformat(c["end_time"]))
                for c in cycles
            ]
            logger.info("Backfilled %d surfacings for float %s", len(row.surfacing_history), row.float_id)
        except Exception:
            logger.debug("Could not backfill surfacing history for %s", row.float_id, exc_info=True)


# --------------------------------------------------------------------------- #
# Small helpers -- model_data-shape-dependent, left for you to fill in
# --------------------------------------------------------------------------- #

def _is_empty(model_data) -> bool:
    """True if `model_data` has no timestamps left after trimming."""
    return model_data.time.size == 0


def _last_timestamp(model_data) -> datetime:
    """Latest timestamp present in `model_data`."""
    return pd.Timestamp(model_data.time.values[-1]).to_pydatetime().replace(tzinfo=None)


def _lookup(trajectory: list[tuple[datetime, float, float]], t: datetime):
    return simulate.lookup_position(trajectory, t)


def _bathy_interp(lat: float, lon: float) -> float:
    """
    Bathymetric depth (m) at (lat, lon).

    MVP: returns a fixed Baltic representative depth of 55 m.
    This is only used to classify park_on_bottom during cycle extraction
    (cycle_extractor.DEPTH_FRAC_THRESH = 2.0), so a constant is adequate
    until a real bathymetry dataset (e.g. GEBCO) is wired up.
    """
    return 55.0


def _haversine_error_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return cycle_extractor.haversine_km(lat1, lon1, lat2, lon2) * 1000.0


if __name__ == "__main__":
    run()
