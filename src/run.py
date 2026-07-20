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
from .simulate import ControlAction

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
    global ARGO_CACHE_DIR, FCOO_CACHE_DIR, RECENT_CYCLES_FOR_VOTE

    MODELS                  = data_handler.MODELS
    DEAD_THRESHOLD          = cfg["thresholds"]["dead_after_missed_pulls"]
    OVERDUE_DAYS            = cfg["thresholds"]["overdue_days"]
    RECENT_CYCLES_FOR_VOTE  = cfg["thresholds"]["recent_cycles_for_vote"]
    STORE_DIR               = Path(cfg["paths"]["store_dir"])
    ARGO_CACHE_DIR          = Path(cfg["paths"]["argo_cache_dir"])
    FCOO_CACHE_DIR          = Path(cfg["paths"]["fcoo_cache_dir"])
    REGION                  = Region(
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
MODELS                 = data_handler.MODELS
DEAD_THRESHOLD         = 5
OVERDUE_DAYS           = 10.0
RECENT_CYCLES_FOR_VOTE = 5
STORE_DIR              = Path("data/store")
ARGO_CACHE_DIR         = Path("data/argo_cache")
FCOO_CACHE_DIR         = Path("data/fcoo_cache")
REGION                 = Region(lat_min=53.5, lat_max=60.0, lon_min=9.0, lon_max=23.0)

# Padding (degrees) around every active float's anchor + current trajectory
# tip when building each run's model-data fetch box (see _active_bounding_box)
# -- ~1 degree is roughly 60-110km at these latitudes, comfortably more than
# a float's typical multi-day drift.
FETCH_MARGIN_DEG = 1.0


def run(config_path: Path | None = None) -> None:
    _build_globals(_load_config(config_path))

    floats_db = float_store.load_floats_db(STORE_DIR)
    error_db = float_store.load_error_db(STORE_DIR)
    forecast_history_db = float_store.load_forecast_history(STORE_DIR)
    cycle_action_history_db = float_store.load_cycle_action_history(STORE_DIR)
    leg_history_db = float_store.load_leg_history(STORE_DIR)

    _backfill_surfacing_history(floats_db)

    cycle_action_rows = _refresh_cycle_actions(floats_db)
    if cycle_action_rows:
        cycle_action_history_db = pd.concat(
            [cycle_action_history_db, pd.DataFrame(cycle_action_rows)], ignore_index=True
        )

    forecast_rows = _extend_trajectories(floats_db)
    if forecast_rows:
        forecast_history_db = pd.concat(
            [forecast_history_db, pd.DataFrame(forecast_rows)], ignore_index=True
        )

    argo_now = data_handler.download_argo_floats_in_domain(REGION)
    error_db, leg_history_rows = _reconcile_with_argo(floats_db, argo_now, error_db)
    if leg_history_rows:
        leg_history_db = pd.concat(
            [leg_history_db, pd.DataFrame(leg_history_rows)], ignore_index=True
        )
    _register_new_floats(floats_db, argo_now)

    float_store.save_floats_db(STORE_DIR, floats_db)
    float_store.save_error_db(STORE_DIR, error_db)
    float_store.save_forecast_history(STORE_DIR, forecast_history_db)
    float_store.save_cycle_action_history(STORE_DIR, cycle_action_history_db)
    float_store.save_leg_history(STORE_DIR, leg_history_db)

    now = datetime.utcnow()
    web_export.export_floats(floats_db, error_db, leg_history_db, now)
    web_export.export_leaderboard(error_db)


# --------------------------------------------------------------------------- #
# Loop 1: extend simulated trajectories
# --------------------------------------------------------------------------- #

def _active_bounding_box(floats_db: dict[str, FloatRow], fallback: Region) -> Region:
    """
    Tight bounding box around every non-dead float's real anchor and current
    trajectory tip (across all models), padded by FETCH_MARGIN_DEG and
    clipped to `fallback`.

    Shrinking the model-data fetch to just where it's actually needed --
    instead of the full configured REGION every run -- cuts the payload
    size of every downloaded chunk. FCOO's OPeNDAP corruption was observed
    live to hit larger requests harder (see _FCOO_TIME_CHUNK's docstring in
    data_handler.py, and why idk uses smaller time-chunks than dk), so a
    smaller box is a plausible lever on that even though the chunk count is
    unchanged.

    Clipped to `fallback` rather than left to grow freely so a float that's
    genuinely drifted outside the model's intended coverage area still gets
    correctly excluded by the in-domain check below, instead of the box
    just silently following it out.
    """
    lats: list[float] = []
    lons: list[float] = []
    for row in floats_db.values():
        if row.is_dead:
            continue
        lat, lon, _ = row.last_real_position
        lats.append(lat)
        lons.append(lon)
        for track in row.models.values():
            if track.trajectory:
                _, tlat, tlon = track.trajectory[-1]
                lats.append(tlat)
                lons.append(tlon)

    if not lats:
        return fallback

    return Region(
        lat_min=max(fallback.lat_min, min(lats) - FETCH_MARGIN_DEG),
        lat_max=min(fallback.lat_max, max(lats) + FETCH_MARGIN_DEG),
        lon_min=max(fallback.lon_min, min(lons) - FETCH_MARGIN_DEG),
        lon_max=min(fallback.lon_max, max(lons) + FETCH_MARGIN_DEG),
    )


def _extend_trajectories(floats_db: dict[str, FloatRow]) -> list[dict]:
    """
    One model_data fetch per model -- not per float. trim_to_forecast_only
    is what runs per (float, model) pair, since each row's own trajectory
    tip defines "what this row has already consumed." That's a cheap
    in-memory filter on already-downloaded data, not a second network call,
    so this stays efficient despite the nested loop shape.

    The fetch itself is spatially restricted to _active_bounding_box rather
    than the full REGION, to shrink the FCOO corruption surface.

    Returns one forecast_history row per (float, model) that actually got
    extended this round -- what this run's freshest trajectory predicts for
    the float's next surfacing. Frozen (no-new-data) rows are skipped: a
    frozen cycle carries no new information, so recording one would just be
    a duplicate of the previous row.
    """
    now        = datetime.utcnow()
    fetch_start = now - timedelta(days=1)   # 1-day lookback so no float is gapped
    fetch_end   = now + timedelta(days=5)   # one full max-cycle horizon
    fetch_region = _active_bounding_box(floats_db, REGION)
    forecast_rows: list[dict] = []

    for model in MODELS:
        try:
            raw = data_handler.download_model_data(model, fetch_region,
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

        lat_min, lat_max, lon_min, lon_max = data_handler.model_domain_bounds(raw)

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
                    bathy_interp=_bathy_interp,
                    float_id=row.float_id,
                )
            except Exception:
                logger.warning(
                    "simulate_cycle failed for float %s model %s -- skipping this float",
                    row.float_id, model, exc_info=True,
                )
                track.missed_model_pulls += 1
                continue
            track.missed_model_pulls = 0
            track.trajectory.extend(new_points)

            surf_time = simulate.next_surfacing(anchor_time, row.cycle_action, now)
            surf_pos = simulate.lookup_position(track.trajectory, surf_time)
            if surf_pos is not None:
                forecast_rows.append({
                    "float_id": row.float_id,
                    "cycle_number": len(row.surfacing_history),
                    "forecast_name": model,
                    "forecast": now,
                    "expected_surfacing_time": surf_time,
                    "expected_surfacing_lat": surf_pos[0],
                    "expected_surfacing_lon": surf_pos[1],
                })

    return forecast_rows


# --------------------------------------------------------------------------- #
# Loop 2: reconcile against real Argo data
# --------------------------------------------------------------------------- #

def _reconcile_with_argo(
    floats_db: dict[str, FloatRow],
    argo_now: dict[str, ArgoPull],
    error_db: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Returns (updated error_db, leg_history_rows). pd.concat does not mutate
    in place, so the caller must reassign -- run() does
    `error_db, leg_history_rows = _reconcile_with_argo(...)`.

    leg_history_rows is every point of every model's trajectory that's about
    to be discarded by this call's anchor resets (see below) -- the full
    simulated path a completed forecast leg actually took, captured here
    because this is the only moment it still exists before design decision
    5's reset wipes it down to a single point. Points strictly after
    real_time are dropped (the trajectory tip usually runs ahead of "now",
    per design decision 3, so whatever it had already simulated past the
    real surfacing never corresponded to anything real).
    """
    new_rows: list[dict] = []
    leg_history_rows: list[dict] = []

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

        if real_time <= row.last_real_position[2]:
            continue  # no new surfacing since last run

        # Overdue is judged against THIS ping's timestamp, not wall-clock
        # "now" -- pipeline lag between an actual surfacing and it appearing
        # in the Argo feed shouldn't affect whether the gap counts as overdue.
        was_overdue = row.is_overdue(real_time, threshold_days=OVERDUE_DAYS)

        if not was_overdue:
            # Distance the float actually drifted since its previous confirmed
            # surfacing -- same for every model at this event, so compute it
            # once. Lets error_pct (error_m / drift_m) be derived downstream
            # without re-deriving "actual drift" from surfacing_history.
            prev_lat, prev_lon, _ = row.last_real_position
            drift_m = _haversine_error_m(prev_lat, prev_lon, real_lat, real_lon)

            for model in MODELS:
                if model not in row.models:
                    continue
                predicted = _lookup(row.models[model].trajectory, real_time)
                if predicted is None:
                    traj = row.models[model].trajectory
                    tip = traj[-1][0].isoformat() if traj else "empty"
                    logger.warning(
                        "Float %s model %s: trajectory tip (%s) is before real surfacing (%s)"
                        " -- scoring event lost; extend model data or reduce freeze gap",
                        float_id, model, tip, real_time.isoformat(),
                    )
                else:
                    error_m = _haversine_error_m(real_lat, real_lon, *predicted)
                    new_rows.append({
                        "float_id": float_id, "model": model, "t": real_time,
                        "error_m": error_m, "drift_m": drift_m,
                        "real_lat": real_lat, "real_lon": real_lon,
                        "predicted_lat": predicted[0], "predicted_lon": predicted[1],
                        # The exact anchor this cycle's forecast leg started from --
                        # not reconstructable later from surfacing_history, which is
                        # QC'd-profile-derived and denser than the actual sparse
                        # sequence of anchor resets (see map.html's use of this field).
                        "leg_start_lat": prev_lat, "leg_start_lon": prev_lon,
                    })
        # else: excluded from error_db per the 10-day rule -- still reset
        # below regardless: "if we get a new ping, great, we start from 0 again."

        row.surfacing_history.append((real_lat, real_lon, real_time))
        row.last_real_position = (real_lat, real_lon, real_time)
        for model in MODELS:
            if model not in row.models:
                continue
            for t, lat, lon in row.models[model].trajectory:
                if t <= real_time:
                    leg_history_rows.append({
                        "float_id": float_id, "model": model, "leg_end_time": real_time,
                        "t": t, "lat": lat, "lon": lon,
                    })
            row.models[model].trajectory = [(real_time, real_lat, real_lon)]
            # missed_model_pulls intentionally NOT reset here -- it tracks model-feed
            # staleness and should only clear when model data actually arrives in
            # _extend_trajectories, not on a real-float surfacing event.

    if new_rows:
        error_db = pd.concat([error_db, pd.DataFrame(new_rows)], ignore_index=True)
    return error_db, leg_history_rows


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


def _derive_cycle_action(float_id: str) -> tuple[ControlAction, list[dict]]:
    """
    Re-derive a float's representative ControlAction (park_mode, cycle_hours,
    etc) by mode-voting across only its most recent RECENT_CYCLES_FOR_VOTE
    real cycles (NOT full lifetime history -- see below), and return the raw
    per-cycle list alongside it (registration also needs the cycles list
    itself, for surfacing_history/last_real_position bootstrapping).

    Restricted to a recent window because floats get reprogrammed to a
    different cycle_hours mid-mission: confirmed on real floats 6990707 and
    7902194, which both ran a ~23h cycle from 2026-02-17/19 through
    2026-06-10, then reverted to their normal ~49h cycle on the same date
    (almost certainly a coordinated field-campaign reprogram across a float
    batch, not a data artifact -- verified against raw CYCLE_NUMBER/JULD
    timestamps). A full-history average blends that stale regime into the
    current estimate and systematically undershoots the float's current
    cycle length -- e.g. 6990707's full-history mean cycle_hours implied a
    ~37h total cycle against a true current ~49h, a 24% error that shows up
    directly as next_surfacing() predictions drifting further from the real
    surfacing time the longer a leg runs. RECENT_CYCLES_FOR_VOTE is
    deliberately small (config.toml default: 5) because a float can have as
    few as ~5 cycles in its current regime by the time this is computed.

    Always re-downloads Rtraj.nc (force_refresh=True) rather than trusting
    whatever's cached -- GDAC's Rtraj.nc grows in place as a float completes
    new real cycles, so a cached copy from this float's first registration
    would freeze the estimate at however little history existed back then.
    Called both at registration and every run thereafter
    (_refresh_cycle_actions) so the estimate keeps improving as more real
    cycles accumulate, not just once.

    Raises on genuine fetch/parse failure -- callers decide how to handle
    that (registration falls back to default_action(); a periodic refresh
    should keep the float's previous estimate rather than reset it).
    """
    rtraj_path = data_handler.download_float_history(
        float_id, cache_dir=ARGO_CACHE_DIR, force_refresh=True,
    )
    cycles = cycle_extractor.extract_cycles(rtraj_path, bathy_interp=_bathy_interp)
    if len(cycles) >= 2:
        actions = cycle_extractor.build_actions(cycles)[-RECENT_CYCLES_FOR_VOTE:]
        action = cycle_extractor.mode_vote_action(actions)
    else:
        action = cycle_extractor.default_action()
    return action, cycles


def _build_new_float_row(float_id: str, pull: ArgoPull) -> FloatRow:
    try:
        cycle_action, cycles = _derive_cycle_action(float_id)
    except Exception:
        logger.warning("History download failed for %s, using default action", float_id, exc_info=True)
        cycle_action, cycles = cycle_extractor.default_action(), []

    surfacing_history = [
        (float(c["last_lat"]), float(c["last_lon"]), datetime.fromisoformat(c["end_time"]))
        for c in cycles
    ]

    if len(cycles) >= 2:
        last_lat, last_lon = cycles[-1]["last_lat"], cycles[-1]["last_lon"]
        last_time = datetime.fromisoformat(cycles[-1]["end_time"])
    else:
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


# --------------------------------------------------------------------------- #
# Loop 4: refresh cycle_action estimates
# --------------------------------------------------------------------------- #

_CYCLE_ACTION_CHANGE_THRESHOLDS = dict(
    cycle_hours=0.5, target_depth=1.0, descent_speed_ms=0.005, ascent_speed_ms=0.005,
)


def _cycle_action_changed(old: ControlAction, new: ControlAction) -> bool:
    if old.park_mode != new.park_mode:
        return True
    for field, thresh in _CYCLE_ACTION_CHANGE_THRESHOLDS.items():
        old_v, new_v = getattr(old, field), getattr(new, field)
        if (old_v is None) != (new_v is None):
            return True
        if old_v is not None and abs(old_v - new_v) > thresh:
            return True
    return False


def _refresh_cycle_actions(floats_db: dict[str, FloatRow]) -> list[dict]:
    """
    Re-derive every alive float's cycle_action from its full, freshly
    re-fetched real cycle history -- every run, not just at registration --
    so the estimate keeps improving as more real cycles accumulate rather
    than staying frozen at whatever was known when the float was first seen.

    Runs before _extend_trajectories so this run's own simulation already
    uses the freshest estimate, not last run's.

    Logs a line when a float's action actually changes (park_mode or any
    numeric field beyond a small tolerance) -- not every run, to avoid
    spamming identical values. Returns one cycle_action_history row per
    float successfully (re)computed this run regardless of whether it
    changed, for save_cycle_action_history to persist the full timeline.
    A failed refresh (network/parse error) keeps that float's previous
    cycle_action untouched rather than resetting it to default_action() --
    unlike registration, an established float shouldn't lose a good
    estimate over one transient fetch failure.
    """
    now = datetime.utcnow()
    log_rows: list[dict] = []

    for row in floats_db.values():
        if row.is_dead:
            continue
        try:
            new_action, _ = _derive_cycle_action(row.float_id)
        except Exception:
            logger.warning(
                "Could not refresh cycle_action for %s -- keeping previous estimate",
                row.float_id, exc_info=True,
            )
            continue

        changed = _cycle_action_changed(row.cycle_action, new_action)
        if changed:
            logger.info(
                "Float %s cycle_action changed: park_mode %s -> %s, cycle_hours %.1f -> %.1f, "
                "target_depth %s -> %s",
                row.float_id, row.cycle_action.park_mode, new_action.park_mode,
                row.cycle_action.cycle_hours, new_action.cycle_hours,
                row.cycle_action.target_depth, new_action.target_depth,
            )
        row.cycle_action = new_action

        log_rows.append({
            "float_id": row.float_id,
            "logged_at": now,
            "park_mode": new_action.park_mode,
            "cycle_hours": new_action.cycle_hours,
            "transmission_duration_minutes": new_action.transmission_duration_minutes,
            "target_depth": new_action.target_depth,
            "descent_speed_ms": new_action.descent_speed_ms,
            "ascent_speed_ms": new_action.ascent_speed_ms,
            "changed": changed,
        })

    return log_rows


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


_bathy_interp_fn = None


def _load_bathy_interp():
    """
    Build a (lat, lon) -> depth_m callable from the EMODnet D6 bathymetry file
    at STORE_DIR/D6_2024.nc.  Loads once and is cached in _bathy_interp_fn.
    Falls back to a fixed 55 m if the file is absent.
    """
    import numpy as np
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    bathy_path = STORE_DIR / "D6_2024.nc"
    if not bathy_path.exists():
        logger.warning(
            "D6 bathymetry not found at %s -- falling back to fixed 55 m "
            "(park_on_bottom classification will be unreliable)",
            bathy_path,
        )
        return lambda lat, lon: 55.0

    logger.info("Loading EMODnet D6 bathymetry from %s (one-time)", bathy_path)
    with xr.open_dataset(bathy_path, mask_and_scale=True) as ds:
        lats = ds["lat"].values.astype(np.float64)
        lons = ds["lon"].values.astype(np.float64)
        elev = ds["elevation"].values.astype(np.float32)  # (lat, lon), <0 = ocean

    # Depth in metres: positive for ocean, 55 m sentinel for land / NaN
    depth = np.where((elev < 0) & ~np.isnan(elev), -elev, np.float32(55.0))

    # RegularGridInterpolator requires strictly monotone coordinate axes
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        depth = depth[::-1, :]
    if lons[0] > lons[-1]:
        lons = lons[::-1]
        depth = depth[:, ::-1]

    interp = RegularGridInterpolator(
        (lats, lons), depth,
        method="linear", bounds_error=False, fill_value=55.0,
    )
    logger.info("D6 bathymetry ready (%d lat × %d lon)", len(lats), len(lons))
    return lambda lat, lon: float(interp([[lat, lon]])[0])


def _bathy_interp(lat: float, lon: float) -> float:
    """Bathymetric depth (m) at (lat, lon), from EMODnet D6 bathymetry."""
    global _bathy_interp_fn
    if _bathy_interp_fn is None:
        _bathy_interp_fn = _load_bathy_interp()
    return _bathy_interp_fn(lat, lon)


def _haversine_error_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return cycle_extractor.haversine_km(lat1, lon1, lat2, lon2) * 1000.0


if __name__ == "__main__":
    run()
