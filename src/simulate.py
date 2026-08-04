"""
simulate.py
============
Deterministic forward integration of a profiling float's position under a
given ocean current model -- used to extend each (float, model) trajectory
in run.py's loop 1.

Adapted from particle_mover.py's simulate_estimate_forward/simulate_real,
but deliberately simpler:

    - No EKF covariance propagation (P, Q). The leaderboard doesn't need an
      uncertainty estimate, only a position to score against real Argo
      positions.
    - No bias state, no process noise. There's no "real" float to simulate
      here -- model_data IS the hypothesis being tested, so adding synthetic
      noise on top of it would only obscure what we're scoring.
    - Repeats the dive profile indefinitely across however many full cycles
      fit inside the trimmed model_data window, rather than running for one
      fixed control_action.duration_hours and stopping. That's the
      "keep advecting, no surfaced flag" design: an overdue float just keeps
      profiling on the same cycle_action until a real ping resets the anchor.

Phase (descending / parking / ascending / communicating) is NOT stored
anywhere -- it's recovered each call from elapsed time since the
trajectory's anchor point, modulo one full cycle duration. That only works
because anchor resets happen exactly at confirmed real surfacings (see
float_store.FloatRow / run.py's reconciliation step). If that invariant
ever breaks, phase recovery here breaks with it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

_DESCENDING = "descending"
_PARKING = "parking"
_ASCENDING = "ascending"
_COMMUNICATING = "communicating"


@dataclass(frozen=True)
class ControlAction:
    """
    One float's representative profiling-cycle parameters.

    Field names follow cycle_extractor.action_from_cycle()'s output, NOT
    your existing sim_types.ControlAction from the EKF/MPC piloting project
    (which uses duration_hours/parking_depth instead of cycle_hours/
    target_depth). Different projects, deliberately not unified here --
    flag if you'd rather these share one schema.

    cycle_hours is descent + parking time ONLY -- ascent and surface
    transmission are already subtracted out by the time action_from_cycle
    produces it. simulate_cycle() reconstructs the full repeat period from
    this plus the speed/transmission fields; don't read cycle_hours as the
    full cycle length anywhere else without accounting for that.
    """
    park_mode: str                          # "park_on_bottom" | "parking_depth" | "drift_on_surface"
    cycle_hours: float                       # descent + parking time only, see above
    transmission_duration_minutes: float
    target_depth: Optional[float]            # dbar; None for drift_on_surface
    descent_speed_ms: float
    ascent_speed_ms: float


def latlon_to_xy(lat: float, lon: float, anchor_lat: float, anchor_lon: float) -> tuple[float, float]:
    """Local planar (x, y) in meters, relative to (anchor_lat, anchor_lon)."""
    y = (lat - anchor_lat) * 111_000.0
    x = (lon - anchor_lon) * 111_000.0 * math.cos(math.radians(anchor_lat))
    return x, y


def xy_to_latlon(x: float, y: float, anchor_lat: float, anchor_lon: float) -> tuple[float, float]:
    lat = anchor_lat + y / 111_000.0
    lon = anchor_lon + x / (111_000.0 * math.cos(math.radians(anchor_lat)))
    return lat, lon


def _grid_interpolators(
    t_s: np.ndarray, depth: np.ndarray, lat: np.ndarray, lon: np.ndarray,
    u_arr: np.ndarray, v_arr: np.ndarray, label: str = "",
) -> tuple[Callable, Callable, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
    """
    Build (interp_u, interp_v, bounds) for one regular (time, depth, lat, lon)
    grid. bounds is ((depth_min, depth_max), (lat_min, lat_max), (lon_min,
    lon_max)) -- used by the multi-grid 'fcoo' path to decide which grid's
    interpolator covers a given query point.
    """
    from scipy.interpolate import RegularGridInterpolator

    # float32 halves the memory vs float64; position errors at km scale
    # don't require mm/s current precision.
    u_arr = u_arr.astype(np.float32)
    v_arr = v_arr.astype(np.float32)

    # Remember which cells were NaN (land/mask) before the zero-fill below,
    # so _warn_on_zero_fill can tell "grid says genuinely no current" apart
    # from "query landed on a masked cell that reads as 0.0" -- otherwise
    # both look identical to a caller, and a persistent masked-cell query
    # (e.g. a float near a coastline) produces silent exact-zero advection.
    masked = np.isnan(u_arr) | np.isnan(v_arr)

    # Replace NaN (land/mask) with 0 so interpolation doesn't propagate NaNs
    u_arr = np.where(np.isnan(u_arr), np.float32(0.0), u_arr)
    v_arr = np.where(np.isnan(v_arr), np.float32(0.0), v_arr)

    # Ensure coordinate arrays are strictly monotone (required by RGI)
    if depth[0] > depth[-1]:
        depth = depth[::-1]
        u_arr = u_arr[:, ::-1, :, :]
        v_arr = v_arr[:, ::-1, :, :]
        masked = masked[:, ::-1, :, :]

    interp_u = RegularGridInterpolator(
        (t_s, depth, lat, lon), u_arr,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    interp_v = RegularGridInterpolator(
        (t_s, depth, lat, lon), v_arr,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    interp_u, interp_v = _warn_on_zero_fill(interp_u, interp_v, t_s, depth, lat, lon, masked, label)
    bounds = ((float(depth.min()), float(depth.max())),
              (float(lat.min()), float(lat.max())),
              (float(lon.min()), float(lon.max())))
    return interp_u, interp_v, bounds


def _warn_on_zero_fill(
    interp_u: Callable, interp_v: Callable,
    t_s: np.ndarray, depth: np.ndarray, lat: np.ndarray, lon: np.ndarray,
    masked: np.ndarray, label: str,
) -> tuple[Callable, Callable]:
    """
    Wrap a grid's raw (u, v) callables so a query whose time or lat/lon
    falls outside this grid's real footprint, or whose nearest real cell
    was NaN (land/mask) before being zero-filled in _grid_interpolators,
    logs one warning each -- the counterpart to _taper_to_seabed's existing
    depth-overrun warning, which only covers the deep-query case. Without
    this, a query that always resolves to fill_value=0.0 or a masked cell
    is indistinguishable from a real "model predicts zero current" result
    at every call site -- exactly the ambiguity behind an fcoo track
    showing suspiciously frequent exact-zero drift.

    The time check matters most for FCOO: each fetched GETM run only
    covers a ~55h forecast horizon (56 hourly steps), refreshed every 6h,
    against a float cycle that can run close to that same length -- once a
    simulated trajectory's clock runs past the fetched data's last
    timestamp, every subsequent step silently gets zero velocity here
    rather than an error, freezing the trajectory's position while its
    timestamps keep advancing (simulate_cycle has no "stop" condition).

    The masked-cell check uses the grid's first time slice as a proxy for
    the (in practice time-invariant) land mask -- fine for a one-time
    diagnostic warning, not meant to be exact per-timestep.
    """
    t_min, t_max = float(t_s.min()), float(t_s.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())
    warned_time = False
    warned_oob = False
    warned_masked = False

    def _wrap(interp_raw: Callable) -> Callable:
        def query(rows):
            nonlocal warned_time, warned_oob, warned_masked
            rows = np.asarray(rows, dtype=np.float64)
            out = np.asarray(interp_raw(rows), dtype=np.float64)

            time_oob = (rows[:, 0] < t_min) | (rows[:, 0] > t_max)
            if np.any(time_oob) and not warned_time:
                over_h = (float(rows[time_oob, 0].max()) - t_max) / 3600.0
                logger.warning(
                    "%s: query time outside fetched forecast window "
                    "(%.1fh past the last available timestep) -- "
                    "fill_value=0.0 used from there on",
                    label, over_h,
                )
                warned_time = True

            oob = (time_oob |
                   (rows[:, 2] < lat_min) | (rows[:, 2] > lat_max) |
                   (rows[:, 3] < lon_min) | (rows[:, 3] > lon_max))
            if np.any(oob & ~time_oob) and not warned_oob:
                logger.warning(
                    "%s: query lat/lon outside grid footprint (lat "
                    "%.4f-%.4f, lon %.4f-%.4f) -- fill_value=0.0 used",
                    label, lat_min, lat_max, lon_min, lon_max,
                )
                warned_oob = True

            in_bounds = ~oob
            if not warned_masked and np.any(in_bounds):
                zi = np.clip(np.searchsorted(depth, rows[in_bounds, 1]), 0, len(depth) - 1)
                yi = np.clip(np.searchsorted(lat, rows[in_bounds, 2]), 0, len(lat) - 1)
                xi = np.clip(np.searchsorted(lon, rows[in_bounds, 3]), 0, len(lon) - 1)
                if np.any(masked[0, zi, yi, xi]):
                    logger.warning(
                        "%s: query landed on a masked/land grid cell -- "
                        "treated as 0.0 current", label,
                    )
                    warned_masked = True
            return out
        return query

    return _wrap(interp_u), _wrap(interp_v)


def _taper_to_seabed(
    interp_u: Callable, interp_v: Callable,
    depth_max: float,
    bathy_interp: Optional[Callable[[float, float], float]],
    label: str,
) -> tuple[Callable, Callable]:
    """
    Wrap a grid's raw (u, v) callables so a query deeper than `depth_max`
    (the grid's deepest real level) linearly tapers the current from its
    value at `depth_max` down to 0 at the local seabed (`bathy_interp(lat,
    lon)`), rather than the interpolator's own hard `fill_value=0.0` cutoff.
    Motivated by bottom friction -- currents go to zero at the seabed, not
    at whatever depth a particular data source happens to stop having
    levels (see config.toml's CMEMS max_depth_m history: a grid's own depth
    ceiling silently zeroing an entire float's parking-phase current is
    exactly the failure mode this replaces). A query at or beyond the
    seabed itself is clipped to 0. Only the deep side is handled here --
    shallow-side out-of-range queries keep the interpolator's normal
    fill_value=0.0.

    Logs one warning (not per-row) the first time this triggers, so this
    class of bug is visible instead of silent.

    If `bathy_interp` is None (no bathymetry available to the caller),
    returns (interp_u, interp_v) unchanged -- old zero-fill behavior.
    """
    if bathy_interp is None:
        return interp_u, interp_v

    warned = False

    def _wrap(interp_raw: Callable) -> Callable:
        def query(rows):
            nonlocal warned
            rows = np.asarray(rows, dtype=np.float64)
            out = np.asarray(interp_raw(rows), dtype=np.float64)
            deep = rows[:, 1] > depth_max
            if np.any(deep):
                if not warned:
                    logger.warning(
                        "%s: query depth exceeds grid's %.0fm range (deepest "
                        "offending query: %.0fm) -- tapering toward seabed "
                        "instead of zero-filling",
                        label, depth_max, float(rows[deep, 1].max()),
                    )
                    warned = True
                for i in np.nonzero(deep)[0]:
                    t, z, lat, lon = rows[i]
                    seabed = bathy_interp(float(lat), float(lon))
                    if seabed > depth_max:
                        frac = min(max((z - depth_max) / (seabed - depth_max), 0.0), 1.0)
                        at_max = float(interp_raw([[t, depth_max, lat, lon]])[0])
                        out[i] = at_max * (1.0 - frac)
                    else:
                        out[i] = 0.0
            return out
        return query

    return _wrap(interp_u), _wrap(interp_v)


def _build_fcoo_interpolators(
    model_data: xr.Dataset, t_s: np.ndarray,
    bathy_interp: Optional[Callable[[float, float], float]] = None,
    float_id: str = "",
) -> tuple[Callable, Callable]:
    """
    Build combined interp_u/interp_v for the unified 'fcoo' model, which
    normally carries two grids sharing one time coordinate (see
    data_handler._fetch_fcoo):
        idk -- finer, inner-Danish-waters only, 6 depth levels to 50 m
        dk  -- coarser, full domain, 10 depth levels to 200 m
    idk is preferred: a query point inside idk's (depth, lat, lon) bounding
    box uses idk's interpolated value; otherwise dk's. Both callables must be
    called with the same (t, z, lat, lon) row so they agree on which grid to
    use -- true for every call site today (simulate_cycle._query_uv always
    queries u and v at the same point).

    Either grid can be missing from `model_data` if _fetch_fcoo could only
    fetch one of dk/idk this round -- falls back to using whichever grid is
    present on its own rather than requiring both. A float that never
    resolves through idk anyway (outside its small inner-Danish-waters
    footprint) is then unaffected by an idk-only failure.
    """
    has_dk = "u_dk" in model_data.data_vars
    has_idk = "u_idk" in model_data.data_vars

    if has_idk:
        idk_u, idk_v, idk_bounds = _grid_interpolators(
            t_s,
            model_data["depth_idk"].values.astype(np.float64),
            model_data["lat_idk"].values.astype(np.float64),
            model_data["lon_idk"].values.astype(np.float64),
            model_data["u_idk"].values,
            model_data["v_idk"].values,
            label=f"{float_id} fcoo_idk".strip(),
        )
    if has_dk:
        dk_u, dk_v, dk_bounds = _grid_interpolators(
            t_s,
            model_data["depth_dk"].values.astype(np.float64),
            model_data["lat_dk"].values.astype(np.float64),
            model_data["lon_dk"].values.astype(np.float64),
            model_data["u_dk"].values,
            model_data["v_dk"].values,
            label=f"{float_id} fcoo_dk".strip(),
        )
        # idk itself never needs tapering: _in_idk_bounds (below) already
        # redirects any query outside idk's own 50m range to dk before it
        # would hit idk's limit -- the taper only becomes relevant once
        # dk's deeper range is also exceeded.
        dk_u, dk_v = _taper_to_seabed(
            dk_u, dk_v, dk_bounds[0][1], bathy_interp, f"{float_id} fcoo_dk".strip()
        )

    if not has_dk:
        return idk_u, idk_v
    if not has_idk:
        return dk_u, dk_v

    (z0, z1), (lat0, lat1), (lon0, lon1) = idk_bounds

    def _in_idk_bounds(z: float, lat: float, lon: float) -> bool:
        return z0 <= z <= z1 and lat0 <= lat <= lat1 and lon0 <= lon <= lon1

    def combined_u(rows):
        out = np.empty(len(rows), dtype=np.float64)
        for i, (t, z, lat, lon) in enumerate(rows):
            src = idk_u if _in_idk_bounds(z, lat, lon) else dk_u
            out[i] = src([[t, z, lat, lon]])[0]
        return out

    def combined_v(rows):
        out = np.empty(len(rows), dtype=np.float64)
        for i, (t, z, lat, lon) in enumerate(rows):
            src = idk_v if _in_idk_bounds(z, lat, lon) else dk_v
            out[i] = src([[t, z, lat, lon]])[0]
        return out

    return combined_u, combined_v


def build_interpolators(
    model_data: xr.Dataset,
    bathy_interp: Optional[Callable[[float, float], float]] = None,
    float_id: str = "",
) -> tuple[Callable, Callable]:
    """
    Build interp_u / interp_v callables from `model_data`.

    For the single-grid schema (CMEMS):
        dims : time, depth, lat, lon
        vars : u, v  (m/s, eastward / northward)
    For the merged FCOO schema (dk + idk, see data_handler._fetch_fcoo):
        dims : time, depth_dk, lat_dk, lon_dk, depth_idk, lat_idk, lon_idk
        vars : u_dk, v_dk, u_idk, v_idk -- either grid's vars/dims may be
        absent if _fetch_fcoo could only fetch one of them this round
    detected via presence of "u_dk" or "u_idk" -- resolved into a single
    combined interpolator pair (idk preferred, dk fallback, or whichever
    grid is actually present; see _build_fcoo_interpolators).

    Each callable takes a 2-D array of [t_s, depth_m, lat, lon] rows
    and returns one float per row. Out-of-bounds *lat/lon* queries return
    0.0 (open-ocean boundary -- callers treat NaN/missing as no current).
    Out-of-range *depth* queries deeper than the grid's own real levels are
    tapered toward the local seabed instead (see _taper_to_seabed) if
    `bathy_interp` is given, otherwise also return 0.0. `float_id` is only
    used to label that taper's log warning.
    """
    t_s = model_data["time"].values.astype("datetime64[s]").astype(np.float64)

    if "u_dk" in model_data.data_vars or "u_idk" in model_data.data_vars:
        return _build_fcoo_interpolators(model_data, t_s, bathy_interp, float_id)

    depth = model_data["depth"].values.astype(np.float64)
    lat   = model_data["lat"].values.astype(np.float64)
    lon   = model_data["lon"].values.astype(np.float64)
    u_arr = model_data["u"].values
    v_arr = model_data["v"].values

    interp_u, interp_v, bounds = _grid_interpolators(
        t_s, depth, lat, lon, u_arr, v_arr, label=f"{float_id} cmems".strip()
    )
    interp_u, interp_v = _taper_to_seabed(
        interp_u, interp_v, bounds[0][1], bathy_interp, f"{float_id} cmems".strip()
    )
    return interp_u, interp_v


def lookup_position(
    trajectory: list[tuple[datetime, float, float, float]],
    t: datetime,
) -> tuple[float, float] | None:
    """
    (lat, lon) at time `t`, nearest point from `trajectory`. Depth (index 3
    of each point) is ignored here -- this is a horizontal-position lookup.
    Returns None if `t` falls outside the trajectory's covered range.
    """
    if not trajectory:
        return None
    t0, t1 = trajectory[0][0], trajectory[-1][0]
    if t < t0 or t > t1:
        return None
    best = min(range(len(trajectory)), key=lambda i: abs((trajectory[i][0] - t).total_seconds()))
    return trajectory[best][1], trajectory[best][2]


# How long past our own estimate of a cycle's communicating-window start we
# tolerate before concluding that window is over and jumping a full cycle
# ahead (see next_surfacing's docstring for the bug this fixes). Not a
# business policy threshold like OVERDUE_DAYS/DEAD_THRESHOLD (those decide
# which floats to track/score); this is a numerical-robustness parameter of
# the phase-recovery formula itself, so it defaults here rather than living
# in run.py -- run.py's own call site still overrides it explicitly via
# config (NEXT_SURFACING_GRACE_HOURS), same pattern as
# FloatRow.is_overdue's threshold_days default vs. run.py's OVERDUE_DAYS.
DEFAULT_NEXT_SURFACING_GRACE_HOURS = 12.0


def next_surfacing(
    anchor_time: datetime,
    control_action: ControlAction,
    now: datetime,
    grace_hours: float = DEFAULT_NEXT_SURFACING_GRACE_HOURS,
) -> datetime:
    """
    Next time after `now` that the float re-enters the communicating (surfaced)
    phase of its repeating dive cycle.

    Reconstructs total_cycle_s exactly as simulate_cycle does, then locates
    the most recent cycle boundary
        boundary(k) = anchor_time + k * total_cycle_s + (descent_plus_parking_s + ascent_s)
    at or before `now`.

    BUG THIS FIXES: real per-cycle duration jitters by a few hours around the
    median-voted cycle_hours estimate (design decision 9) -- a real float
    might run 108h one cycle and 120h the next around a 114h median. The old
    code found the smallest k with boundary(k) > now, full stop. The instant
    `now` ticked past *our own estimate* of boundary(k) -- even by a minute,
    even though the real float hadn't actually surfaced yet -- it concluded
    that whole communicating window must already be over and jumped straight
    to boundary(k+1), a FULL CYCLE later (visible in section 3 of
    prediction_diagnostics.ipynb as `timing_error_h` spikes of ~one cycle
    length: ~100h+ for a ~114h-cycle float, ~20-50h for a ~49h-cycle float).
    In plain terms: the model would go from "surfacing any minute now" to
    "not for another 4-5 days" over the course of a single missed minute,
    just because the float was running a little late.

    THE FIX: once `now` is within `grace_hours` of boundary(k), keep
    predicting boundary(k) itself (clamped to just after `now` if that
    boundary is already in the past) instead of committing to boundary(k+1)
    -- i.e. "any time now" rather than "not for another full cycle." Only
    once the overrun exceeds `grace_hours` do we conclude the window is
    genuinely over and advance to boundary(k+1). This stays fully
    deterministic (no probability distributions, no covariance -- design
    decision 10 still holds); it only changes the threshold for how long we
    keep betting on the current cycle before giving up on it.
    """
    target_depth = control_action.target_depth or 0.0
    descent_s             = target_depth / control_action.descent_speed_ms if target_depth > 0 else 0.0
    ascent_s              = target_depth / control_action.ascent_speed_ms  if target_depth > 0 else 0.0
    transmission_s        = control_action.transmission_duration_minutes * 60.0
    descent_plus_parking_s = control_action.cycle_hours * 3600.0
    total_cycle_s         = descent_plus_parking_s + ascent_s + transmission_s
    surface_offset        = descent_plus_parking_s + ascent_s   # communicating starts here

    elapsed = (now - anchor_time).total_seconds()
    grace_s = grace_hours * 3600.0

    # Largest k >= 0 with boundary(k) <= elapsed (i.e. the most recently
    # reached -- or still-pending, if elapsed < surface_offset -- predicted
    # window-open time).
    k = max(0, math.floor((elapsed - surface_offset) / total_cycle_s))
    boundary_s = surface_offset + k * total_cycle_s
    overrun_s = elapsed - boundary_s  # negative if boundary(k) is still in the future

    if overrun_s <= grace_s:
        # Either haven't reached this cycle's window yet, or recently passed
        # our own estimate of it -- keep betting on this cycle.
        t_surface = anchor_time + timedelta(seconds=boundary_s)
    else:
        # Genuinely past the grace period -- this window is over.
        t_surface = anchor_time + timedelta(seconds=boundary_s + total_cycle_s)

    # Contract is "next time AFTER now" -- both branches above can land at or
    # before `now` (the grace branch deliberately does, to report "any time
    # now" rather than a stale exact instant). Clamp forward by an epsilon.
    if t_surface <= now:
        t_surface = now + timedelta(seconds=1)
    return t_surface


def _query_uv(x, y, z, t, interp_u, interp_v, anchor_lat, anchor_lon) -> tuple[float, float]:
    lat, lon = xy_to_latlon(x, y, anchor_lat, anchor_lon)
    t_s = np.datetime64(t, "s").astype(np.float64)
    u = float(interp_u([[t_s, z, lat, lon]])[0])
    v = float(interp_v([[t_s, z, lat, lon]])[0])
    if math.isnan(u):
        u = 0.0
    if math.isnan(v):
        v = 0.0
    return u, v


def simulate_cycle(
    model_data: xr.Dataset,
    control_action: ControlAction,
    anchor_lat: float,
    anchor_lon: float,
    anchor_time: datetime,
    tip_lat: float,
    tip_lon: float,
    tip_time: datetime,
    until_time: datetime,
    dt: float = 3600.0,
    bathy_interp: Optional[Callable[[float, float], float]] = None,
    float_id: str = "",
    force_drift: bool = False,
) -> list[tuple[datetime, float, float, float]]:
    """
    Extend a trajectory forward from `tip_time` to `until_time`, using
    `model_data`'s currents and `control_action`'s dive profile.

    bathy_interp: optional (lat, lon) -> seabed depth callable, forwarded to
        build_interpolators so queries deeper than a grid's real depth range
        taper toward the seabed instead of hard zero-filling (see
        _taper_to_seabed). float_id is only used to label that taper's log
        warning. Both default to the old zero-fill behavior if omitted.

    force_drift: if True, disables the park_on_bottom skip below -- the
        float gets advected by the parking-depth current for the entire
        cycle instead of sitting motionless while "on the bottom". Used to
        run a shadow track alongside the normal one (see run.py's
        SHADOW_MODELS) so the two park-phase assumptions can be compared
        against real surfacings without either affecting the public
        leaderboard/map.

    anchor_lat/anchor_lon/anchor_time: the float's last confirmed real
        surfacing. Defines (x=0, y=0) and cycle-phase zero for every repeat
        of the dive profile until the next real ping resets it (run.py's
        reconciliation step does that reset, not this function).
    tip_lat/tip_lon/tip_time: the trajectory's current last point -- this
        call resumes from here, NOT from the anchor. (x, y) at the start of
        this call are reconstructed from tip_lat/tip_lon via latlon_to_xy,
        so resuming correctly does not depend on tip == anchor.
    until_time: stop extending once simulated time reaches this. Should be
        model_data's own last available timestamp -- run.py is responsible
        for not asking this function to extrapolate past what model_data
        actually covers.
    model_data: expected to already be trimmed to forecast-only timestamps
        (data_handler.trim_to_forecast_only) before it reaches here -- this
        function doesn't re-check that.

    Returns points strictly after tip_time as (t, lat, lon, depth) tuples --
    depth (dbar) is this same phase-recovery formula's own descent/park/
    ascent number, already computed per-point to decide advection (the
    park_on_bottom skip below), just also returned now instead of discarded.
    Caller appends these to the existing trajectory; the tip itself is not
    repeated in the output.
    """
    target_depth = control_action.target_depth or 0.0
    descent_s = target_depth / control_action.descent_speed_ms if target_depth > 0 else 0.0
    ascent_s = target_depth / control_action.ascent_speed_ms if target_depth > 0 else 0.0
    transmission_s = control_action.transmission_duration_minutes * 60.0

    # cycle_hours is descent+parking only (see ControlAction docstring) --
    # reconstruct the full repeat period rather than treating cycle_hours
    # itself as the total.
    descent_plus_parking_s = control_action.cycle_hours * 3600.0
    parking_s = max(descent_plus_parking_s - descent_s, 0.0)
    total_cycle_s = descent_plus_parking_s + ascent_s + transmission_s

    interp_u, interp_v = build_interpolators(model_data, bathy_interp=bathy_interp, float_id=float_id)

    x, y = latlon_to_xy(tip_lat, tip_lon, anchor_lat, anchor_lon)
    elapsed = (tip_time - anchor_time).total_seconds()
    t = tip_time

    points: list[tuple[datetime, float, float, float]] = []

    while t < until_time:
        cycle_elapsed = elapsed % total_cycle_s

        if cycle_elapsed < descent_s:
            depth = control_action.descent_speed_ms * cycle_elapsed
        elif cycle_elapsed < descent_s + parking_s:
            depth = target_depth
        elif cycle_elapsed < descent_s + parking_s + ascent_s:
            into_ascent = cycle_elapsed - (descent_s + parking_s)
            depth = max(target_depth - control_action.ascent_speed_ms * into_ascent, 0.0)
        else:
            depth = 0.0

        parked_on_bottom = (
            not force_drift
            and control_action.park_mode == "park_on_bottom"
            and descent_s <= cycle_elapsed < descent_s + parking_s
        )

        if not parked_on_bottom:
            u, v = _query_uv(x, y, depth, t, interp_u, interp_v, anchor_lat, anchor_lon)
            x += u * dt
            y += v * dt

        t += timedelta(seconds=dt)
        elapsed += dt
        lat, lon = xy_to_latlon(x, y, anchor_lat, anchor_lon)
        points.append((t, lat, lon, depth))

    return points
