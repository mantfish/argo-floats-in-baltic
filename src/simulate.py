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


def _fill_masked_nearest(arr: np.ndarray, masked: np.ndarray) -> np.ndarray:
    """
    Replace every masked (land/no-data) cell with its nearest real-data
    neighbor's value, per depth level, instead of zero-filling it.

    Previously a masked cell (coastline, or a depth level that's locally
    below the seabed at that lat/lon) zero-filled to exactly 0.0 -- reading
    as "current is genuinely zero here", indistinguishable from a real
    open-ocean fill_value=0.0. Since simulate_cycle's park_on_bottom floats
    only ever advect during their brief (1-2h) descent/ascent window each
    cycle, a single masked-cell hit there was enough to freeze an entire
    multi-day leg at the anchor position -- confirmed as the dominant cause
    of the leaderboard's exact-zero-error events. _warn_on_zero_fill already
    detected and logged this exact failure mode; this is the correction
    that was missing to go with it.

    The land/sea mask is treated as time-invariant -- masked[0] (first
    timestep) stands in for every timestep, same assumption
    _warn_on_zero_fill's own masked-cell diagnostic already makes -- so the
    nearest-valid-cell index map is computed once per depth level (one
    scipy.ndimage distance transform each) and reused across the whole time
    axis. Only the masked cells themselves are gathered/written (not the
    whole lat/lon grid) -- a naive `out[:, zi] = out[:, zi][:, idx[0],
    idx[1]]` re-copies every already-valid cell into itself too, which for
    a full-size grid is most of the array's bytes for no reason.
    """
    from scipy.ndimage import distance_transform_edt

    out = arr.copy()
    n_depth = arr.shape[1]
    for zi in range(n_depth):
        m = masked[0, zi]
        if not m.any() or m.all():
            continue  # nothing masked at this level, or nothing valid to fill from
        idx = distance_transform_edt(m, return_distances=False, return_indices=True)
        mi, mj = np.nonzero(m)
        si, sj = idx[0][mi, mj], idx[1][mi, mj]
        out[:, zi, mi, mj] = out[:, zi, si, sj]
    return out


def _clamp_shallow(interp_raw: Callable, depth_min: float) -> Callable:
    """
    Wrap a grid's raw (u or v) callable so a query shallower than the grid's
    shallowest real depth level clamps to that level instead of falling
    into RegularGridInterpolator's own out-of-range fill_value=0.0.

    simulate_cycle's phase-recovery formula queries depth=0.0 exactly at
    the start of every descent (cycle_elapsed=0) -- but CMEMS's shallowest
    real level is ~0.5 m and FCOO's is 5 m, so that query previously fell
    below the grid's real depth range and silently zero-filled for that
    entire step, discarding what's typically the fastest-moving layer of
    the water column. Only the shallow side is handled here -- the deep
    side (query beyond the grid's deepest level) is _taper_to_seabed's job.
    """
    def query(rows):
        rows = np.array(rows, dtype=np.float64, copy=True)
        np.maximum(rows[:, 1], depth_min, out=rows[:, 1])
        return interp_raw(rows)
    return query


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

    # Remember which cells were NaN (land/mask) before the nearest-fill
    # below, so _warn_on_zero_fill can still flag "query landed on a
    # masked cell" for visibility even though it's no longer silently
    # zero -- otherwise a persistent masked-cell query (e.g. a float
    # parked in a channel narrower than the grid resolves) would be
    # invisible in the logs despite being corrected.
    masked = np.isnan(u_arr) | np.isnan(v_arr)

    # Fill masked (land/no-data) cells from their nearest real neighbor
    # rather than zeroing them -- see _fill_masked_nearest.
    u_arr = _fill_masked_nearest(u_arr, masked)
    v_arr = _fill_masked_nearest(v_arr, masked)
    # Any cell still NaN (fully masked depth level, nothing to fill from)
    # falls back to 0 so interpolation doesn't propagate NaNs.
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
    depth_min = float(depth.min())
    interp_u = _clamp_shallow(interp_u, depth_min)
    interp_v = _clamp_shallow(interp_v, depth_min)
    interp_u, interp_v = _warn_on_zero_fill(interp_u, interp_v, t_s, depth, lat, lon, masked, label)
    bounds = ((depth_min, float(depth.max())),
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
    was NaN (land/mask) before _fill_masked_nearest corrected it in
    _grid_interpolators, logs one warning each -- the counterpart to
    _taper_to_seabed's existing depth-overrun warning, which only covers
    the deep-query case. The masked-cell case is corrected (nearest real
    neighbor, not zero) by the time it reaches here; this warning stays so
    a persistently masked query -- e.g. a float parked in a channel
    narrower than the grid resolves -- is still visible in the logs, not
    just silently patched over. A query outside the time/lat/lon footprint
    is still a genuine fill_value=0.0 (open-ocean boundary / stale forecast
    data), unaffected by the masked-cell fix.

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

            before_start = rows[:, 0] < t_min
            past_end = rows[:, 0] > t_max
            time_oob = before_start | past_end
            if np.any(time_oob) and not warned_time:
                # Two genuinely different situations were previously conflated
                # into one "Xh past the last available timestep" message,
                # always measured against t_max -- for a before_start query
                # that always produced a confusing negative number instead of
                # correctly describing a query before the window's start.
                if np.any(past_end):
                    gap_h = (float(rows[past_end, 0].max()) - t_max) / 3600.0
                    logger.warning(
                        "%s: query time %.1fh past the fetched forecast "
                        "window's last available timestep -- fill_value=0.0 "
                        "used from there on",
                        label, gap_h,
                    )
                else:
                    gap_h = (t_min - float(rows[before_start, 0].min())) / 3600.0
                    logger.warning(
                        "%s: query time %.1fh before the fetched forecast "
                        "window's first available timestep -- fill_value=0.0 "
                        "used until the window is reached (expected when "
                        "catching a trajectory up from an old anchor)",
                        label, gap_h,
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
                        "treated as nearest real cell's current (see "
                        "_fill_masked_nearest)", label,
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


def _surface_interpolators(
    t_s: np.ndarray, lat: np.ndarray, lon: np.ndarray,
    u_arr: np.ndarray, v_arr: np.ndarray, label: str = "",
) -> tuple[Callable, Callable, tuple[tuple[float, float], tuple[float, float]]]:
    """
    Build (interp_u, interp_v, bounds) for FCOO's dedicated near-surface
    grid -- (time, lat, lon), no depth axis, since it's already FCOO's own
    level-averaged representative surface value (see
    data_handler._build_fcoo_surface_dataset). bounds is ((lat_min,
    lat_max), (lon_min, lon_max)).

    Reuses _fill_masked_nearest for the same masked-cell correction as the
    depth-resolved grids (via a throwaway length-1 depth axis, since that
    helper is written for a (time, depth, lat, lon) array). The returned
    callables still accept the pipeline's standard (t, z, lat, lon) 4-column
    rows -- z is simply ignored -- so every call site can query this grid
    the same way as any depth-resolved one without a special case.
    """
    from scipy.interpolate import RegularGridInterpolator

    u_arr = u_arr.astype(np.float32)
    v_arr = v_arr.astype(np.float32)
    masked = np.isnan(u_arr) | np.isnan(v_arr)

    u_arr = _fill_masked_nearest(u_arr[:, None], masked[:, None])[:, 0]
    v_arr = _fill_masked_nearest(v_arr[:, None], masked[:, None])[:, 0]
    u_arr = np.where(np.isnan(u_arr), np.float32(0.0), u_arr)
    v_arr = np.where(np.isnan(v_arr), np.float32(0.0), v_arr)

    interp_u = RegularGridInterpolator(
        (t_s, lat, lon), u_arr, method="linear", bounds_error=False, fill_value=0.0,
    )
    interp_v = RegularGridInterpolator(
        (t_s, lat, lon), v_arr, method="linear", bounds_error=False, fill_value=0.0,
    )

    def _drop_z(interp_raw: Callable) -> Callable:
        def query(rows):
            rows = np.asarray(rows, dtype=np.float64)
            return interp_raw(rows[:, [0, 2, 3]])
        return query

    bounds = ((float(lat.min()), float(lat.max())), (float(lon.min()), float(lon.max())))
    return _drop_z(interp_u), _drop_z(interp_v), bounds


# Below this depth, prefer FCOO's dedicated near-surface grid over dk/idk --
# matches dk/idk's own shallowest real level (both start at 5m), so this is
# exactly the gap _clamp_shallow used to paper over by holding the current
# constant at dk/idk's 5m value. Real near-surface data there instead.
_FCOO_SURFACE_DEPTH_MAX_M = 5.0


def _build_fcoo_interpolators(
    model_data: xr.Dataset, t_s: np.ndarray,
    bathy_interp: Optional[Callable[[float, float], float]] = None,
    float_id: str = "",
) -> tuple[Callable, Callable]:
    """
    Build combined interp_u/interp_v for the unified 'fcoo' model, which
    normally carries three grids (see data_handler._fetch_fcoo):
        surf -- dedicated near-surface product, no depth axis, own time coord
        idk  -- finer, inner-Danish-waters only, 6 depth levels to 50 m
        dk   -- coarser, full domain, 10 depth levels to 200 m
    Resolution order per query point: surf if shallower than
    _FCOO_SURFACE_DEPTH_MAX_M and inside surf's own footprint; else idk if
    inside its (depth, lat, lon) bounding box; else dk. All callables are
    called with the same (t, z, lat, lon) row so they agree on which grid to
    use -- true for every call site today (simulate_cycle._query_uv always
    queries u and v at the same point).

    Any of the three can be missing from `model_data` if _fetch_fcoo could
    only fetch some of them this round -- falls back to whichever combination
    is present rather than requiring all three. A float that never resolves
    through idk or surf anyway (outside their footprints) is unaffected by
    either one being absent.
    """
    has_dk = "u_dk" in model_data.data_vars
    has_idk = "u_idk" in model_data.data_vars
    has_surf = "u_surf" in model_data.data_vars

    idk_u = idk_v = dk_u = dk_v = surf_u = surf_v = None
    idk_bounds = dk_bounds = surf_bounds = None

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
    if has_surf:
        t_s_surf = model_data["time_surf"].values.astype("datetime64[s]").astype(np.float64)
        surf_u, surf_v, surf_bounds = _surface_interpolators(
            t_s_surf,
            model_data["lat_surf"].values.astype(np.float64),
            model_data["lon_surf"].values.astype(np.float64),
            model_data["u_surf"].values,
            model_data["v_surf"].values,
            label=f"{float_id} fcoo_surf".strip(),
        )

    def _in_idk_bounds(z: float, lat: float, lon: float) -> bool:
        if idk_bounds is None:
            return False
        (z0, z1), (lat0, lat1), (lon0, lon1) = idk_bounds
        return z0 <= z <= z1 and lat0 <= lat <= lat1 and lon0 <= lon <= lon1

    def _in_surf_bounds(lat: float, lon: float) -> bool:
        if surf_bounds is None:
            return False
        (lat0, lat1), (lon0, lon1) = surf_bounds
        return lat0 <= lat <= lat1 and lon0 <= lon <= lon1

    def _resolve(z: float, lat: float, lon: float) -> tuple[Callable, Callable]:
        if has_surf and z < _FCOO_SURFACE_DEPTH_MAX_M and _in_surf_bounds(lat, lon):
            return surf_u, surf_v
        if has_idk and _in_idk_bounds(z, lat, lon):
            return idk_u, idk_v
        if has_dk:
            return dk_u, dk_v
        # dk absent this round and the query fell outside idk/surf coverage
        # -- idk is all that's left; RGI's own fill_value=0.0 covers the
        # out-of-range portion.
        return idk_u, idk_v

    def combined_u(rows):
        out = np.empty(len(rows), dtype=np.float64)
        for i, (t, z, lat, lon) in enumerate(rows):
            src_u, _ = _resolve(z, lat, lon)
            out[i] = src_u([[t, z, lat, lon]])[0]
        return out

    def combined_v(rows):
        out = np.empty(len(rows), dtype=np.float64)
        for i, (t, z, lat, lon) in enumerate(rows):
            _, src_v = _resolve(z, lat, lon)
            out[i] = src_v([[t, z, lat, lon]])[0]
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
    For the merged FCOO schema (dk + idk + surf, see data_handler._fetch_fcoo):
        dims : time, depth_dk, lat_dk, lon_dk, depth_idk, lat_idk, lon_idk,
               time_surf, lat_surf, lon_surf
        vars : u_dk, v_dk, u_idk, v_idk, u_surf, v_surf -- any grid's vars/
        dims may be absent if _fetch_fcoo could only fetch some of them
        this round
    detected via presence of "u_dk" or "u_idk" -- resolved into a single
    combined interpolator pair (surf preferred below dk/idk's shallowest
    real level, then idk, then dk, or whichever subset is actually present;
    see _build_fcoo_interpolators).

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
    dt_fine: float = 60.0,
    dt_parked: float = 600.0,
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
    ascent number, evaluated at that point's own timestamp (the end of the
    RK4 step that produced it, not the step's start -- see the RK4 note
    below). Caller appends these to the existing trajectory; the tip itself
    is not repeated in the output. Steps land exactly on phase boundaries
    (see dt_parked below), so every returned point's timestamp can fall on
    an irregular grid -- callers must not assume a fixed spacing.

    Integration: classic 4th-order Runge-Kutta on dx/dt=u(x,y,t),
    dy/dt=v(x,y,t), not forward Euler. Each step evaluates the slope
    function _velocity() four times (at t, t+dt/2 twice, and t+dt), each
    evaluation independently re-deriving depth and the park_on_bottom skip
    from its own cycle_elapsed -- so a step that straddles a phase boundary
    (e.g. descent -> parking) blends the advecting and frozen sub-evaluations
    correctly instead of committing the whole step to whichever phase was
    active at its start.

    Step size is adaptive, not fixed: dt_fine while descending, ascending,
    or communicating at the surface -- where depth (and often current) is
    changing fast enough that coarse steps would blur real structure --
    and the much larger dt_parked while at constant parking depth, where
    depth isn't changing and the current doesn't need fine time resolution
    to track. This applies regardless of park_mode -- a "parking_depth"
    float that's still advecting the whole time benefits from dt_parked
    just as much as a frozen "park_on_bottom" one, since the criterion is
    "is depth constant right now", not "is the float moving". Each parked
    step is still clamped to never overshoot the upcoming phase boundary
    (_next_phase_boundary_s), so a large dt_parked never blurs across into
    the fine-grained descent/ascent/communicating regime next to it.
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

    # cycle_elapsed values at which the phase changes -- descent -> parking
    # -> ascent -> communicating -> (wrap) descent. Used both by _phase()
    # and to size/clamp parked steps so they land exactly on these.
    _boundaries = (descent_s, descent_s + parking_s, descent_s + parking_s + ascent_s, total_cycle_s)

    interp_u, interp_v = build_interpolators(model_data, bathy_interp=bathy_interp, float_id=float_id)

    x, y = latlon_to_xy(tip_lat, tip_lon, anchor_lat, anchor_lon)
    elapsed = (tip_time - anchor_time).total_seconds()
    t = tip_time

    def _phase(cycle_elapsed: float) -> tuple[float, bool]:
        """(depth, parked_on_bottom) at a given point in the cycle."""
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
        return depth, parked_on_bottom

    def _velocity(x_: float, y_: float, elapsed_: float, t_: datetime) -> tuple[float, float, float]:
        """RK4 slope function: (dx/dt, dy/dt, depth) at (x_, y_, t_).
        Zero velocity while parked_on_bottom (skip advection), otherwise
        the model's current at this point's own phase-recovered depth."""
        depth, parked_on_bottom = _phase(elapsed_ % total_cycle_s)
        if parked_on_bottom:
            return 0.0, 0.0, depth
        u, v = _query_uv(x_, y_, depth, t_, interp_u, interp_v, anchor_lat, anchor_lon)
        return u, v, depth

    def _step_size(cycle_elapsed: float) -> float:
        """
        dt for the step starting at this point in the cycle -- dt_fine
        during descent/ascent/communicating, dt_parked (clamped to not
        overshoot the next phase boundary) while at constant parking depth.
        """
        is_parking_window = descent_s <= cycle_elapsed < descent_s + parking_s
        if not is_parking_window:
            return dt_fine
        next_boundary = next(b for b in _boundaries if b > cycle_elapsed)
        return min(dt_parked, next_boundary - cycle_elapsed)

    points: list[tuple[datetime, float, float, float]] = []

    while t < until_time:
        step = _step_size(elapsed % total_cycle_s)
        step = min(step, (until_time - t).total_seconds())

        t_mid = t + timedelta(seconds=step / 2.0)
        t_end = t + timedelta(seconds=step)

        k1u, k1v, _ = _velocity(x, y, elapsed, t)
        k2u, k2v, _ = _velocity(x + (step / 2.0) * k1u, y + (step / 2.0) * k1v, elapsed + step / 2.0, t_mid)
        k3u, k3v, _ = _velocity(x + (step / 2.0) * k2u, y + (step / 2.0) * k2v, elapsed + step / 2.0, t_mid)
        k4u, k4v, depth_end = _velocity(x + step * k3u, y + step * k3v, elapsed + step, t_end)

        x += (step / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        y += (step / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

        t = t_end
        elapsed += step
        lat, lon = xy_to_latlon(x, y, anchor_lat, anchor_lon)
        points.append((t, lat, lon, depth_end))

    return points
