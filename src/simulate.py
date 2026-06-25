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

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np
import xarray as xr

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


def build_interpolators(model_data: xr.Dataset) -> tuple[Callable, Callable]:
    """
    Build interp_u / interp_v callables from `model_data`.

    Expects the normalised schema that data_handler produces:
        dims : time, depth, lat, lon
        vars : u, v  (m/s, eastward / northward)

    Each callable takes a 2-D array of [t_s, depth_m, lat, lon] rows
    and returns one float per row.  Out-of-bounds queries return 0.0
    (open-ocean boundary -- callers treat NaN/missing as no current).
    """
    from scipy.interpolate import RegularGridInterpolator

    # Pull 1-D coordinate arrays and convert time to float64 seconds since epoch
    t_s   = model_data["time"].values.astype("datetime64[s]").astype(np.float64)
    depth = model_data["depth"].values.astype(np.float64)
    lat   = model_data["lat"].values.astype(np.float64)
    lon   = model_data["lon"].values.astype(np.float64)

    u_arr = model_data["u"].values.astype(np.float64)  # (time, depth, lat, lon)
    v_arr = model_data["v"].values.astype(np.float64)

    # Replace NaN (land/mask) with 0 so interpolation doesn't propagate NaNs
    u_arr = np.where(np.isnan(u_arr), 0.0, u_arr)
    v_arr = np.where(np.isnan(v_arr), 0.0, v_arr)

    # Ensure coordinate arrays are strictly monotone (required by RGI)
    if depth[0] > depth[-1]:
        depth = depth[::-1]
        u_arr = u_arr[:, ::-1, :, :]
        v_arr = v_arr[:, ::-1, :, :]

    interp_u = RegularGridInterpolator(
        (t_s, depth, lat, lon), u_arr,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    interp_v = RegularGridInterpolator(
        (t_s, depth, lat, lon), v_arr,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    return interp_u, interp_v


def lookup_position(
    trajectory: list[tuple[datetime, float, float]],
    t: datetime,
) -> tuple[float, float] | None:
    """
    (lat, lon) at time `t`, nearest point from `trajectory`.
    Returns None if `t` falls outside the trajectory's covered range.
    """
    if not trajectory:
        return None
    t0, t1 = trajectory[0][0], trajectory[-1][0]
    if t < t0 or t > t1:
        return None
    best = min(range(len(trajectory)), key=lambda i: abs((trajectory[i][0] - t).total_seconds()))
    return trajectory[best][1], trajectory[best][2]


def next_surfacing(
    anchor_time: datetime,
    control_action: ControlAction,
    now: datetime,
) -> datetime:
    """
    Next time after `now` that the float re-enters the communicating (surfaced)
    phase of its repeating dive cycle.

    Reconstructs total_cycle_s exactly as simulate_cycle does, then finds the
    smallest k >= 0 such that
        anchor_time + k * total_cycle_s + (descent_plus_parking_s + ascent_s) > now
    """
    target_depth = control_action.target_depth or 0.0
    descent_s             = target_depth / control_action.descent_speed_ms if target_depth > 0 else 0.0
    ascent_s              = target_depth / control_action.ascent_speed_ms  if target_depth > 0 else 0.0
    transmission_s        = control_action.transmission_duration_minutes * 60.0
    descent_plus_parking_s = control_action.cycle_hours * 3600.0
    total_cycle_s         = descent_plus_parking_s + ascent_s + transmission_s
    surface_offset        = descent_plus_parking_s + ascent_s   # communicating starts here

    elapsed = (now - anchor_time).total_seconds()
    # Smallest k >= 0 such that k * total_cycle_s + surface_offset > elapsed
    k = max(0, math.ceil((elapsed - surface_offset + 1e-6) / total_cycle_s))
    t_surface = anchor_time + timedelta(seconds=k * total_cycle_s + surface_offset)
    # Guard against floating-point edge cases
    if t_surface <= now:
        t_surface = anchor_time + timedelta(seconds=(k + 1) * total_cycle_s + surface_offset)
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
) -> list[tuple[datetime, float, float]]:
    """
    Extend a trajectory forward from `tip_time` to `until_time`, using
    `model_data`'s currents and `control_action`'s dive profile.

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

    Returns points strictly after tip_time as (t, lat, lon) tuples. Caller
    appends these to the existing trajectory; the tip itself is not
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

    interp_u, interp_v = build_interpolators(model_data)

    x, y = latlon_to_xy(tip_lat, tip_lon, anchor_lat, anchor_lon)
    elapsed = (tip_time - anchor_time).total_seconds()
    t = tip_time

    points: list[tuple[datetime, float, float]] = []

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
            control_action.park_mode == "park_on_bottom"
            and descent_s <= cycle_elapsed < descent_s + parking_s
        )

        if not parked_on_bottom:
            u, v = _query_uv(x, y, depth, t, interp_u, interp_v, anchor_lat, anchor_lon)
            x += u * dt
            y += v * dt

        t += timedelta(seconds=dt)
        elapsed += dt
        lat, lon = xy_to_latlon(x, y, anchor_lat, anchor_lon)
        points.append((t, lat, lon))

    return points
