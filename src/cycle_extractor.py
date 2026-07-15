"""
cycle_extractor.py
====================
Derive a representative profiling-cycle ControlAction for a float from its
historical .traj (Rtraj.nc) + profiles.nc files.

Adapted from your argo_data_analyser cycle-extraction notebook cell, with
one fix and one addition:

    - Fix: the profiles-file suffix search tried ('_loologlal', '_profiles.nc').
      '_loologlal' isn't a real file suffix and can never match, so GDAC-format
      floats (which use '_prof.nc', per _build_profile_pos's own docstring)
      were never actually checked -- they silently fell through to the Rtraj
      fallback every time. Fixed to ('_profiles.nc', '_prof.nc'). Flag if that
      wasn't what you intended.

    - Addition: build_actions() / mode_vote_action() / default_action(), which
      is the piece add_new_float() actually needs -- not a single cycle's
      ControlAction, but one representative action mode-voted across a
      float's full history, with the 5-day-bottom-park fallback for floats
      with no usable history at all.

    - Fix: each cycle's "last_lat"/"last_lon" (its own surfacing position --
      what run.py uses to seed surfacing_history/last_real_position, the
      ground truth every model is scored against) used to be a raw scan for
      the last non-NaN LATITUDE/LONGITUDE anywhere in that cycle's Rtraj
      block. That's exactly the raw-.traj GPS fix your notebook flagged as
      occasionally erroneous (a pre-GPS-lock fix). Now it reuses the same
      QC'd profiles-file position (profile_pos, with _robust_start_pos
      fallback) already computed as start_lat/start_lon for that cycle.

This module is pure computation -- no fetching, no store I/O. run.py calls
data_handler.download_float_history() to get rtraj_path, then this module
to turn it into a ControlAction.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import xarray as xr

from .simulate import ControlAction

DESC_CODE = 190
PARKING_CODE = 290
ASC_CODE = 590

DRIFT_THRESH_KM = -1.0      # disabled: drift-based parked_on_bottom classification not used (depth-based only)
DEPTH_FRAC_THRESH = 0.85    # avg_parking_depth / bathy >= this -> parked_on_bottom

# Fallback for a brand-new float with no usable cycle history.
DEFAULT_CYCLE_HOURS = 120.0   # 5 days, descent+parking only (see ControlAction docstring)
DEFAULT_TRANSMISSION_MINUTES = 30.0
DEFAULT_DESCENT_SPEED_MS = 0.08
DEFAULT_ASCENT_SPEED_MS = 0.08


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def _speed_from_pres_juld(sub_ds) -> Optional[float]:
    pres = sub_ds.PRES.values.astype(float)
    juld = sub_ds.JULD.values
    dt_s = np.diff(juld) / np.timedelta64(1, "s")
    valid = dt_s != 0
    if not valid.any():
        return None
    return float(np.nanmean(np.diff(pres)[valid] / dt_s[valid]))


def _build_profile_pos(profiles_path: Path) -> dict[int, tuple[float, float]]:
    """Build cycle -> (lat, lon) from ascending profiles, preferring QC=1.
    Handles argopy format (_profiles.nc, unicode + int64) and
    GDAC format (_prof.nc, bytes/object)."""
    ds_p = xr.open_dataset(profiles_path).load()
    direction = ds_p["DIRECTION"].values
    if direction.dtype == object:  # GDAC bytes: b'A'
        asc_mask = np.array([d == b"A" for d in direction])
    else:  # argopy unicode: 'A'
        asc_mask = direction == "A"
    prof_cyc = ds_p["CYCLE_NUMBER"].values[asc_mask].astype(int)
    prof_lat = ds_p["LATITUDE"].values[asc_mask].astype(float)
    prof_lon = ds_p["LONGITUDE"].values[asc_mask].astype(float)
    raw_qc = ds_p["POSITION_QC"].values[asc_mask]
    try:
        prof_qc = raw_qc.astype(int)
    except (ValueError, TypeError):
        prof_qc = np.array([
            int(q.decode().strip()) if isinstance(q, bytes) else int(str(q).strip())
            for q in raw_qc
        ])
    pos: dict[int, tuple[float, float]] = {}
    for c in np.unique(prof_cyc):
        idx = np.where(prof_cyc == c)[0]
        good = idx[prof_qc[idx] == 1]
        use = good if len(good) else idx
        pos[int(c)] = (float(prof_lat[use[0]]), float(prof_lon[use[0]]))
    return pos


def _robust_start_pos(cycle_ds) -> tuple[Optional[float], Optional[float]]:
    """Fallback position using code-703 GPS fixes when no profiles file has an entry."""
    mask_703 = cycle_ds.MEASUREMENT_CODE.values == 703
    lats = cycle_ds.LATITUDE.values.astype(float)[mask_703]
    lons = cycle_ds.LONGITUDE.values.astype(float)[mask_703]
    j = cycle_ds.JULD.values.astype(float)[mask_703]
    valid = ~np.isnan(lats) & ~np.isnan(lons) & ~np.isnan(j)
    if not valid.any():
        all_lat = cycle_ds.LATITUDE.values.astype(float)
        all_lon = cycle_ds.LONGITUDE.values.astype(float)
        v = ~np.isnan(all_lat) & ~np.isnan(all_lon)
        return (float(np.median(all_lat[v])), float(np.median(all_lon[v]))) if v.any() else (None, None)
    lats, lons, j = lats[valid], lons[valid], j[valid]
    order = np.argsort(j)
    lats, lons, j = lats[order], lons[order], j[order]
    if len(lats) == 1:
        return float(lats[0]), float(lons[0])
    t_sec = (j - j[0]) * 86400
    within_30 = t_sec[1:] <= 30
    close = (np.abs(lons[1:] - lons[0]) <= 0.3) & (np.abs(lats[1:] - lats[0]) <= 0.3)
    confirmed = (within_30 & close).any()
    if confirmed:
        early = (t_sec <= 60) & (np.abs(lons - lons[0]) <= 0.5) & (np.abs(lats - lats[0]) <= 0.5)
        return float(np.mean(lats[early])), float(np.mean(lons[early]))
    else:
        return float(np.median(lats[1:])), float(np.median(lons[1:]))


def extract_cycles(rtraj_path: Path, bathy_interp: Callable) -> list[dict]:
    """Derive per-cycle metadata from an Argo Rtraj NetCDF file. Logic
    unchanged from your original except the profiles-suffix fix above."""
    rtraj_path = Path(rtraj_path)
    ds = xr.open_dataset(rtraj_path)

    wmo_stem = rtraj_path.name.replace("_Rtraj.nc", "")
    for suffix in ("_profiles.nc", "_prof.nc"):
        candidate = rtraj_path.parent / (wmo_stem + suffix)
        if candidate.exists():
            profile_pos = _build_profile_pos(candidate)
            break
    else:
        profile_pos = {}

    all_cycles = [int(c) for c in np.unique(ds.CYCLE_NUMBER.values) if not np.isnan(float(c))]
    cycle_numbers = sorted(all_cycles)

    meta: dict[int, dict] = {}
    for cyc in cycle_numbers:
        mask = ds.CYCLE_NUMBER.values == cyc
        cyc_ds = ds.isel(N_MEASUREMENT=mask)
        juld = cyc_ds.JULD.values
        valid_t = juld[~np.isnat(juld)]
        if len(valid_t) == 0:
            continue

        codes = cyc_ds.MEASUREMENT_CODE.values
        desc_ds = cyc_ds.isel(N_MEASUREMENT=codes == DESC_CODE)
        asc_ds = cyc_ds.isel(N_MEASUREMENT=codes == ASC_CODE)
        park_ds = cyc_ds.isel(N_MEASUREMENT=codes == PARKING_CODE)

        desc_t = desc_ds.JULD.values
        desc_t = desc_t[~np.isnat(desc_t)]
        asc_t = asc_ds.JULD.values
        asc_t = asc_t[~np.isnat(asc_t)]

        if cyc in profile_pos:
            start_lat, start_lon = profile_pos[cyc]
        else:
            start_lat, start_lon = _robust_start_pos(cyc_ds)
            if start_lat is None:
                continue

        # This cycle's own surfacing position is the same QC'd profile
        # position used above as start_lat/start_lon -- not a raw scan for
        # the last non-NaN GPS fix in the cycle's Rtraj block, which can
        # pick up a bad pre-GPS-lock fix (see _robust_start_pos docstring).
        last_lat, last_lon = start_lat, start_lon

        pres = park_ds.PRES.values.astype(float)
        pres = pres[~np.isnan(pres)]

        bathy_depth_m = float(bathy_interp(start_lat, start_lon))

        depth_based_parked = False
        avg_bottom_depth_dbar = None

        if len(pres) > 0:
            avg_bottom_depth_dbar = float(np.mean(pres))
            if not math.isnan(bathy_depth_m) and bathy_depth_m > 0:
                depth_based_parked = bool(avg_bottom_depth_dbar / bathy_depth_m >= DEPTH_FRAC_THRESH)

        meta[cyc] = dict(
            cycle=cyc,
            start_lat=start_lat,
            start_lon=start_lon,
            last_lat=last_lat,
            last_lon=last_lon,
            start_time=str(valid_t[0]),
            end_time=str(valid_t[-1]),
            first_desc_juld=desc_t[0] if len(desc_t) > 0 else valid_t[0],
            last_asc_juld=asc_t[-1] if len(asc_t) > 0 else valid_t[-1],
            descent_speed_dbar_s=_speed_from_pres_juld(desc_ds) if len(desc_ds.N_MEASUREMENT) >= 2 else None,
            ascent_speed_dbar_s=_speed_from_pres_juld(asc_ds) if len(asc_ds.N_MEASUREMENT) >= 2 else None,
            avg_bottom_depth_dbar=avg_bottom_depth_dbar,
            bathy_depth_m=bathy_depth_m,
            depth_based_parked=depth_based_parked,
            parked_on_bottom=False,
        )

    for i, cyc in enumerate(cycle_numbers[:-1]):
        nxt = cycle_numbers[i + 1]
        if cyc not in meta or nxt not in meta:
            continue
        m, nm = meta[cyc], meta[nxt]
        surf_min = float((nm["first_desc_juld"] - m["last_asc_juld"]) / np.timedelta64(1, "m"))
        dlat_m = (nm["start_lat"] - m["last_lat"]) * 111320.0
        dlon_m = (nm["start_lon"] - m["last_lon"]) * 111320.0 * math.cos(math.radians(m["last_lat"]))
        drift_km = math.sqrt(dlat_m ** 2 + dlon_m ** 2) / 1000.0
        m["surface_duration_min"] = round(max(surf_min, 0.0), 1)
        m["drift_km"] = round(drift_km, 3)
        m["parked_on_bottom"] = (drift_km <= DRIFT_THRESH_KM) or m.get("depth_based_parked", False)

    if cycle_numbers and cycle_numbers[-1] in meta:
        last = meta[cycle_numbers[-1]]
        last.setdefault("surface_duration_min", 30.0)
        last.setdefault("drift_km", None)
        last.setdefault("parked_on_bottom", False)

    desc_speeds = [v["descent_speed_dbar_s"] for v in meta.values() if v["descent_speed_dbar_s"] is not None]
    asc_speeds = [v["ascent_speed_dbar_s"] for v in meta.values() if v["ascent_speed_dbar_s"] is not None]
    mean_desc = float(np.mean(desc_speeds)) if desc_speeds else DEFAULT_DESCENT_SPEED_MS
    mean_asc = float(np.mean(asc_speeds)) if asc_speeds else DEFAULT_ASCENT_SPEED_MS
    for v in meta.values():
        if v["descent_speed_dbar_s"] is None:
            v["descent_speed_dbar_s"] = mean_desc
        if v["ascent_speed_dbar_s"] is None:
            v["ascent_speed_dbar_s"] = mean_asc

    ds.close()
    return [meta[c] for c in cycle_numbers if c in meta]


def action_from_cycle(cyc: dict, next_cyc: dict) -> ControlAction:
    """Build a ControlAction from a cycle dict and the following cycle dict."""
    t_start = np.datetime64(cyc["start_time"])
    t_next = np.datetime64(next_cyc["start_time"])
    surf_min = float(cyc.get("surface_duration_min") or 30.0)

    try:
        asc_s = cyc["avg_bottom_depth_dbar"] / abs(cyc["ascent_speed_dbar_s"])
        cycle_hours = float(
            (t_next - t_start - np.timedelta64(int(surf_min), "m") - np.timedelta64(int(asc_s), "s"))
            / np.timedelta64(1, "h")
        )
    except Exception:
        # Fallback when depth or ascent speed is unavailable (e.g. drift_on_surface).
        # Subtract surface transmission time so cycle_hours stays descent+parking only,
        # matching the ControlAction invariant. Ascent time is omitted (unknown), so
        # simulate_cycle will slightly overestimate parking duration -- acceptable.
        cycle_hours = float(
            (t_next - t_start - np.timedelta64(int(surf_min), "m"))
            / np.timedelta64(1, "h")
        )

    if cyc.get("parked_on_bottom", False):
        park_mode = "park_on_bottom"
    elif cyc.get("avg_bottom_depth_dbar") is not None:
        park_mode = "parking_depth"
    else:
        park_mode = "drift_on_surface"

    return ControlAction(
        park_mode=park_mode,
        cycle_hours=max(cycle_hours, 0.5),
        transmission_duration_minutes=surf_min,
        target_depth=cyc.get("avg_bottom_depth_dbar"),
        descent_speed_ms=abs(cyc["descent_speed_dbar_s"]),
        ascent_speed_ms=abs(cyc["ascent_speed_dbar_s"]),
    )


def build_actions(cycles: list[dict]) -> list[ControlAction]:
    """One ControlAction per consecutive cycle pair -- the per-cycle history
    that mode_vote_action() collapses into a single representative action."""
    return [action_from_cycle(cycles[i], cycles[i + 1]) for i in range(len(cycles) - 1)]


def mode_vote_action(actions: list[ControlAction]) -> ControlAction:
    """
    Collapse a float's per-cycle history into one representative ControlAction.

    park_mode: mode (most common value) across history -- the agreed
    resolution for mixed histories, rather than treating mixed history as
    "no usable history."
    All numeric fields: mean across history.

    Raises if `actions` is empty -- callers (add_new_float) are responsible
    for routing empty history to default_action() instead of calling this
    with nothing to vote on.
    """
    if not actions:
        raise ValueError("mode_vote_action requires at least one ControlAction")

    park_mode = Counter(a.park_mode for a in actions).most_common(1)[0][0]
    target_depths = [a.target_depth for a in actions if a.target_depth is not None]

    return ControlAction(
        park_mode=park_mode,
        cycle_hours=float(np.mean([a.cycle_hours for a in actions])),
        transmission_duration_minutes=float(np.mean([a.transmission_duration_minutes for a in actions])),
        target_depth=float(np.mean(target_depths)) if target_depths else None,
        descent_speed_ms=float(np.mean([a.descent_speed_ms for a in actions])),
        ascent_speed_ms=float(np.mean([a.ascent_speed_ms for a in actions])),
    )


def default_action() -> ControlAction:
    """Fallback ControlAction for a float with no usable cycle history at all."""
    return ControlAction(
        park_mode="park_on_bottom",
        cycle_hours=DEFAULT_CYCLE_HOURS,
        transmission_duration_minutes=DEFAULT_TRANSMISSION_MINUTES,
        target_depth=None,
        descent_speed_ms=DEFAULT_DESCENT_SPEED_MS,
        ascent_speed_ms=DEFAULT_ASCENT_SPEED_MS,
    )
