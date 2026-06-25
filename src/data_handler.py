"""
data_handler.py
================
All external data acquisition for the CMEMS/FCOO leaderboard.

FCOO data comes from a single nested file that holds two grids:
    fcoo_dk  -- coarser DK-wide grid  (uu_dk / vv_dk)
    fcoo_idk -- 600 m inner-Danish grid (uu_idk / vv_idk)
Both are extracted from dk_nested.velocities.Z3D_<YYYYMMDDHH>.nc.

All returned datasets follow a standard schema so the rest of the
pipeline never needs to know which model it's working with:
    dims : time, depth, lat, lon
    vars : u, v  (m/s, eastward / northward)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import xarray as xr

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Region:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


MODELS = ("cmems", "fcoo_dk", "fcoo_idk")

CMEMS_DATASET_ID = "cmems_mod_bal_phy_anfc_PT1H-i_202411"
CMEMS_DEPTH_MAX  = 200.0   # metres -- floats don't go deeper in the Baltic

# FCOO run times are 00 and 12 UTC; file appears ~4 h after run time.
FCOO_3D_URL_TEMPLATE = (
    "https://data.fcoo.dk/webmap/v2/data/FCOO/GETM/"
    "dk_nested.velocities.Z3D_{dt}.nc"
)
FCOO_KNOTS_TO_MS = 0.514444
FCOO_FILL        = -9999.0
FCOO_CACHE_DIR   = Path("data/fcoo_cache")

GDAC_HTTP      = "https://data-argo.ifremer.fr"
ARGO_CACHE_DIR = Path("data/argo_cache")

# Which suffix in the nested file maps to which MODELS entry
_FCOO_SUFFIX = {"fcoo_dk": "_dk", "fcoo_idk": "_idk"}


# --------------------------------------------------------------------------- #
# 1. Model current fields
# --------------------------------------------------------------------------- #

def download_model_data(
    model: str,
    region: Region,
    issue_time: Optional[datetime] = None,
) -> xr.Dataset:
    """
    Pull the latest forecast for `model` over `region`.

    Returns an xr.Dataset with:
        dims : time, depth, lat, lon
        vars : u (m/s eastward), v (m/s northward)
    Does NOT trim -- caller must call trim_to_forecast_only() first.
    """
    if model == "cmems":
        return _fetch_cmems(region, issue_time)
    elif model in _FCOO_SUFFIX:
        return _fetch_fcoo(model, region)
    else:
        raise ValueError(f"Unknown model {model!r}. Expected one of {MODELS}")


def trim_to_forecast_only(model_data: xr.Dataset, issue_time: datetime) -> xr.Dataset:
    """Drop every timestamp at-or-before issue_time (keep strict future only)."""
    issue_np = np.datetime64(issue_time.replace(tzinfo=None), "ns")
    return model_data.sel(time=model_data.time > issue_np)


# -- CMEMS -------------------------------------------------------------------

def _fetch_cmems(region: Region, issue_time: Optional[datetime]) -> xr.Dataset:
    import copernicusmarine

    kwargs: dict = dict(
        dataset_id=CMEMS_DATASET_ID,
        variables=["uo", "vo"],
        minimum_latitude=region.lat_min,
        maximum_latitude=region.lat_max,
        minimum_longitude=region.lon_min,
        maximum_longitude=region.lon_max,
        minimum_depth=0.0,
        maximum_depth=CMEMS_DEPTH_MAX,
    )
    if issue_time is not None:
        kwargs["start_datetime"] = issue_time.strftime("%Y-%m-%dT%H:%M:%S")

    ds = copernicusmarine.open_dataset(**kwargs)

    rename = {}
    for old, new in [("uo", "u"), ("vo", "v"), ("latitude", "lat"), ("longitude", "lon")]:
        if old in ds:
            rename[old] = new
    return ds.rename(rename) if rename else ds


# -- FCOO --------------------------------------------------------------------

_fcoo_file_cache: str | None = None  # path of last downloaded file -- skip re-probe


def _fetch_fcoo(model: str, region: Region) -> xr.Dataset:
    """Read the requested FCOO grid from the nested 3-D file into memory."""
    local = _get_fcoo_file()
    return _read_fcoo_grid(local, suffix=_FCOO_SUFFIX[model], region=region)


def _read_fcoo_grid(path: Path, suffix: str, region: Region) -> xr.Dataset:
    """
    Read one grid (suffix = '_dk' or '_idk') from the NetCDF3 nested file
    using netCDF4 directly (avoids xarray+dask segfault on Python 3.14).

    Subsets to region before loading into memory, then returns a plain
    in-memory xr.Dataset with the standard (time, depth, lat, lon) schema
    and u/v in m/s.
    """
    import netCDF4 as nc4

    with nc4.Dataset(str(path)) as f:
        lat   = f[f"latc{suffix}"][:].data.astype(float)
        lon   = f[f"lonc{suffix}"][:].data.astype(float)
        depth = f[f"zax{suffix}"][:].data.astype(float)

        # Decode time to numpy datetime64 from "seconds since <epoch>" string.
        # The NetCDF3 record dimension is preallocated (56 slots) but only the
        # first N are written; trailing entries repeat the epoch (raw value 0).
        # Keep only the strictly-ascending prefix.
        t_raw   = f["time"][:].data.astype(float)
        t_units = f["time"].units          # e.g. "seconds since 2026-06-23 11:00:00"
        epoch   = np.datetime64(t_units.replace("seconds since ", "").strip().replace(" ", "T"))
        diffs   = np.diff(t_raw)
        n_valid = int(np.argmax(diffs <= 0)) + 1 if (diffs <= 0).any() else len(t_raw)
        t_raw   = t_raw[:n_valid]
        times   = epoch + t_raw.astype("timedelta64[s]")

        # Region mask (boolean, 1-D)
        lat_ok = (lat >= region.lat_min) & (lat <= region.lat_max)
        lon_ok = (lon >= region.lon_min) & (lon <= region.lon_max)

        # Read subset -- index fancy on lat/lon axes (2, 3) to avoid loading full array;
        # slice to n_valid on the time axis to drop the zero-padded tail
        u_raw = f[f"uu{suffix}"][:n_valid, :, lat_ok, :][:, :, :, lon_ok]  # masked array
        v_raw = f[f"vv{suffix}"][:n_valid, :, lat_ok, :][:, :, :, lon_ok]

    # Convert masked arrays to plain float32, fill -> NaN, knots -> m/s
    u = np.where(u_raw.mask if np.ma.is_masked(u_raw) else (u_raw.data == FCOO_FILL),
                 np.nan, u_raw.data).astype(np.float32) * FCOO_KNOTS_TO_MS
    v = np.where(v_raw.mask if np.ma.is_masked(v_raw) else (v_raw.data == FCOO_FILL),
                 np.nan, v_raw.data).astype(np.float32) * FCOO_KNOTS_TO_MS

    return xr.Dataset(
        {"u": (["time", "depth", "lat", "lon"], u),
         "v": (["time", "depth", "lat", "lon"], v)},
        coords={
            "time":  times,
            "depth": depth,
            "lat":   lat[lat_ok],
            "lon":   lon[lon_ok],
        },
    )


def _get_fcoo_file(cache_dir: Path = FCOO_CACHE_DIR) -> Path:
    """Return path to the latest FCOO 3-D velocity NetCDF, downloading if needed."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    url      = _get_latest_fcoo_url()
    filename = url.split("/")[-1]
    local    = cache_dir / filename
    if not local.exists():
        tmp = local.with_suffix(".nc.tmp")
        logger.info("Downloading FCOO file: %s", url)
        try:
            r = requests.get(url, timeout=300, stream=True)
            r.raise_for_status()
            bytes_written = 0
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
                    bytes_written += len(chunk)
            tmp.rename(local)
            logger.info("Saved FCOO file: %s (%.1f MB)", local, bytes_written / 1e6)
        except Exception:
            tmp.unlink(missing_ok=True)  # don't leave a partial file
            raise
    return local


def _get_latest_fcoo_url() -> str:
    """
    Probe FCOO for the most recent available 3-D velocity file.
    Runs are published at 00 and 12 UTC, typically available ~4 h later.
    """
    now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    for days_back in range(3):
        base = now - timedelta(days=days_back)
        for run_hour in (12, 0):
            dt = base.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            if dt > now:
                continue
            url = FCOO_3D_URL_TEMPLATE.format(dt=dt.strftime("%Y%m%d%H"))
            try:
                r = requests.head(url, timeout=10)
                if r.status_code == 200:
                    logger.info("Latest FCOO run: %s", url)
                    return url
            except requests.RequestException:
                pass
    raise RuntimeError(
        "Could not find a recent FCOO 3-D velocity file at data.fcoo.dk. "
        "Check network access and that the URL template is still valid."
    )


# --------------------------------------------------------------------------- #
# 2. Argo domain pull
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ArgoPull:
    float_id: str
    last_position: tuple[float, float]   # (lat, lon)
    last_time: datetime
    traj_path: Path                      # set by download_float_history, not here


def download_argo_floats_in_domain(region: Region) -> dict[str, ArgoPull]:
    """
    Pull every float currently reporting within `region` over the last 60 days.
    Returns {float_id: ArgoPull} keyed by WMO string.
    """
    import argopy

    end   = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=60)

    fetcher = argopy.DataFetcher(src="gdac").region([
        region.lon_min, region.lon_max,
        region.lat_min, region.lat_max,
        0, 2000,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    ])

    try:
        ds = fetcher.to_xarray()
    except Exception as exc:
        logger.warning("argopy region fetch failed: %s", exc)
        return {}

    if "PLATFORM_NUMBER" not in ds:
        return {}

    platform_raw = ds["PLATFORM_NUMBER"].values
    try:
        platform_strs = np.array([
            v.decode().strip() if isinstance(v, bytes) else str(v).strip()
            for v in platform_raw
        ])
    except Exception:
        platform_strs = platform_raw.astype(str)

    times = pd.to_datetime(ds["TIME"].values)
    lats  = ds["LATITUDE"].values.astype(float)
    lons  = ds["LONGITUDE"].values.astype(float)

    result: dict[str, ArgoPull] = {}
    for wmo in np.unique(platform_strs):
        mask  = platform_strs == wmo
        valid = mask & ~np.isnan(lats) & ~np.isnan(lons) & pd.notna(times)
        if not valid.any():
            continue
        idx = np.where(valid)[0][int(np.argmax(times[valid]))]
        result[wmo] = ArgoPull(
            float_id=wmo,
            last_position=(float(lats[idx]), float(lons[idx])),
            last_time=times[idx].to_pydatetime().replace(tzinfo=None),
            traj_path=Path(),
        )
    return result


# --------------------------------------------------------------------------- #
# 3. Full history pull
# --------------------------------------------------------------------------- #

def download_float_history(
    float_id: str,
    cache_dir: Path = ARGO_CACHE_DIR,
) -> Path:
    """
    Download Rtraj.nc (and _prof.nc) for float_id from the GDAC HTTP mirror.
    Returns local Rtraj.nc path; files are cached.
    """
    cache_dir  = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rtraj_path = cache_dir / f"{float_id}_Rtraj.nc"

    if rtraj_path.exists():
        return rtraj_path

    dac      = _find_dac(float_id)
    base_url = f"{GDAC_HTTP}/dac/{dac}/{float_id}"

    for fname in (f"{float_id}_Rtraj.nc", f"{float_id}_prof.nc"):
        url   = f"{base_url}/{fname}"
        local = cache_dir / fname
        if local.exists():
            continue
        for attempt in range(2):
            try:
                r = requests.get(url, timeout=120, stream=True)
                r.raise_for_status()
                with open(local, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                logger.info("Downloaded %s -> %s", url, local)
                break
            except requests.HTTPError as exc:
                logger.warning("Could not fetch %s: %s", url, exc)
                break  # 4xx/5xx won't improve on retry
            except requests.ConnectionError as exc:
                if attempt == 0:
                    logger.warning("Connection error fetching %s, retrying: %s", url, exc)
                else:
                    logger.warning("Connection error fetching %s (gave up): %s", url, exc)

    if not rtraj_path.exists():
        raise FileNotFoundError(
            f"Rtraj not found for float {float_id} (DAC={dac}). "
            f"Check {GDAC_HTTP}/dac/{dac}/{float_id}/"
        )
    return rtraj_path


def _find_dac(float_id: str) -> str:
    """Look up the float's DAC in the argopy Argo index."""
    import argopy

    try:
        idx = argopy.ArgoIndex(src="gdac")
        idx.load()
        df  = idx.to_dataframe()
        rows = df[df["wmo"] == int(float_id)]
        if rows.empty:
            raise ValueError(f"Float {float_id} not in Argo index")
        # 'dac' column is present; fall back to parsing 'file' if absent
        if "dac" in rows.columns:
            return str(rows["dac"].iloc[0])
        return rows["file"].iloc[0].split("/")[0]
    except Exception as exc:
        raise RuntimeError(f"Could not find DAC for float {float_id}: {exc}") from exc
