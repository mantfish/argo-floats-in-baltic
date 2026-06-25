"""
data_handler.py
================
All external data acquisition for the CMEMS/FCOO leaderboard.

FCOO data is accessed via OPeNDAP (pydap) -- two products:
    fcoo_idk -- 600 m inner-Danish grid  (idk-600m_3D-velocities_surface)
    fcoo_dk  -- 1 nm North Sea + Baltic  (nsbalt-1nm_velocities_surface)

Chunked pydap requests (~8 timesteps each) replace the previous single
1.85 GB HTTP download, which was reliably dropped by the server on
GitHub Actions runners (Azure data-centre IPs).

All returned datasets follow a standard schema so the rest of the
pipeline never needs to know which model it's working with:
    dims : time, depth, lat, lon
    vars : u, v  (m/s, eastward / northward)

depth has two levels [0, 2000] m with identical surface velocities at
both -- this keeps build_interpolators (RegularGridInterpolator) in-bounds
for all queried depths rather than returning fill_value=0 for depth
queries that exceed a singleton axis.
"""

from __future__ import annotations

import logging
import re
import time
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

FCOO_BASE        = "https://data.fcoo.dk/webmap/v2/data/FCOO/GETM/"
FCOO_KNOTS_TO_MS = 0.514444
FCOO_FILL        = -9999.0

# OPeNDAP product prefix for each model key
_FCOO_PREFIX = {
    "fcoo_idk": "idk-600m_3D-velocities_surface",
    "fcoo_dk":  "nsbalt-1nm_velocities_surface",
}

# Browser User-Agent so institutional servers don't block the requests
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

GDAC_HTTP      = "https://data-argo.ifremer.fr"
ARGO_CACHE_DIR = Path("data/argo_cache")


# --------------------------------------------------------------------------- #
# 1. Model current fields
# --------------------------------------------------------------------------- #

def download_model_data(
    model: str,
    region: Region,
    issue_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> xr.Dataset:
    """
    Pull the latest forecast for `model` over `region`.

    issue_time / end_time: optional start/end window passed to the CMEMS
    subsetting API to avoid downloading the full archive.  FCOO ignores them
    (the file on the server already covers a fixed window).

    Returns an xr.Dataset with:
        dims : time, depth, lat, lon
        vars : u (m/s eastward), v (m/s northward)
    Does NOT trim -- caller must call trim_to_forecast_only() first.
    """
    if model == "cmems":
        return _fetch_cmems(region, issue_time, end_time)
    elif model in _FCOO_PREFIX:
        return _fetch_fcoo(model, region)
    else:
        raise ValueError(f"Unknown model {model!r}. Expected one of {MODELS}")


def trim_to_forecast_only(model_data: xr.Dataset, issue_time: datetime) -> xr.Dataset:
    """Drop every timestamp at-or-before issue_time (keep strict future only)."""
    issue_np = np.datetime64(issue_time.replace(tzinfo=None), "ns")
    return model_data.sel(time=model_data.time > issue_np)


# -- CMEMS -------------------------------------------------------------------

def _fetch_cmems(
    region: Region,
    issue_time: Optional[datetime],
    end_time: Optional[datetime] = None,
) -> xr.Dataset:
    """
    Download a CMEMS subset to a local NetCDF4 file, then load it into memory.

    Uses copernicusmarine.subset() (a single HTTP download) instead of
    open_dataset() (hundreds of parallel dask/zarr chunk requests).  The
    single-stream download is faster on CI runners, avoids connection-pool
    exhaustion, and sidesteps the Python 3.14 + dask SIGSEGV.

    Spatial grid is subsampled 2x after loading (stride-indexed reads, so
    only ~2.5 GB is pulled into memory rather than ~10 GB for the full grid).
    """
    import copernicusmarine
    import tempfile
    import os

    kwargs: dict = dict(
        dataset_id=CMEMS_DATASET_ID,
        variables=["uo", "vo"],
        minimum_latitude=region.lat_min,
        maximum_latitude=region.lat_max,
        minimum_longitude=region.lon_min,
        maximum_longitude=region.lon_max,
        minimum_depth=0.0,
        maximum_depth=CMEMS_DEPTH_MAX,
        output_filename="cmems_subset.nc",
    )
    if issue_time is not None:
        kwargs["start_datetime"] = issue_time.strftime("%Y-%m-%dT%H:%M:%S")
    if end_time is not None:
        kwargs["end_datetime"] = end_time.strftime("%Y-%m-%dT%H:%M:%S")

    with tempfile.TemporaryDirectory() as tmp_dir:
        kwargs["output_directory"] = tmp_dir
        tmp_path = os.path.join(tmp_dir, "cmems_subset.nc")

        logger.info("Downloading CMEMS subset (single-stream)…")
        copernicusmarine.subset(**kwargs)
        logger.info("CMEMS subset downloaded, loading into memory")

        # Open without dask -- file-backed lazy reads via xarray+netCDF4.
        # isel with a slice does a strided read from the file so peak memory
        # is only the subsampled shape, not the full spatial grid.
        ds = xr.open_dataset(tmp_path, chunks=None, mask_and_scale=True)

        lat_dim = "latitude" if "latitude" in ds.dims else "lat"
        lon_dim = "longitude" if "longitude" in ds.dims else "lon"
        ds = ds.isel(**{lat_dim: slice(None, None, 2), lon_dim: slice(None, None, 2)})

        rename = {}
        for old, new in [("uo", "u"), ("vo", "v"), ("latitude", "lat"), ("longitude", "lon")]:
            if old in ds:
                rename[old] = new
        ds = ds.rename(rename) if rename else ds

        # Force into a plain in-memory Dataset (no file handles) before the
        # temp directory is deleted.  copy=False avoids a ~2 GB transient
        # duplicate since CMEMS already stores uo/vo as float32.
        return xr.Dataset(
            {
                "u": (["time", "depth", "lat", "lon"], ds["u"].values.astype(np.float32, copy=False)),
                "v": (["time", "depth", "lat", "lon"], ds["v"].values.astype(np.float32, copy=False)),
            },
            coords={
                "time":  ds["time"].values,
                "depth": ds["depth"].values.astype(np.float64),
                "lat":   ds["lat"].values.astype(np.float64),
                "lon":   ds["lon"].values.astype(np.float64),
            },
        )


# -- FCOO (OPeNDAP) ----------------------------------------------------------

_fcoo_ds_cache: dict[str, xr.Dataset] = {}


def _fetch_fcoo(model: str, region: Region) -> xr.Dataset:
    """Download the requested FCOO grid via OPeNDAP and return a pipeline Dataset."""
    if model not in _fcoo_ds_cache:
        files = _list_getm_files()
        if model == "fcoo_idk":
            _fcoo_ds_cache[model] = _fetch_fcoo_idk(files, region)
        else:
            _fcoo_ds_cache[model] = _fetch_fcoo_nsbalt(files, region)
    return _fcoo_ds_cache[model]


def _list_getm_files() -> list[str]:
    """Scrape the FCOO GETM directory for available NetCDF filenames."""
    resp = requests.get(FCOO_BASE, timeout=20, headers={"User-Agent": _BROWSER_UA})
    resp.raise_for_status()
    # hrefs end in .nc.html; extract just the .nc filename
    return re.findall(r'([^\s/"]+\.nc)\.html', resp.text)


def _latest_getm_file(files: list[str], prefix: str) -> str | None:
    """Return the filename with the newest run timestamp for a given product prefix."""
    matches = [f for f in files if f.startswith(prefix)]
    if not matches:
        return None

    def _ts(name: str) -> datetime:
        m = re.search(r"_(\d{10})\.nc$", name)
        return datetime.strptime(m.group(1), "%Y%m%d%H") if m else datetime.min

    return max(matches, key=_ts)


def _fetch_coords_1d(fname: str, *varnames: str) -> dict[str, np.ndarray]:
    """Fetch 1-D coordinate arrays from the OPeNDAP .ascii endpoint."""
    url = FCOO_BASE + fname + ".ascii?" + ",".join(varnames)
    r = requests.get(url, timeout=30, headers={"User-Agent": _BROWSER_UA})
    r.raise_for_status()

    coords: dict[str, np.ndarray] = {}
    current_var: str | None = None
    vals: list[float] = []
    for line in r.text.split("\n"):
        line = line.strip()
        if line in varnames:
            if current_var and vals:
                coords[current_var] = np.array(vals)
            current_var = line
            vals = []
        elif line.startswith("[") and current_var:
            vals.append(float(line.split("]", 1)[1].strip()))
    if current_var and vals:
        coords[current_var] = np.array(vals)
    return coords


def _fetch_time_coord(fname: str) -> np.ndarray:
    """
    Fetch the time coordinate from OPeNDAP .ascii and return as datetime64[s].
    Trims zero-padded trailing slots (FCOO NetCDF3 preallocates 56 slots but
    only writes the filled prefix).
    """
    coords = _fetch_coords_1d(fname, "time")
    t_raw = coords.get("time", np.array([]))
    if not len(t_raw):
        return t_raw

    # Parse epoch string from .das metadata
    das = requests.get(
        FCOO_BASE + fname + ".das", timeout=15, headers={"User-Agent": _BROWSER_UA}
    ).text
    m = re.search(r'units\s+"seconds since (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"', das)
    if not m:
        raise RuntimeError(f"Cannot parse time units from {FCOO_BASE + fname}.das")
    epoch = np.datetime64(m.group(1).replace(" ", "T"), "s")

    # Drop zero-padded trailing slots (raw value 0 repeats the epoch timestamp)
    diffs = np.diff(t_raw)
    n_valid = int(np.argmax(diffs <= 0)) + 1 if (diffs <= 0).any() else len(t_raw)
    return epoch + t_raw[:n_valid].astype("timedelta64[s]")


def _pydap_open(fname: str):
    from pydap.client import open_url
    return open_url(FCOO_BASE + fname)


def _fetch_variable_chunked(
    fname: str,
    varname: str,
    lat0: int, lat1: int,
    lon0: int, lon1: int,
    chunk_size: int = 8,
    max_retries: int = 5,
) -> np.ndarray:
    """
    Fetch a (time, lat, lon) variable via pydap in time chunks of chunk_size.
    Each chunk is retried independently with exponential backoff and a fresh
    OPeNDAP connection on retry, so a dropped connection never aborts the
    whole download.
    """
    ds  = _pydap_open(fname)
    var = ds[varname]
    n_time = var.shape[0]
    chunks = []

    for t_start in range(0, n_time, chunk_size):
        t_end = min(t_start + chunk_size - 1, n_time - 1)
        for attempt in range(max_retries):
            try:
                raw = var[t_start:t_end + 1, lat0:lat1 + 1, lon0:lon1 + 1]
                chunks.append(np.asarray(raw.data).astype(np.float32))
                logger.debug("  %s t[%d:%d] ok", varname, t_start, t_end)
                break
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                wait = 5 * 2 ** attempt   # 5 s, 10 s, 20 s, 40 s
                logger.warning(
                    "  %s t[%d:%d] attempt %d failed (%s), retry in %ds",
                    varname, t_start, t_end, attempt + 1, exc, wait,
                )
                time.sleep(wait)
                ds  = _pydap_open(fname)  # fresh connection
                var = ds[varname]

    return np.concatenate(chunks, axis=0)


def _build_fcoo_dataset(
    uu: np.ndarray,
    vv: np.ndarray,
    latc: np.ndarray,
    lonc: np.ndarray,
    times: np.ndarray,
) -> xr.Dataset:
    """
    Package uu/vv arrays into the pipeline's standard (time, depth, lat, lon) schema.

    depth = [0, 2000] m with surface velocities duplicated at both levels.
    RegularGridInterpolator then returns the surface current at any queried
    depth (interpolating between two equal values) rather than returning
    fill_value=0 for out-of-bounds depth queries.
    """
    u = np.where(uu == FCOO_FILL, np.nan, uu).astype(np.float32) * FCOO_KNOTS_TO_MS
    v = np.where(vv == FCOO_FILL, np.nan, vv).astype(np.float32) * FCOO_KNOTS_TO_MS

    # Duplicate surface layer on depth axis: (time, lat, lon) -> (time, 2, lat, lon)
    u2 = np.stack([u, u], axis=1)
    v2 = np.stack([v, v], axis=1)

    return xr.Dataset(
        {"u": (["time", "depth", "lat", "lon"], u2),
         "v": (["time", "depth", "lat", "lon"], v2)},
        coords={
            "time":  times,
            "depth": np.array([0.0, 2000.0]),
            "lat":   latc.astype(np.float64),
            "lon":   lonc.astype(np.float64),
        },
    )


def _fetch_fcoo_idk(files: list[str], region: Region) -> xr.Dataset:
    """Fetch IDK (600 m inner-Danish) surface velocities via OPeNDAP."""
    fname = _latest_getm_file(files, _FCOO_PREFIX["fcoo_idk"])
    if fname is None:
        raise RuntimeError(f"No IDK velocity file found at {FCOO_BASE}")
    logger.info("FCOO IDK: %s", fname)

    coords = _fetch_coords_1d(fname, "latc", "lonc")
    latc, lonc = coords["latc"], coords["lonc"]
    times = _fetch_time_coord(fname)
    n_t = len(times)

    lat_idx = np.where((latc >= region.lat_min) & (latc <= region.lat_max))[0]
    lon_idx = np.where((lonc >= region.lon_min) & (lonc <= region.lon_max))[0]
    if not lat_idx.size or not lon_idx.size:
        raise RuntimeError("IDK grid has no points inside the pipeline region")
    lat0, lat1 = int(lat_idx[0]), int(lat_idx[-1])
    lon0, lon1 = int(lon_idx[0]), int(lon_idx[-1])

    logger.info("IDK subset: %d lat × %d lon, %d timesteps",
                lat1 - lat0 + 1, lon1 - lon0 + 1, n_t)

    uu = _fetch_variable_chunked(fname, "uu", lat0, lat1, lon0, lon1)[:n_t]
    vv = _fetch_variable_chunked(fname, "vv", lat0, lat1, lon0, lon1)[:n_t]

    return _build_fcoo_dataset(uu, vv, latc[lat0:lat1 + 1], lonc[lon0:lon1 + 1], times)


def _fetch_fcoo_nsbalt(files: list[str], region: Region) -> xr.Dataset:
    """
    Fetch NSBALT (1 nm North Sea + Baltic) surface velocities via OPeNDAP.
    Subsets to region before downloading; chunked to keep each request small.
    """
    fname = _latest_getm_file(files, _FCOO_PREFIX["fcoo_dk"])
    if fname is None:
        raise RuntimeError(f"No NSBALT velocity file found at {FCOO_BASE}")
    logger.info("FCOO NSBALT: %s", fname)

    coords = _fetch_coords_1d(fname, "latc", "lonc")
    latc, lonc = coords["latc"], coords["lonc"]
    times = _fetch_time_coord(fname)
    n_t = len(times)

    lat_idx = np.where((latc >= region.lat_min) & (latc <= region.lat_max))[0]
    lon_idx = np.where((lonc >= region.lon_min) & (lonc <= region.lon_max))[0]
    if not lat_idx.size or not lon_idx.size:
        raise RuntimeError("NSBALT grid has no points inside the pipeline region")
    lat0, lat1 = int(lat_idx[0]), int(lat_idx[-1])
    lon0, lon1 = int(lon_idx[0]), int(lon_idx[-1])

    n_lat, n_lon = lat1 - lat0 + 1, lon1 - lon0 + 1
    frac = (n_lat * n_lon) / (len(latc) * len(lonc))
    logger.info("NSBALT subset: %d lat × %d lon (%.0f%% of grid), %d timesteps",
                n_lat, n_lon, frac * 100, n_t)

    uu = _fetch_variable_chunked(fname, "uu", lat0, lat1, lon0, lon1)[:n_t]
    vv = _fetch_variable_chunked(fname, "vv", lat0, lat1, lon0, lon1)[:n_t]

    return _build_fcoo_dataset(uu, vv, latc[lat0:lat1 + 1], lonc[lon0:lon1 + 1], times)


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
