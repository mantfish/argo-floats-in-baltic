"""
data_handler.py
================
All external data acquisition for the CMEMS/FCOO leaderboard.

FCOO ("fcoo" model) is accessed via OPeNDAP through xarray's "pydap" engine
(not the default netCDF4 engine -- see _open_fcoo_dataset), against the
shared dk_nested.velocities.Z3D_<YYYYMMDDHH>.nc GETM product -- one file per
forecast run containing BOTH FCOO grids as separate variable groups, merged
here into a single model rather than two:
    dk  -- 1 nm North Sea + Baltic, full domain, 10 real depth levels (5-200 m)
    idk -- 600 m inner-Danish waters (a spatial nest, not a subgrid of dk),
           6 real depth levels (5-50 m)

The file is discovered (latest by run timestamp) and opened once per
pipeline run, shared between both grids' fetches.

CMEMS returns the pipeline's plain single-grid schema:
    dims : time, depth, lat, lon
    vars : u, v  (m/s, eastward / northward)
FCOO returns both grids merged, sharing one time coord but keeping separate
depth/lat/lon dims per grid (dk and idk are different meshes, not one
rectilinear grid -- they can't be merged into a single RegularGridInterpolator
input). See simulate.build_interpolators for how a single (u, v) query is
resolved from the two grids (idk preferred where its bounds cover the query
point, dk as fallback).

depth is each grid's real vertical levels -- no fabricated axis. Floats
queried below a grid's deepest level or above its shallowest get
fill_value=0.0 from RegularGridInterpolator, which is the correct behavior
once depth is real rather than duplicated.
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


MODELS = ("cmems", "fcoo")

CMEMS_DATASET_ID = "cmems_mod_bal_phy_anfc_PT1H-i_202411"
CMEMS_DEPTH_MAX  = 200.0   # metres -- floats don't go deeper in the Baltic

FCOO_BASE        = "https://data.fcoo.dk/webmap/v2/data/FCOO/GETM/"
FCOO_KNOTS_TO_MS = 0.514444
FCOO_FILL        = -9999.0

# Grid-variable suffixes (uu_<suffix>, latc_<suffix>, ...) within the shared
# dk_nested.velocities.Z3D_<ts>.nc file -- 'fcoo' merges both into one model
# (see simulate.build_interpolators: idk preferred, dk fallback).
_FCOO_GRID_DK  = "dk"
_FCOO_GRID_IDK = "idk"

# Filename prefix used to discover the latest shared Z3D file via _latest_getm_file
_FCOO_Z3D_PREFIX = "dk_nested.velocities.Z3D_"

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

    Returns an xr.Dataset with, for "cmems":
        dims : time, depth, lat, lon
        vars : u (m/s eastward), v (m/s northward)
    or for "fcoo" (both GETM grids merged, sharing one time coord --
    see simulate.build_interpolators for how these get resolved into a
    single current field):
        dims : time, depth_dk, lat_dk, lon_dk, depth_idk, lat_idk, lon_idk
        vars : u_dk, v_dk, u_idk, v_idk  (m/s eastward / northward)
    Does NOT trim -- caller must call trim_to_forecast_only() first.
    """
    if model == "cmems":
        return _fetch_cmems(region, issue_time, end_time)
    elif model == "fcoo":
        return _fetch_fcoo(region)
    else:
        raise ValueError(f"Unknown model {model!r}. Expected one of {MODELS}")


def trim_to_forecast_only(model_data: xr.Dataset, issue_time: datetime) -> xr.Dataset:
    """Drop every timestamp at-or-before issue_time (keep strict future only)."""
    issue_np = np.datetime64(issue_time.replace(tzinfo=None), "ns")
    return model_data.sel(time=model_data.time > issue_np)


def model_domain_bounds(model_data: xr.Dataset) -> tuple[float, float, float, float]:
    """
    (lat_min, lat_max, lon_min, lon_max) actually covered by `model_data`.

    Handles both schemas download_model_data can return: the single-grid
    "lat"/"lon" coords (cmems), and fcoo's merged dual-grid coords
    ("lat_dk"/"lat_idk" etc, no plain "lat") -- callers that just need the
    overall domain extent (e.g. run.py's in-domain check) shouldn't need to
    know which schema they got.
    """
    if "lat" in model_data.coords:
        lat = model_data["lat"].values
        lon = model_data["lon"].values
    else:
        lat = np.concatenate([model_data["lat_dk"].values, model_data["lat_idk"].values])
        lon = np.concatenate([model_data["lon_dk"].values, model_data["lon_idk"].values])
    return float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())


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
        minimum_depth=0.52,
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

_fcoo_z3d_url: Optional[str] = None
_fcoo_ds_cache: dict[str, xr.Dataset] = {}

# Timesteps per OPeNDAP request, per grid. 'idk' is a larger array per
# timestep than 'dk' (478x450x6 vs 362x290x10 points) and was observed live
# to hit the silent-corruption failure mode far more often at the same chunk
# size -- smaller chunks make each individual request less likely to trigger it.
_FCOO_TIME_CHUNK = {"dk": 8, "idk": 3}

# The corruption itself is a genuine, confirmed-intermittent upstream
# pydap/OPeNDAP server issue (reproduced live: identical requests fail then
# succeed then fail again, no correlation found with request size, caching,
# or time-of-day) -- retrying with a fresh connection is the right response,
# it just needs enough attempts/time to outlast a bad window. 5 attempts
# (~165s of backoff) wasn't always enough; 8 attempts with backoff capped at
# 60s (~330s total) gives a longer runway without any one retry ballooning.
_FCOO_LOAD_MAX_RETRIES = 8
_FCOO_LOAD_BACKOFF_CAP_S = 60


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


_fcoo_session: Optional[requests.Session] = None


def _get_fcoo_session() -> requests.Session:
    """
    Shared requests.Session carrying the browser User-Agent, reused for every
    OPeNDAP data request (not just the directory-listing scrape). Previously
    the browser UA was only ever attached to the plain requests.get() calls
    (_list_getm_files); the actual bulk data reads went through netCDF4's
    built-in DAP client, which has no hook for custom headers and so silently
    presented as a generic (non-browser) client. Routing those reads through
    pydap instead (see _open_fcoo_dataset) lets this session's UA apply there
    too.
    """
    global _fcoo_session
    if _fcoo_session is None:
        _fcoo_session = requests.Session()
        _fcoo_session.headers.update({"User-Agent": _BROWSER_UA})
    return _fcoo_session


def _open_fcoo_dataset(url: str) -> xr.Dataset:
    """Open a FCOO OPeNDAP URL via pydap, using the shared browser-UA session."""
    return xr.open_dataset(url, engine="pydap", session=_get_fcoo_session())


def _get_fcoo_z3d_url() -> str:
    """Discover (once) the latest shared dk_nested Z3D file URL."""
    global _fcoo_z3d_url
    if _fcoo_z3d_url is None:
        files = _list_getm_files()
        fname = _latest_getm_file(files, _FCOO_Z3D_PREFIX)
        if fname is None:
            raise RuntimeError(f"No dk_nested Z3D velocity file found at {FCOO_BASE}")
        _fcoo_z3d_url = FCOO_BASE + fname
        logger.info("FCOO Z3D: %s", fname)
    return _fcoo_z3d_url


_FCOO_MAX_PLAUSIBLE_KNOTS = 50.0   # far above any real current; anything past this is corruption


def _looks_valid(arr: np.ndarray) -> bool:
    """
    Cheap sanity check against observed silent-corruption failure modes:
      - a real velocity field always has a mix of NaN (land) and finite
        values; an all-zero or all-NaN array means the backend returned
        garbage rather than raising.
      - occasionally only *some* cells come back corrupted, as wildly
        implausible values (e.g. ~1e41) rather than all-zero/all-NaN --
        catch those too via a generous physical bound (raw units are knots).
    """
    if np.all(arr == 0) or np.all(np.isnan(arr)):
        return False
    finite = arr[np.isfinite(arr)]
    if finite.size and np.any(np.abs(finite) > _FCOO_MAX_PLAUSIBLE_KNOTS):
        return False
    return True


def _load_fcoo_var_chunked(url: str, var_name: str, sel: dict, n_time: int, chunk_size: int) -> np.ndarray:
    """
    Load a (time, depth, lat, lon) variable in small time chunks, each
    retried independently (fresh connection) on failure or on a suspicious
    all-zero/all-NaN result.

    Chunking -- rather than one .isel().load() for all 56 timesteps at once
    -- is deliberate: a full-size single-shot request was observed live to
    come back silently corrupted (all-zero, no exception), more often for
    larger requests. This mirrors why the original pydap-based fetch chunked
    by ~8 timesteps too.
    """
    chunks = []
    ds = _open_fcoo_dataset(url)
    for t0 in range(0, n_time, chunk_size):
        t1 = min(t0 + chunk_size, n_time)
        chunk_sel = dict(sel, time=slice(t0, t1))
        last_exc: Optional[Exception] = None
        for attempt in range(_FCOO_LOAD_MAX_RETRIES):
            try:
                if attempt > 0:
                    ds = _open_fcoo_dataset(url)  # fresh connection on retry
                arr = ds[var_name].isel(**chunk_sel).load().values
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "FCOO %s t[%d:%d] attempt %d/%d raised %s",
                    var_name, t0, t1, attempt + 1, _FCOO_LOAD_MAX_RETRIES, exc,
                )
                time.sleep(min(5 * 2 ** attempt, _FCOO_LOAD_BACKOFF_CAP_S))
                continue

            if _looks_valid(arr):
                chunks.append(arr)
                break

            logger.warning(
                "FCOO %s t[%d:%d] attempt %d/%d returned a suspicious "
                "(all-zero/all-NaN/implausible-magnitude) chunk, retrying "
                "with a fresh connection",
                var_name, t0, t1, attempt + 1, _FCOO_LOAD_MAX_RETRIES,
            )
            time.sleep(min(5 * 2 ** attempt, _FCOO_LOAD_BACKOFF_CAP_S))
        else:
            raise RuntimeError(
                f"FCOO {var_name} t[{t0}:{t1}]: repeatedly got invalid data from {url}"
            ) from last_exc

    return np.concatenate(chunks, axis=0)


def _fetch_fcoo(region: Region) -> xr.Dataset:
    """
    Fetch both FCOO grids from the shared dk_nested Z3D file and merge them
    into one dataset for the unified 'fcoo' model. Both grids share one
    "time" coord (so trim_to_forecast_only/_is_empty/_last_timestamp in
    run.py keep working unchanged) but keep their own depth/lat/lon dims,
    since 'idk' (finer, inner-Danish-waters only) is not a subgrid of 'dk'
    (coarser, full domain) -- simulate.build_interpolators resolves the two
    into a single current field at query time (idk preferred, dk fallback).
    """
    if "fcoo" not in _fcoo_ds_cache:
        url = _get_fcoo_z3d_url()
        dk = _build_fcoo_dataset(url, _FCOO_GRID_DK, region)
        idk = _build_fcoo_dataset(url, _FCOO_GRID_IDK, region)
        _fcoo_ds_cache["fcoo"] = xr.Dataset(
            {
                "u_dk":  (["time", "depth_dk", "lat_dk", "lon_dk"], dk["u"].values),
                "v_dk":  (["time", "depth_dk", "lat_dk", "lon_dk"], dk["v"].values),
                "u_idk": (["time", "depth_idk", "lat_idk", "lon_idk"], idk["u"].values),
                "v_idk": (["time", "depth_idk", "lat_idk", "lon_idk"], idk["v"].values),
            },
            coords={
                "time":      dk["time"].values,
                "depth_dk":  dk["depth"].values,
                "lat_dk":    dk["lat"].values,
                "lon_dk":    dk["lon"].values,
                "depth_idk": idk["depth"].values,
                "lat_idk":   idk["lat"].values,
                "lon_idk":   idk["lon"].values,
            },
        )
    return _fcoo_ds_cache["fcoo"]


def _build_fcoo_dataset(url: str, suffix: str, region: Region) -> xr.Dataset:
    """
    Subset one grid ('dk' or 'idk') of the shared Z3D dataset to `region`
    and normalize to the pipeline's standard schema:
        dims : time, depth, lat, lon
        vars : u, v  (m/s eastward / northward)
    depth is the grid's real vertical levels -- no fabricated duplicate-
    surface depth axis.
    """
    lat_name, lon_name, zax_name = f"latc_{suffix}", f"lonc_{suffix}", f"zax_{suffix}"
    u_name, v_name = f"uu_{suffix}", f"vv_{suffix}"

    ds = _open_fcoo_dataset(url)   # cheap: metadata + small 1-D coords only, here
    latc = ds[lat_name].values
    lonc = ds[lon_name].values
    times = ds["time"].values
    n_time = ds.sizes["time"]

    lat_idx = np.where((latc >= region.lat_min) & (latc <= region.lat_max))[0]
    lon_idx = np.where((lonc >= region.lon_min) & (lonc <= region.lon_max))[0]
    if not lat_idx.size or not lon_idx.size:
        raise RuntimeError(f"FCOO {suffix} grid has no points inside the pipeline region")
    lat0, lat1 = int(lat_idx[0]), int(lat_idx[-1])
    lon0, lon1 = int(lon_idx[0]), int(lon_idx[-1])

    sel = {lat_name: slice(lat0, lat1 + 1), lon_name: slice(lon0, lon1 + 1)}
    logger.info(
        "FCOO %s subset: %d lat x %d lon, %d depths, %d timesteps",
        suffix, lat1 - lat0 + 1, lon1 - lon0 + 1, ds.sizes[zax_name], n_time,
    )

    # dims are already (time, zax, latc, lonc) == (time, depth, lat, lon) order
    chunk_size = _FCOO_TIME_CHUNK[suffix]
    uu = _load_fcoo_var_chunked(url, u_name, sel, n_time, chunk_size)
    vv = _load_fcoo_var_chunked(url, v_name, sel, n_time, chunk_size)

    # Defensive re-mask even though xr.open_dataset already CF-decodes
    # _FillValue -> NaN (belt-and-suspenders in case decoding is ever bypassed).
    u = np.where(uu == FCOO_FILL, np.nan, uu).astype(np.float32) * FCOO_KNOTS_TO_MS
    v = np.where(vv == FCOO_FILL, np.nan, vv).astype(np.float32) * FCOO_KNOTS_TO_MS

    return xr.Dataset(
        {"u": (["time", "depth", "lat", "lon"], u),
         "v": (["time", "depth", "lat", "lon"], v)},
        coords={
            "time":  times,
            "depth": ds[zax_name].values.astype(np.float64),
            "lat":   latc[lat0:lat1 + 1].astype(np.float64),
            "lon":   lonc[lon0:lon1 + 1].astype(np.float64),
        },
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
    force_refresh: bool = False,
) -> Path:
    """
    Download Rtraj.nc (and _prof.nc) for float_id from the GDAC HTTP mirror.
    Returns local Rtraj.nc path; files are cached by default.

    force_refresh=True re-downloads even if a local copy already exists.
    GDAC's Rtraj.nc grows in place as a float completes new real cycles --
    callers that want an up-to-date cycle_action estimate (not just whatever
    was true the first time this float was ever seen) need force_refresh so
    the cache doesn't silently freeze the history at its earliest snapshot.
    """
    cache_dir  = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rtraj_path = cache_dir / f"{float_id}_Rtraj.nc"
    prof_path  = cache_dir / f"{float_id}_prof.nc"

    # Both files must already be cached to skip fetching -- Rtraj.nc alone
    # existing (e.g. from an interrupted earlier fetch) used to short-circuit
    # here and silently never attempt prof.nc, leaving cycle_extractor stuck
    # on its raw-fix fallback for that float indefinitely.
    if rtraj_path.exists() and prof_path.exists() and not force_refresh:
        return rtraj_path

    dac      = _find_dac(float_id)
    base_url = f"{GDAC_HTTP}/dac/{dac}/{float_id}"

    for fname in (f"{float_id}_Rtraj.nc", f"{float_id}_prof.nc"):
        url   = f"{base_url}/{fname}"
        local = cache_dir / fname
        if local.exists() and not force_refresh:
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
