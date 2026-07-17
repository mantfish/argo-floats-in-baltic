"""
float_store.py
===============
Persistent state for the leaderboard: the float registry (one row per
float, mode-voted cycle_action + reconciliation bookkeeping) and each
float's per-model simulated trajectories.

Storage format: parquet, two tables on disk:
    floats_meta.parquet   -- one row per float_id (scalar fields)
    trajectories.parquet  -- long format: one row per (float_id, model, t)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from .simulate import ControlAction


# --------------------------------------------------------------------------- #
# In-memory schema
# --------------------------------------------------------------------------- #

@dataclass
class ModelTrack:
    """One model's simulated trajectory for one float."""
    trajectory: list[tuple[datetime, float, float]] = field(default_factory=list)
    missed_model_pulls: int = 0


@dataclass
class FloatRow:
    float_id: str
    cycle_action: ControlAction
    last_real_position: tuple[float, float, datetime]   # (lat, lon, time)
    missed_pulls: int = 0
    is_dead: bool = False
    models: dict[str, ModelTrack] = field(default_factory=dict)
    surfacing_history: list[tuple[float, float, datetime]] = field(default_factory=list)  # (lat, lon, time) confirmed real surfacings

    def is_overdue(self, now: datetime, threshold_days: float = 10.0) -> bool:
        _, _, last_time = self.last_real_position
        return (now - last_time).total_seconds() / 86400.0 > threshold_days


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

def load_floats_db(store_dir: Path) -> dict[str, FloatRow]:
    """
    Reconstruct {float_id: FloatRow} from parquet on disk.
    Returns {} if no store exists yet (first-ever run).
    """
    store_dir = Path(store_dir)
    meta_path = store_dir / "floats_meta.parquet"
    traj_path = store_dir / "trajectories.parquet"

    if not meta_path.exists():
        return {}

    meta_df  = pd.read_parquet(meta_path)
    traj_df  = pd.read_parquet(traj_path) if traj_path.exists() else pd.DataFrame(
        columns=["float_id", "model", "t", "lat", "lon"]
    )
    surf_path = store_dir / "surfacings.parquet"
    surf_df   = pd.read_parquet(surf_path) if surf_path.exists() else pd.DataFrame(
        columns=["float_id", "t", "lat", "lon"]
    )

    # Build trajectory lookup: (float_id, model) -> sorted list of (t, lat, lon)
    traj_lookup: dict[tuple[str, str], list] = {}
    if not traj_df.empty:
        for (fid, model), grp in traj_df.groupby(["float_id", "model"]):
            grp = grp.sort_values("t")
            pts = [
                (_to_dt(row["t"]), float(row["lat"]), float(row["lon"]))
                for _, row in grp.iterrows()
            ]
            traj_lookup[(fid, model)] = pts

    # Build surfacing history lookup: float_id -> sorted list of (lat, lon, t)
    surf_lookup: dict[str, list] = {}
    if not surf_df.empty:
        for fid, grp in surf_df.groupby("float_id"):
            grp = grp.sort_values("t")
            surf_lookup[str(fid)] = [
                (float(r["lat"]), float(r["lon"]), _to_dt(r["t"]))
                for _, r in grp.iterrows()
            ]

    floats_db: dict[str, FloatRow] = {}
    for _, m in meta_df.iterrows():
        fid = str(m["float_id"])
        ca  = ControlAction(
            park_mode=str(m["park_mode"]),
            cycle_hours=float(m["cycle_hours"]),
            transmission_duration_minutes=float(m["transmission_duration_minutes"]),
            target_depth=None if pd.isna(m.get("target_depth")) else float(m["target_depth"]),
            descent_speed_ms=float(m["descent_speed_ms"]),
            ascent_speed_ms=float(m["ascent_speed_ms"]),
        )
        last_time = _to_dt(m["last_time"])

        # Find which models are stored for this float
        stored_models = {k[1] for k in traj_lookup if k[0] == fid}
        models: dict[str, ModelTrack] = {}
        for model in stored_models:
            missed = int(m.get(f"{model}_missed_model_pulls", 0) or 0)
            models[model] = ModelTrack(
                trajectory=traj_lookup.get((fid, model), []),
                missed_model_pulls=missed,
            )

        floats_db[fid] = FloatRow(
            float_id=fid,
            cycle_action=ca,
            last_real_position=(float(m["last_lat"]), float(m["last_lon"]), last_time),
            missed_pulls=int(m["missed_pulls"]),
            is_dead=bool(m["is_dead"]),
            models=models,
            surfacing_history=surf_lookup.get(fid, []),
        )

    return floats_db


def load_error_db(store_dir: Path) -> pd.DataFrame:
    """
    Read errors.parquet.  Returns empty DataFrame with correct columns if absent.
    """
    p = Path(store_dir) / "errors.parquet"
    if not p.exists():
        return pd.DataFrame(columns=[
            "float_id", "model", "t", "error_m", "drift_m",
            "real_lat", "real_lon", "predicted_lat", "predicted_lon",
            "leg_start_lat", "leg_start_lon",
        ])
    return pd.read_parquet(p)


def load_forecast_history(store_dir: Path) -> pd.DataFrame:
    """
    Read forecast_history.parquet -- one row per (float, model, pipeline run)
    recording what that run's trajectory predicted for the float's next
    surfacing, so successive predictions for the same real-world cycle can be
    compared to see whether they converge as fresher forecast data arrives.
    Returns empty DataFrame with correct columns if absent.
    """
    p = Path(store_dir) / "forecast_history.parquet"
    if not p.exists():
        return pd.DataFrame(columns=[
            "float_id", "cycle_number", "forecast_name", "forecast",
            "expected_surfacing_time", "expected_surfacing_lat", "expected_surfacing_lon",
        ])
    return pd.read_parquet(p)


def load_leg_history(store_dir: Path) -> pd.DataFrame:
    """
    Read leg_history.parquet -- the full simulated path of every completed
    forecast leg, one row per (float, model, cycle) point, up to and
    including the anchor reset. Captured in run._reconcile_with_argo right
    before a leg's trajectory is discarded on a real-surfacing reset (design
    decision 5), since that's the only moment the full path still exists --
    otherwise only its start/end survive. leg_end_time identifies which leg
    a point belongs to (the real surfacing time that closed it out), joinable
    against error_db's/scoring_history's `t`. Returns empty DataFrame with
    correct columns if absent.
    """
    p = Path(store_dir) / "leg_history.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["float_id", "model", "leg_end_time", "t", "lat", "lon"])
    return pd.read_parquet(p)


def load_cycle_action_history(store_dir: Path) -> pd.DataFrame:
    """
    Read cycle_action_history.parquet -- one row per (float, pipeline run)
    recording that run's re-derived ControlAction estimate (park_mode,
    cycle_hours, etc, mode-voted across the float's full real cycle history
    as of that run -- see run._refresh_cycle_actions), so you can see how
    the estimate evolved as more real cycles accumulated, not just its
    current value. `changed` is True if this row's action differs from the
    previous one for that float. Returns empty DataFrame with correct
    columns if absent.
    """
    p = Path(store_dir) / "cycle_action_history.parquet"
    if not p.exists():
        return pd.DataFrame(columns=[
            "float_id", "logged_at", "park_mode", "cycle_hours",
            "transmission_duration_minutes", "target_depth",
            "descent_speed_ms", "ascent_speed_ms", "changed",
        ])
    return pd.read_parquet(p)


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #

def save_floats_db(store_dir: Path, floats_db: dict[str, FloatRow]) -> None:
    """Flatten FloatRow dict to floats_meta.parquet + trajectories.parquet."""
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    meta_rows: list[dict] = []
    traj_rows: list[dict] = []

    for fid, row in floats_db.items():
        ca = row.cycle_action
        last_lat, last_lon, last_time = row.last_real_position

        meta: dict = {
            "float_id": fid,
            "park_mode": ca.park_mode,
            "cycle_hours": ca.cycle_hours,
            "transmission_duration_minutes": ca.transmission_duration_minutes,
            "target_depth": ca.target_depth,
            "descent_speed_ms": ca.descent_speed_ms,
            "ascent_speed_ms": ca.ascent_speed_ms,
            "last_lat": last_lat,
            "last_lon": last_lon,
            "last_time": last_time,
            "missed_pulls": row.missed_pulls,
            "is_dead": row.is_dead,
        }
        for model, track in row.models.items():
            meta[f"{model}_missed_model_pulls"] = track.missed_model_pulls

        meta_rows.append(meta)

        for model, track in row.models.items():
            for t, lat, lon in track.trajectory:
                traj_rows.append({"float_id": fid, "model": model, "t": t, "lat": lat, "lon": lon})

    surf_rows: list[dict] = []
    for fid, row in floats_db.items():
        for lat, lon, t in row.surfacing_history:
            surf_rows.append({"float_id": fid, "t": t, "lat": lat, "lon": lon})

    pd.DataFrame(meta_rows).to_parquet(store_dir / "floats_meta.parquet", index=False)
    pd.DataFrame(traj_rows).to_parquet(store_dir / "trajectories.parquet", index=False)
    pd.DataFrame(surf_rows).to_parquet(store_dir / "surfacings.parquet", index=False)


def save_error_db(store_dir: Path, error_db: pd.DataFrame) -> None:
    """Write error_db to errors.parquet (whole-frame overwrite)."""
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    error_db.to_parquet(store_dir / "errors.parquet", index=False)


def save_forecast_history(store_dir: Path, forecast_history_db: pd.DataFrame) -> None:
    """Write forecast_history_db to forecast_history.parquet (whole-frame overwrite)."""
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    forecast_history_db.to_parquet(store_dir / "forecast_history.parquet", index=False)


def save_cycle_action_history(store_dir: Path, cycle_action_history_db: pd.DataFrame) -> None:
    """Write cycle_action_history_db to cycle_action_history.parquet (whole-frame overwrite)."""
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    cycle_action_history_db.to_parquet(store_dir / "cycle_action_history.parquet", index=False)


def save_leg_history(store_dir: Path, leg_history_db: pd.DataFrame) -> None:
    """Write leg_history_db to leg_history.parquet (whole-frame overwrite)."""
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    leg_history_db.to_parquet(store_dir / "leg_history.parquet", index=False)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _to_dt(val) -> datetime:
    """Coerce a parquet time value (Timestamp, numpy datetime64, str) to datetime."""
    if isinstance(val, datetime):
        return val.replace(tzinfo=None)
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime().replace(tzinfo=None)
    return pd.Timestamp(val).to_pydatetime().replace(tzinfo=None)
