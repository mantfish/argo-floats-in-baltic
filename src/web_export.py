"""
web_export.py
=============
Convert in-memory floats_db / error_db to the JSON files consumed by the
static map and leaderboard pages in docs/.

Both functions are pure over their inputs (no fetching, no DB) -- they only
write to docs/data/.  Call them from run.py after the parquet saves.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .float_store import FloatRow
from .simulate import lookup_position, next_surfacing

logger = logging.getLogger(__name__)

DOCS_DATA_DIR = Path(__file__).parent.parent / "docs" / "data"
HISTORY_DAYS  = 30   # trajectory history window exported to the map


# Human-readable display names for the model keys
MODEL_DISPLAY = {
    "cmems": "CMEMS Baltic",
    "fcoo":  "FCOO GETM",
}

# Canonical model colors -- same palette used in both HTML pages
MODEL_COLOR = {
    "cmems": "#00D4FF",
    "fcoo":  "#FF8C42",
}


def export_floats(floats_db: dict[str, FloatRow], now: datetime) -> list[dict]:
    """
    Build floats.json: per-float current predictions, next-surfacing estimates,
    and recent trajectory history for each model.

    `now` is the reference time for "predicted_now" lookups and is embedded in
    the output so the frontend can show data age.
    """
    cutoff = now - timedelta(days=HISTORY_DAYS)
    floats_out: list[dict] = []

    for float_id, row in floats_db.items():
        if row.is_dead:
            continue

        anchor_lat, anchor_lon, anchor_time = row.last_real_position
        ca = row.cycle_action

        models_out: dict[str, dict] = {}
        for model, track in row.models.items():
            predicted = lookup_position(track.trajectory, now)

            surf_time = next_surfacing(anchor_time, ca, now)
            surf_pos  = lookup_position(track.trajectory, surf_time)
            if surf_pos is None and track.trajectory:
                # Trajectory doesn't reach the next surfacing time yet --
                # use last known point as best-available estimate.
                _, surf_lat, surf_lon = track.trajectory[-1]
                surf_pos = (surf_lat, surf_lon)

            history = [
                {"t": t.isoformat(), "lat": round(lat, 5), "lon": round(lon, 5)}
                for t, lat, lon in track.trajectory
                if t >= cutoff
            ]

            models_out[model] = {
                "display_name": MODEL_DISPLAY.get(model, model),
                "color": MODEL_COLOR.get(model, "#FFFFFF"),
                "predicted_now": (
                    {"lat": round(predicted[0], 5), "lon": round(predicted[1], 5)}
                    if predicted else None
                ),
                "next_surfacing_time": surf_time.isoformat(),
                "next_surfacing_position": (
                    {"lat": round(surf_pos[0], 5), "lon": round(surf_pos[1], 5)}
                    if surf_pos else None
                ),
                "trajectory_history": history,
                "missed_model_pulls": track.missed_model_pulls,
            }

        floats_out.append({
            "float_id": float_id,
            "euro_argo_url": f"https://fleetmonitoring.euro-argo.eu/float/{float_id}",
            "last_real_position": {
                "lat": round(anchor_lat, 5),
                "lon": round(anchor_lon, 5),
                "time": anchor_time.isoformat(),
            },
            "surfacing_history": [
                {"lat": round(lat, 5), "lon": round(lon, 5), "time": t.isoformat()}
                for lat, lon, t in row.surfacing_history
            ],
            "cycle_action": {
                "park_mode": ca.park_mode,
                "cycle_hours": ca.cycle_hours,
                "transmission_duration_minutes": ca.transmission_duration_minutes,
                "target_depth": ca.target_depth,
                "descent_speed_ms": ca.descent_speed_ms,
                "ascent_speed_ms": ca.ascent_speed_ms,
            },
            "models": models_out,
        })

    payload = {"updated_at": now.isoformat(), "floats": floats_out}
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DATA_DIR / "floats.json").write_text(json.dumps(payload, indent=2))
    logger.info("Exported %d floats to docs/data/floats.json", len(floats_out))
    return floats_out


MIN_DRIFT_M_FOR_PCT = 100.0   # below this, the float barely moved -- error/drift blows up on noise


def export_leaderboard(error_db: pd.DataFrame) -> dict:
    """
    Build leaderboard.json: per-model aggregate error statistics and the 50
    most recent scoring events.

    error_pct = error_m / drift_m, i.e. the position error as a fraction of
    how far the float actually drifted between its previous and current
    confirmed real surfacing -- a raw error_km alone reads very differently
    for a float that drifted 5km vs. 200km. Only computed where drift_m is
    present and exceeds MIN_DRIFT_M_FOR_PCT; older rows saved before drift_m
    was tracked, or events where the float barely moved, get None.
    """
    now = datetime.utcnow()

    models_stats: dict[str, dict] = {}
    recent_events: list[dict] = []

    if not error_db.empty:
        edb = error_db.copy()
        edb["t"] = pd.to_datetime(edb["t"])
        edb["error_km"] = edb["error_m"] / 1000.0

        if "drift_m" in edb.columns:
            has_drift = edb["drift_m"].notna() & (edb["drift_m"] >= MIN_DRIFT_M_FOR_PCT)
            edb["error_pct"] = pd.NA
            edb.loc[has_drift, "error_pct"] = (
                edb.loc[has_drift, "error_m"] / edb.loc[has_drift, "drift_m"] * 100.0
            )
        else:
            edb["error_pct"] = pd.NA

        cutoff_7d  = now - timedelta(days=7)
        cutoff_30d = now - timedelta(days=30)

        for model, sub in edb.groupby("model"):
            r7  = sub[sub["t"] >= cutoff_7d]["error_km"]
            r30 = sub[sub["t"] >= cutoff_30d]["error_km"]
            pct = sub["error_pct"].dropna().astype(float)

            models_stats[str(model)] = {
                "display_name": MODEL_DISPLAY.get(str(model), str(model)),
                "color": MODEL_COLOR.get(str(model), "#FFFFFF"),
                "n_total": int(len(sub)),
                "mean_error_km":   _round(sub["error_km"].mean()),
                "median_error_km": _round(sub["error_km"].median()),
                "mean_error_pct":   _round(pct.mean())   if len(pct) else None,
                "median_error_pct": _round(pct.median()) if len(pct) else None,
                "recent_7d": {
                    "n":        int(len(r7)),
                    "mean_km":  _round(r7.mean())  if len(r7)  else None,
                },
                "recent_30d": {
                    "n":        int(len(r30)),
                    "mean_km":  _round(r30.mean()) if len(r30) else None,
                },
            }

        for _, ev in edb.sort_values("t", ascending=False).head(50).iterrows():
            recent_events.append({
                "float_id": str(ev["float_id"]),
                "model":    str(ev["model"]),
                "t":        ev["t"].isoformat(),
                "error_km": _round(ev["error_km"]),
                "error_pct": _round(ev["error_pct"]) if pd.notna(ev["error_pct"]) else None,
            })

    result = {
        "updated_at": now.isoformat(),
        "model_colors": MODEL_COLOR,
        "models": models_stats,
        "recent_events": recent_events,
    }
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DATA_DIR / "leaderboard.json").write_text(json.dumps(result, indent=2))
    logger.info("Exported leaderboard (%d model(s)) to docs/data/leaderboard.json",
                len(models_stats))
    return result


def _round(v) -> float | None:
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return None
