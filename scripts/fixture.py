"""
scripts/fixture.py
==================
Generate synthetic docs/data/*.json for local development and testing of the
map/leaderboard frontend, without needing the live data pipeline to have run.

Usage:  uv run scripts/fixture.py
"""

from __future__ import annotations

import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root so src imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.float_store import FloatRow, ModelTrack
from src.simulate import ControlAction
from src.web_export import export_floats, export_leaderboard
import pandas as pd

rng = random.Random(42)

NOW = datetime.utcnow().replace(microsecond=0)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _drift_trajectory(
    start_lat: float, start_lon: float, start_time: datetime,
    n_hours: int, u_ms: float = 0.05, v_ms: float = 0.03,
    noise: float = 0.02,
) -> list[tuple[datetime, float, float]]:
    """Build a simple hourly trajectory with slow drift + noise."""
    pts: list[tuple[datetime, float, float]] = []
    lat, lon = start_lat, start_lon
    for h in range(n_hours):
        t = start_time + timedelta(hours=h)
        u = u_ms + rng.gauss(0, noise)
        v = v_ms + rng.gauss(0, noise)
        # rough lat/lon update per hour
        lat += v * 3600 / 111_000
        lon += u * 3600 / (111_000 * math.cos(math.radians(lat)))
        pts.append((t, round(lat, 5), round(lon, 5)))
    return pts


def _make_float(
    wmo: str,
    anchor_lat: float, anchor_lon: float, anchor_time: datetime,
    park_mode: str = "park_on_bottom",
    cycle_hours: float = 120.0,
    target_depth: float | None = 60.0,
) -> FloatRow:
    ca = ControlAction(
        park_mode=park_mode,
        cycle_hours=cycle_hours,
        transmission_duration_minutes=30.0,
        target_depth=target_depth,
        descent_speed_ms=0.08,
        ascent_speed_ms=0.08,
    )

    models = {}
    for model, (du, dv) in {
        "cmems":    ( 0.04,  0.02),
        "fcoo_dk":  ( 0.06,  0.01),
        "fcoo_idk": ( 0.03,  0.03),
    }.items():
        traj = _drift_trajectory(
            anchor_lat, anchor_lon, anchor_time,
            n_hours=int((NOW - anchor_time).total_seconds() / 3600) + 1,
            u_ms=du, v_ms=dv,
        )
        models[model] = ModelTrack(trajectory=traj, missed_model_pulls=0)

    return FloatRow(
        float_id=wmo,
        cycle_action=ca,
        last_real_position=(anchor_lat, anchor_lon, anchor_time),
        missed_pulls=0,
        is_dead=False,
        models=models,
    )


# --------------------------------------------------------------------------- #
# Build fixture floats
# --------------------------------------------------------------------------- #

floats_db: dict[str, FloatRow] = {
    "1902766": _make_float(
        "1902766", 55.57, 15.91,
        anchor_time=NOW - timedelta(days=12),
        park_mode="park_on_bottom", cycle_hours=118.0, target_depth=55.0,
    ),
    "2904032": _make_float(
        "2904032", 55.15, 18.99,
        anchor_time=NOW - timedelta(days=4),
        park_mode="parking_depth", cycle_hours=96.0, target_depth=1000.0,
    ),
    "3902607": _make_float(
        "3902607", 56.36, 19.34,
        anchor_time=NOW - timedelta(days=7),
        park_mode="parking_depth", cycle_hours=240.0, target_depth=500.0,
    ),
    "4903900": _make_float(
        "4903900", 59.10, 20.86,
        anchor_time=NOW - timedelta(days=2),
        park_mode="drift_on_surface", cycle_hours=24.0, target_depth=None,
    ),
}

# --------------------------------------------------------------------------- #
# Build synthetic error history (last 45 days)
# --------------------------------------------------------------------------- #

error_rows = []
for wmo, row in floats_db.items():
    base_errors = {"cmems": 14.2, "fcoo_dk": 19.7, "fcoo_idk": 11.3}
    surf_interval_days = row.cycle_action.cycle_hours / 24.0
    t = NOW - timedelta(days=45)
    while t < NOW - timedelta(days=2):
        t += timedelta(days=surf_interval_days * rng.uniform(0.9, 1.1))
        for model, base_km in base_errors.items():
            err_km = max(0.5, rng.gauss(base_km, base_km * 0.4))
            error_rows.append({
                "float_id": wmo,
                "model": model,
                "t": t,
                "error_m": err_km * 1000,
            })

error_db = pd.DataFrame(error_rows)

# --------------------------------------------------------------------------- #
# Write JSON
# --------------------------------------------------------------------------- #

export_floats(floats_db, NOW)
export_leaderboard(error_db)
print(f"Wrote docs/data/floats.json ({len(floats_db)} floats)")
print(f"Wrote docs/data/leaderboard.json ({len(error_rows)} scoring events)")
