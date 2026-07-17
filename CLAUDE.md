# CMEMS / FCOO Argo Leaderboard

## What this is

An operational leaderboard that scores ocean current forecast models
(CMEMS, FCOO/GETM) against where Argo profiling floats actually surface, in
Danish/Baltic waters. Each tracked float gets one simulated trajectory per
model; when the float really surfaces, both models' predicted positions at
that timestamp are compared against the real one, and the error goes into
`error_db`.

This is **not** the EKF/MPC float-piloting project (Argo+). That project
estimates state and controls a float in real time, with covariance, bias,
and process noise. This one runs a much simpler, deterministic forward
simulation purely to evaluate forecast skill retrospectively. They share a
*name* (`ControlAction`) for an unrelated reason explained below, and
nothing else.

## Architecture

Five modules, each with one job. Arrows show what calls what.

```
run.py
 ├─ data_handler.py     (fetch: model currents, Argo population, float history)
 ├─ float_store.py      (load/save: FloatRow / ModelTrack, parquet on disk)
 ├─ cycle_extractor.py  (compute: .traj history -> ControlAction)
 │   └─ simulate.py     (defines ControlAction; cycle_extractor imports it)
 └─ simulate.py          (compute: forward-integrate a trajectory)
```

`float_store.py` does no fetching and no simulation -- pure data in/out.
`data_handler.py` does no simulation and doesn't know about `FloatRow`.
`cycle_extractor.py` and `simulate.py` do no I/O at all. `run.py` is the
only module that knows about all the others; it owns every policy constant
(thresholds, region bounds) so changing one never requires touching the
modules it calls.

## Data model

**`FloatRow`** (`float_store.py`) -- one per float, keyed by WMO id:

```python
FloatRow:
    float_id: str
    cycle_action: ControlAction        # mode-voted across history, or default_action()
    last_real_position: (lat, lon, time)  # last CONFIRMED real surfacing
    missed_pulls: int                  # consecutive absences from the Argo feed
    is_dead: bool                      # missed_pulls >= DEAD_THRESHOLD
    models: {model_name: ModelTrack}
```

**`ModelTrack`**:

```python
ModelTrack:
    trajectory: [(t, lat, lon), ...]   # ordered; trajectory[0] is always the anchor
    missed_model_pulls: int            # consecutive cycles with no new forecast data
```

**`ControlAction`** (`simulate.py`) -- one float's representative dive profile:

```python
ControlAction:
    park_mode: "park_on_bottom" | "parking_depth" | "drift_on_surface"
    cycle_hours: float          # ⚠ descent + parking time ONLY -- see "Gotchas" below
    transmission_duration_minutes: float
    target_depth: float | None  # dbar; None only for drift_on_surface
    descent_speed_ms: float
    ascent_speed_ms: float
```

**`ArgoPull`** (`data_handler.py`) -- one float's state in a single domain pull:
`float_id`, `last_position`, `last_time`, `traj_path`.

**Storage** (`float_store.py`): parquet, seven tables in `STORE_DIR`:
`floats_meta.parquet` (scalar per-float fields), `trajectories.parquet`
(long format: `float_id, model, t, lat, lon`), `surfacings.parquet` (real
confirmed surfacing history, long format: `float_id, t, lat, lon`),
`errors.parquet` (`error_db`: `float_id, model, t, error_m, drift_m,
real_lat, real_lon, predicted_lat, predicted_lon, leg_start_lat,
leg_start_lon`), `forecast_history.parquet`
(one row per float/model/pipeline-run predicted next surfacing, see
`run._extend_trajectories`), `cycle_action_history.parquet` (one row per
float/pipeline-run re-derived `cycle_action`, see `run._refresh_cycle_actions`
and design decision 9 below), and `leg_history.parquet` (long format:
`float_id, model, leg_end_time, t, lat, lon` -- every point of a completed
forecast leg's actual simulated path, captured in `run._reconcile_with_argo`
right before an anchor reset would otherwise discard it; `leg_end_time` is
the real surfacing time that closed the leg out, joinable against
`errors.parquet`'s `t`). All seven are missing-file-tolerant on load
(`float_store.load_*` returns an empty/correctly-columned result rather than
erroring) and whole-frame-overwritten on save.

## Design decisions worth knowing before you touch this code

These were each settled deliberately, across a long design conversation --
listed here so a fresh implementer doesn't accidentally re-litigate or
silently violate them.

1. **No `surfaced` flag anywhere.** Trajectories just keep advecting
   forward indefinitely on the same `cycle_action`, repeating the dive
   profile, until a real Argo ping resets the anchor. There's no flag to
   check or maintain.

2. **Forced scoring on real-data arrival, not on model-internal state.**
   A float is scored the instant a new real surfacing is confirmed in the
   Argo feed -- never based on whether either model's own simulation
   thinks it has "surfaced."

3. **Trajectories are timestamp-indexed, not last-point.** Because models
   keep advecting into the future between forecast pulls, the trajectory
   tip is usually *ahead* of the real world. Scoring requires a lookup at
   the real surfacing's exact timestamp (`_lookup` in `run.py`), never
   "compare to the last point."

4. **`trim_to_forecast_only` exists to keep analysis-grade data out of
   already-simulated segments.** Each model pull is trimmed to strictly
   after the previous issue time before being used to extend a trajectory.
   This is what makes the leaderboard measure *forecast* skill rather than
   a mix of forecast skill and "how much better analysis data is than
   forecast data." Do not let this trimming get bypassed or the metric's
   meaning quietly changes.

5. **Anchor reset happens exactly at confirmed real surfacings.** When a
   new real ping lands, `trajectory` is reset to a single point:
   `(real_time, real_lat, real_lon)`. Phase (descending/parking/ascending/
   communicating) is *not* stored anywhere -- it's recovered in
   `simulate_cycle` from elapsed time since that anchor, modulo one full
   cycle duration. This recovery is only correct because of this exact
   reset behavior; if anchor resets ever stop happening at every confirmed
   surfacing, phase recovery breaks silently. Immediately before the reset,
   `_reconcile_with_argo` copies the outgoing trajectory's points (up to and
   including `real_time`) into `leg_history_rows` -> `leg_history.parquet`,
   since this is the only moment that completed leg's full simulated path
   still exists -- purely for later display/audit (`web_export`'s
   `scoring_history[*].predictions[model].path`, drawn by `map.html`), not
   read by any live simulation logic.

6. **Freeze-on-gap, not backfill.** If no new model data arrives, or it
   doesn't extend past where a row already is, that row's trajectory is
   simply not extended this cycle (`missed_model_pulls` increments). There
   is deliberately no "resimulate from further back once better data
   shows up" logic -- see point 4.

7. **Overdue (10+ days since last real surfacing) excludes a float from
   scoring but does not kill it.** Computed on demand
   (`FloatRow.is_overdue`) from `last_real_position`'s timestamp, never
   stored as a separate boolean -- there is nothing to keep in sync. A
   fresh ping clears it for free. This is judged against the **new ping's
   timestamp**, not wall-clock "now," so pipeline lag in the Argo feed
   doesn't affect the determination.

8. **`missed_pulls` (Argo-feed absence) and `missed_model_pulls`
   (model-feed staleness) are deliberately separate counters,** tracking
   different failure modes that happen to look similar (a frozen
   trajectory sitting around for a while). Don't conflate them.

9. **`cycle_action` is mode-voted across a float's full `.traj` history,
   every run, not just at registration** (`run._derive_cycle_action`, called
   from both `_build_new_float_row` and `run._refresh_cycle_actions`):
   `park_mode` by majority vote, everything else by mean. A float with
   fewer than 2 usable historical cycles gets `default_action()` instead
   (5-day bottom-park fallback) rather than attempting to vote on
   insufficient data. `_refresh_cycle_actions` re-fetches `Rtraj.nc` fresh
   (`force_refresh=True`) every run before re-voting, since GDAC's Rtraj.nc
   grows in place as a float completes new real cycles -- a cached copy
   would otherwise freeze the estimate at whatever was known the first time
   the float was ever seen. A failed refresh keeps the float's *previous*
   `cycle_action` rather than falling back to `default_action()` -- that
   fallback only makes sense at registration, where there's no previous
   estimate to protect. Every run's derived action (whether it changed or
   not) is appended to `cycle_action_history.parquet`
   (`float_store.save_cycle_action_history`) so how the estimate evolves
   over time can be inspected later; a change from the previous run also
   gets a `logger.info` line.

10. **No EKF, no covariance, no process noise, no bias state, anywhere in
    this codebase.** `simulate_cycle` is deterministic position-only
    integration. If you find yourself wanting to add uncertainty
    propagation here, that almost certainly belongs in the Argo+ piloting
    project instead, not this one.

## Gotchas / things to double-check, not yet independently verified

- **`ControlAction.cycle_hours` is descent + parking time ONLY** -- ascent
  and surface transmission time are already subtracted out by
  `action_from_cycle` when it's derived from real cycle data. `simulate_cycle`
  reconstructs the *full* repeat period as
  `cycle_hours*3600 + ascent_s + transmission_s`. If you ever change how
  `cycle_hours` is computed, you must change that reconstruction too, or
  cycle phasing will silently drift.
- **`cycle_extractor.py`'s profiles-file suffix search had a likely typo**
  in the original notebook code (`'_loologlal'` instead of a real GDAC
  suffix) -- fixed here to `('_profiles.nc', '_prof.nc')`. Flag to the
  user if this wasn't the intent.
- **Two unrelated `ControlAction` classes exist across this codebase**:
  this project's (`simulate.py`: `cycle_hours`/`target_depth`/`park_mode`)
  and the Argo+ piloting project's (`sim_types.py`: `duration_hours`/
  `parking_depth`). They are intentionally not unified. Do not import one
  where the other is expected.
- **`_reconcile_with_argo` detects a new surfacing via exact tuple equality**
  on `(lat, lon)` against `last_real_position`. This matches the original
  design but is fragile if Argo position quantization isn't perfectly
  consistent pull-to-pull; consider keying off `real_time` instead if this
  ever produces missed/duplicate detections.
- **`cycles[-1]["end_time"]` is a numpy-datetime64-derived string**
  (`str(valid_t[-1])` inside `extract_cycles`). `run.py`'s
  `_build_new_float_row` parses it with `datetime.fromisoformat`, which can
  be picky about fractional-second digit counts depending on Python
  version. Not yet verified against real data.

## What's implemented vs. stubbed

Fully implemented (no `NotImplementedError`):

- `float_store.py`: `FloatRow`, `ModelTrack`, `FloatRow.is_overdue`
- `simulate.py`: `ControlAction`, `latlon_to_xy`, `xy_to_latlon`,
  `simulate_cycle` (the full repeating phase-machine integrator)
- `cycle_extractor.py`: everything -- `extract_cycles`, `action_from_cycle`,
  `build_actions`, `mode_vote_action`, `default_action`
- `run.py`: everything in `_extend_trajectories`, `_reconcile_with_argo`,
  `_register_new_floats`, `_build_new_float_row` -- i.e. all the
  orchestration logic itself

Stubbed (`raise NotImplementedError`) -- these are the genuine remaining
work, and every one depends on a real interface this session never saw:

| Function | File | Needs |
|---|---|---|
| `download_model_data` | `data_handler.py` | Your `DataGetter`/`ChunkStrategy` copernicusmarine wrapper (CMEMS); DMI DKSS EDR API (FCOO) |
| `trim_to_forecast_only` | `data_handler.py` | Knowing the actual time-coord name/dtype in the returned `xr.Dataset` |
| `download_argo_floats_in_domain` | `data_handler.py` | argopy, matching the GDAC/argopy dual-format handling already in `cycle_extractor._build_profile_pos` |
| `download_float_history` | `data_handler.py` | Same argopy/GDAC fetch, for a single float's full `.traj` + `profiles.nc` |
| `load_floats_db` / `save_floats_db` | `float_store.py` | Flatten/reconstruct `FloatRow`↔parquet |
| `load_error_db` / `save_error_db` | `float_store.py` | Trivial parquet read/write once the above pattern is set |
| `build_interpolators` | `simulate.py` | The actual variable names / grid shape of `model_data`; likely `scipy.interpolate.RegularGridInterpolator` over (time, depth, lat, lon) if CMEMS/GETM grids are regular |
| `_is_empty`, `_last_timestamp`, `_lookup` | `run.py` | Same `model_data`/trajectory shape knowledge as above |
| `_bathy_interp` | `run.py` | Whatever bathymetry interpolator already exists elsewhere in the user's environment |

`display.py` does not exist yet. `run.py` has the call site commented out
(`# display.show(error_db)`) -- it's the natural next module once the
stubs above are filled in and there's real `error_db` data to look at.

## Conventions

- Relative imports throughout (`from .simulate import ControlAction`) --
  `src/` is a package (`__init__.py` present).
- Dataclasses for all structured data, not dicts, except where the original
  notebook code (`extract_cycles`) already returns plain dicts -- left as
  is rather than refactored, since that logic was copied with minimal
  changes from working code.
- Every policy constant (thresholds, region bounds) lives in `run.py`, not
  buried in the modules it calls.
