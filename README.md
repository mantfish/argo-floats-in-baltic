# CMEMS / FCOO Argo Leaderboard

An operational leaderboard that scores two ocean current forecast models —
**CMEMS** (Copernicus Marine Baltic product) and **FCOO/GETM** (Danish
Defence Centre for Operational Oceanography) — against where Argo profiling
floats actually surface, in Danish/Baltic waters. Every tracked float gets
one simulated trajectory per model; when the float genuinely surfaces, both
models' predicted positions at that timestamp are compared against the real
one, and the error is recorded. The results are published as a static map +
leaderboard site from `docs/`.

This is **not** the EKF/MPC float-piloting project. That project estimates
state and controls a float in real time, with covariance, bias, and process
noise. This one runs a simpler, deterministic forward simulation purely to
evaluate forecast skill retrospectively. The two projects are deliberately
decoupled — this one talks to the piloting server only over its public HTTP
API (see `scripts/notify_ekf_server.py`), never by importing its code.

## How it works

```
main.py -> src/run.py
             ├─ data_handler.py     fetch: model currents, Argo population, float history
             ├─ float_store.py      load/save: FloatRow / ModelTrack, parquet on disk
             ├─ cycle_extractor.py  compute: .traj history -> ControlAction (dive profile)
             └─ simulate.py         compute: forward-integrate a trajectory (RK4)
```

Each pipeline run does four things, in order:

1. **Extend trajectories.** Pull the latest forecast for each model, and for
   every tracked float advance its simulated position forward using that
   model's currents and the float's own dive profile (descent → park →
   ascent → surface transmission, repeating). Only genuinely new forecast
   data is used — a row is frozen in place if nothing new has arrived since
   its last extension, so the metric measures *forecast* skill, not a mix of
   forecast and after-the-fact analysis skill.
2. **Reconcile against real Argo data.** The instant a float's *real*
   surfacing is confirmed in the Argo feed, both models' predicted positions
   at that exact timestamp are looked up and scored against it. The
   simulated trajectory is then reset to a single point at the real
   position — the anchor for the next leg.
3. **Register new floats** seen in the Argo feed for the first time, seeding
   their dive profile from their own real cycle history (or falling back to
   a default profile if none exists yet).
4. **Refresh each float's dive-profile estimate** (`ControlAction`) from its
   most recent real cycles, and **persist + export** everything to parquet
   and to the JSON the map/leaderboard pages read.

Every policy constant (thresholds, region bounds, grace periods) lives in
`run.py`; the modules it calls are otherwise policy-free, so a threshold
never needs to be found by hunting through unrelated modules.

## Data model

- **`FloatRow`** — one per float (WMO id): its dive-profile estimate
  (`ControlAction`), last confirmed real position, per-model simulated
  tracks (`ModelTrack`), and bookkeeping (missed-pull counters, dead flag).
- **`ModelTrack`** — one model's *current, in-progress* simulated leg for a
  float: a list of `(t, lat, lon, depth)` points from the last real
  surfacing to now. Reset to a single point every time a new real surfacing
  arrives — it is **not** the float's full lifetime history (see below).
- **`ControlAction`** — a float's representative dive-cycle parameters
  (park mode, cycle length, target depth, descent/ascent speed), mode-voted
  across the float's most recent real cycles so a mid-mission reprogram
  doesn't get diluted by stale history.

Storage is seven parquet tables under `data/store/` — scalar float metadata,
long-format trajectories, real surfacing history, scoring errors, forecast
history, dive-profile-estimate history, and completed-leg path history (see
below). All are missing-file- and missing-column-tolerant on load, so
schema changes don't break loading older data.

**A float's full simulated history over time is split across two tables**,
not held in one place: `trajectories.parquet` only ever holds the *current*
leg (reset to a single anchor point at every real surfacing); the full
simulated path of every *completed* leg is captured into
`leg_history.parquet` in the instant before that reset would otherwise
discard it. To reconstruct one float's complete simulated path over all
time, concatenate `leg_history.parquet` (past legs) with whatever's
currently in `trajectories.parquet` (the in-progress leg) — this is what
`web_export.py` and `notebooks/prediction_diagnostics.ipynb` already do for
display and diagnostics.

## Running it

```bash
uv sync                 # installs deps from pyproject.toml (Python >= 3.11)
uv run main.py           # runs one full pipeline cycle against config.toml
```

`config.toml` holds region bounds, thresholds (dead-float cutoff, overdue
days, next-surfacing grace period, how many recent cycles to vote a dive
profile over), and storage/cache paths. A CMEMS account is required
(`COPERNICUSMARINE_SERVICE_USERNAME`/`_PASSWORD` env vars); FCOO and Argo
GDAC access need no credentials.

For frontend development without running the live pipeline (no network
access, no CMEMS account needed), generate synthetic data instead:

```bash
uv run scripts/fixture.py    # writes synthetic docs/data/*.json
```

In production this runs automatically twice a day via
`.github/workflows/run-pipeline.yml` (after each FCOO 00Z/12Z forecast
publishes), which commits the updated `data/store/` and `docs/data/` back to
the repo — that's what the recurring "Update leaderboard" commits are.
`D6_2024.nc` (EMODnet bathymetry, used for seabed-taper current queries) is
tracked via git-lfs and pulled/cached separately since it's ~255 MB. After
each run, `scripts/notify_ekf_server.py` tells the separate EKF/MPC piloting
project about any float that genuinely surfaced this cycle.

## Recent changes (2026-08-11)

A deep-dive investigation found that a large fraction of the leaderboard's
"forecast errors" weren't genuine forecast misses at all — the simulator was
silently defaulting to "the float didn't move" for a majority of scored
legs, which then got compared against real drift and counted as
near-total failure. The following fixes address the root causes; **the
leaderboard's historical error stats predate these fixes and are not
directly comparable to results going forward.**

- **Fixed silent zero-current bugs in `simulate.py`.** Two distinct issues
  meant a masked/out-of-range depth query silently returned exactly `0.0`
  current instead of a real value: (1) queries shallower than a model
  grid's shallowest real level (e.g. the very start of every descent) fell
  outside `RegularGridInterpolator`'s valid range and hard-zeroed; (2) a
  float's parking-depth query landing on a masked/land grid cell (common,
  since park-on-bottom floats by definition sit at the deepest, least-well-
  resolved point locally) silently zeroed for that entire cycle. Since
  park-on-bottom floats only advect during their brief non-parked window
  each cycle, one bad sample there froze the *entire* multi-day leg at the
  anchor position — confirmed empirically as the dominant cause of the
  leaderboard's near-100%-of-drift error events. Fixed via
  `_clamp_shallow` (hold at the shallowest real level instead of
  zero-filling) and `_fill_masked_nearest` (nearest-real-neighbor fill for
  masked cells, instead of zero-fill).
- **Switched integration from forward Euler to classic 4th-order
  Runge-Kutta**, with each of the four per-step evaluations independently
  re-deriving phase/depth so a step straddling a phase boundary (e.g.
  descent → parking) blends correctly instead of committing the whole step
  to whichever phase was active at the step's start.
- **Adaptive step size**: fine (60 s) while descending, ascending, or
  communicating at the surface — where depth and current can change fast —
  and much coarser (up to 3600 s, clamped to never overshoot the next phase
  boundary) while at constant parking depth, where fine time resolution
  isn't needed. Cuts the number of current queries substantially without
  losing accuracy where it matters.
- **`park_on_bottom` classification now uses Argo's own `GROUNDED` QC flag**
  (Rtraj.nc, reference table 20 — the float's own onboard/DAC-derived
  "did it touch the seabed this cycle" determination) instead of an inferred
  bathymetry-ratio heuristic. The heuristic (`DEPTH_FRAC_THRESH`) has been
  removed entirely, along with the `bathy_interp` parameter it needed in
  `cycle_extractor.extract_cycles`. Classification requires **unanimity**
  across a float's recent cycles, not a majority vote: since park_on_bottom
  vs. parking_depth is a qualitative fork in the simulator (one skips
  advection entirely during parking, the other doesn't), a single
  non-grounded cycle in the recent window is treated as genuine uncertainty
  rather than resolved by outvoting it. The shadow-track mechanism
  (`SHADOW_MODELS`, `force_drift=True` — see `run.py`) that A/B-tests this
  assumption against real surfacings runs unconditionally regardless of how
  this vote lands, so it isn't affected by the tightened rule.
- **Added FCOO's dedicated near-surface product** (`nsbalt-1nm_velocities_
  surface_*.nc`, fetched independently of the depth-resolved `dk`/`idk`
  grids) as a third resolution tier in `simulate._build_fcoo_interpolators`,
  preferred for any query shallower than 5 m (`dk`/`idk`'s own shallowest
  real level). Real near-surface current data instead of holding the
  current constant at the 5 m value for that whole gap.
- **Investigated switching FCOO's fetch from chunked OPeNDAP (pydap) to a
  single-shot HTTP `.nc` download** (as used successfully for CMEMS, and as
  a user had had some success with via `wget`) to see if it would reduce
  the intermittent silent-corruption failures. Tested directly against the
  live server: the `.nc` endpoint does **not** honor OPeNDAP constraint
  expressions at all (variable selection and lat/lon-box array slicing are
  both silently ignored — it always serves the complete, full-resolution,
  all-variables file), and every single-shot attempt (5/5 trials, various
  variable subsets) silently truncated partway through in the 13-26s range
  regardless of payload size, which is undetectable without an explicit
  `Content-Length` check since HDF5's chunked storage zero-fills whatever
  never arrived. **Not adopted** — the existing many-small-chunked-requests
  design is actually the correct response to this server's behavior, not a
  suboptimal legacy choice.
- **Reset all trajectory data.** `data/store/trajectories.parquet` and
  `data/store/leg_history.parquet` — the actual simulated path data, which
  predates all of the above fixes and was largely frozen/incorrect — have
  been cleared (each float's current leg reset to just its real anchor
  point; a backup of the pre-reset files lives in
  `data/store/backup_2026-08-11_pre_rk4_masked_cell_fix/`, and git history
  preserves the exact prior state regardless). **Scoring history
  (`errors.parquet`) and the leaderboard's aggregate stats were
  deliberately left untouched** as the historical record of what the
  pre-fix simulation actually produced. Each float's `cycle_action`
  (dive-profile estimate, including `park_mode`) was refreshed under the
  new classification logic as part of this reset.

## Going deeper

- **`CLAUDE.md`** — the full architectural reference: every design decision
  and the specific real-data incident that motivated it, module
  responsibilities, storage schema, and known gotchas not yet independently
  verified. Written for whoever (human or AI) next needs to change this
  code without re-litigating settled decisions.
- **`docs/pipeline.html`** — the public-facing "how this works" page linked
  from the site itself; a lighter-weight duplicate of the architecture
  section above, kept in sync by hand whenever a module's behavior changes.
- **`notebooks/prediction_diagnostics.ipynb`** — real surfacings vs. model
  predictions, timing accuracy, depth-vs-time reconstruction, and the
  park-on-bottom vs. drift-through-parking shadow-track comparison.
