"""
Render current magnitude and direction frames from the FCOO/GETM dk_nested
velocities dataset. Reads directly via xarray -- either a local NetCDF file
or an OPeNDAP URL -- so no manual ASCII download/parsing is needed.

One PNG per (timestamp, depth), showing magnitude and direction side by
side. 'idk' (the finer nested inner-Danish-waters grid) is drawn on top of
'dk' (the coarser outer grid) on the same axes, filling in the hole 'dk'
leaves at that location, instead of producing a separate image per grid.

Assumes both 'dk' and 'idk' grids are present with variables:
    time, zax_dk, latc_dk, lonc_dk, uu_dk, vv_dk
    zax_idk, latc_idk, lonc_idk, uu_idk, vv_idk
(No uu_vv variable required -- magnitude is computed as sqrt(uu^2 + vv^2).)
'idk' has fewer depth levels than 'dk' (6 vs 10); depths beyond idk's range
are rendered as 'dk' only.

Usage:
    python fcoo_model_viewer.py \
        --input https://data.fcoo.dk/webmap/v2/data/FCOO/GETM/dk_nested.velocities.Z3D_2026070700.nc \
        --outdir frames --max-times 5

    python fcoo_model_viewer.py --input dk_subset.nc --outdir frames
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def make_frames(ds: xr.Dataset, outdir: Path, max_times: int | None = None):
    outdir.mkdir(parents=True, exist_ok=True)

    if "uu_dk" not in ds or "vv_dk" not in ds:
        print("No dk grid data (uu_dk/vv_dk) in dataset -- nothing to render")
        return

    n_times = ds.sizes["time"] if max_times is None else min(max_times, ds.sizes["time"])
    times = ds["time"].values

    uu_dk = ds["uu_dk"].isel(time=slice(0, n_times)).load().values
    vv_dk = ds["vv_dk"].isel(time=slice(0, n_times)).load().values
    lat_dk = ds["latc_dk"].values
    lon_dk = ds["lonc_dk"].values
    depths_dk = ds["zax_dk"].values
    speed_dk = np.sqrt(uu_dk ** 2 + vv_dk ** 2)

    have_idk = "uu_idk" in ds and "vv_idk" in ds
    if have_idk:
        uu_idk = ds["uu_idk"].isel(time=slice(0, n_times)).load().values
        vv_idk = ds["vv_idk"].isel(time=slice(0, n_times)).load().values
        lat_idk = ds["latc_idk"].values
        lon_idk = ds["lonc_idk"].values
        depths_idk = ds["zax_idk"].values
        speed_idk = np.sqrt(uu_idk ** 2 + vv_idk ** 2)
        vmax = np.nanmax([np.nanmax(speed_dk), np.nanmax(speed_idk)])
    else:
        print("No idk grid data (uu_idk/vv_idk) in dataset -- rendering dk grid only")
        vmax = np.nanmax(speed_dk)

    for t_idx in range(n_times):
        for z_idx, depth in enumerate(depths_dk):
            speed = speed_dk[t_idx, z_idx]
            direction = np.degrees(np.arctan2(uu_dk[t_idx, z_idx], vv_dk[t_idx, z_idx])) % 360

            fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

            mesh0 = axes[0].pcolormesh(
                lon_dk, lat_dk, speed, shading="auto", cmap="viridis", vmin=0, vmax=vmax
            )
            mesh1 = axes[1].pcolormesh(
                lon_dk, lat_dk, direction, shading="auto", cmap="hsv", vmin=0, vmax=360
            )

            if have_idk:
                zi_matches = np.nonzero(np.isclose(depths_idk, depth))[0]
                if len(zi_matches):
                    zi = zi_matches[0]
                    speed_i = speed_idk[t_idx, zi]
                    direction_i = np.degrees(
                        np.arctan2(uu_idk[t_idx, zi], vv_idk[t_idx, zi])
                    ) % 360
                    axes[0].pcolormesh(
                        lon_idk, lat_idk, speed_i, shading="auto", cmap="viridis", vmin=0, vmax=vmax
                    )
                    axes[1].pcolormesh(
                        lon_idk, lat_idk, direction_i, shading="auto", cmap="hsv", vmin=0, vmax=360
                    )

            axes[0].set_title("Magnitude (knots)")
            axes[0].set_xlabel("Longitude")
            axes[0].set_ylabel("Latitude")
            fig.colorbar(mesh0, ax=axes[0], shrink=0.85)

            axes[1].set_title("Direction (deg, 0=N)")
            axes[1].set_xlabel("Longitude")
            fig.colorbar(mesh1, ax=axes[1], shrink=0.85)

            t_val = np.datetime_as_string(times[t_idx], unit="h")
            fig.suptitle(
                f"time={t_val}  depth={depth:.0f} m  "
                f"extent=lat[{lat_dk.min():.2f},{lat_dk.max():.2f}] "
                f"lon[{lon_dk.min():.2f},{lon_dk.max():.2f}]"
            )
            fig.tight_layout()

            fname = outdir / f"t{t_idx:03d}_z{z_idx:02d}_{depth:.0f}m.png"
            fig.savefig(fname, dpi=130)
            plt.close(fig)
            print(f"Saved {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Local .nc path or OPeNDAP URL")
    parser.add_argument("--outdir", default="frames", help="Directory to write PNG frames to")
    parser.add_argument(
        "--max-times",
        type=int,
        default=None,
        help="Only render the first N timesteps (default: all -- can be a large transfer for remote URLs)",
    )
    args = parser.parse_args()

    ds = xr.open_dataset(args.input)
    make_frames(ds, Path(args.outdir), max_times=args.max_times)


if __name__ == "__main__":
    main()
