#!/usr/bin/env python3
"""Spatial visualisation for thetaFlow 2-D simulations.

Produces:
* An MP4 animation of spatially distributed soil moisture with a hillshade
  underlay from the DEM.
* An MP4 animation of overland-flow vectors (quiver arrows) overlaid on the
  soil moisture map.
* A combined side-by-side MP4.
* Optional static PNG snapshots at user-specified intervals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for MP4 export
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter


# ---------------------------------------------------------------------------
# Hillshade helper
# ---------------------------------------------------------------------------

def _hillshade(
    dem: np.ndarray,
    cell_size_m: float,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
) -> np.ndarray:
    """Compute a simple hillshade array (0–1) from a DEM."""
    az = np.deg2rad(360.0 - azimuth_deg + 90.0)
    alt = np.deg2rad(altitude_deg)
    dz_dx = np.gradient(dem, cell_size_m, axis=1)
    dz_dy = np.gradient(dem, cell_size_m, axis=0)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    hs = (
        np.cos(alt) * np.cos(slope_rad)
        + np.sin(alt) * np.sin(slope_rad) * np.cos(az - aspect_rad)
    )
    hs = np.clip(hs, 0.0, 1.0)
    return hs


# ---------------------------------------------------------------------------
# Main animation function
# ---------------------------------------------------------------------------

def animate_spatial_results(
    output_dir: Path,
    dem: np.ndarray,
    cell_size_m: float,
    n_steps: int,
    fps: int = 4,
    quiver_stride: int = 2,
    theta_vmin: float = 0.05,
    theta_vmax: float = 0.50,
    snapshot_interval: int = 0,
) -> None:
    """Create MP4 animations from per-step .npy files saved by the simulation.

    Expected files in *output_dir*::

        theta_mean_{step:05d}.npy   – (nrows, ncols) mean soil moisture
        flow_depth_{step:05d}.npy  – (nrows, ncols) overland flow depth (m)
        flow_vx_{step:05d}.npy     – (nrows, ncols) x-component of flow vector
        flow_vy_{step:05d}.npy     – (nrows, ncols) y-component of flow vector

    Outputs
    -------
    soil_moisture.mp4   – soil moisture animation with hillshade underlay
    overland_flow.mp4   – flow-depth + quiver animation
    combined.mp4        – side-by-side of the above two panels
    """
    output_dir = Path(output_dir)

    hs = _hillshade(dem, cell_size_m)
    nrows, ncols = dem.shape
    extent = [0, ncols * cell_size_m, nrows * cell_size_m, 0]

    # -----------------------------------------------------------------------
    # Helper: load one frame
    # -----------------------------------------------------------------------
    def _load(prefix: str, step: int) -> Optional[np.ndarray]:
        p = output_dir / f"{prefix}_{step:05d}.npy"
        if p.exists():
            return np.load(str(p))
        return None

    # -----------------------------------------------------------------------
    # 1.  Soil moisture animation
    # -----------------------------------------------------------------------
    fig_sm, ax_sm = plt.subplots(figsize=(7, 6), tight_layout=True)
    ax_sm.imshow(hs, cmap="gray", extent=extent, vmin=0, vmax=1, origin="upper")
    theta0 = _load("theta_mean", 0)
    if theta0 is None:
        theta0 = np.full((nrows, ncols), float("nan"))
    im_sm = ax_sm.imshow(
        theta0,
        cmap="YlGnBu",
        vmin=theta_vmin,
        vmax=theta_vmax,
        alpha=0.75,
        extent=extent,
        origin="upper",
    )
    cb_sm = fig_sm.colorbar(im_sm, ax=ax_sm, label="Mean θ (−)")
    title_sm = ax_sm.set_title("Soil Moisture  |  Step 0")
    ax_sm.set_xlabel("Easting (m)")
    ax_sm.set_ylabel("Northing (m)")

    def _update_sm(step: int) -> list:
        arr = _load("theta_mean", step)
        if arr is not None:
            im_sm.set_data(arr)
        title_sm.set_text(f"Soil Moisture  |  Step {step:d}")
        return [im_sm, title_sm]

    ani_sm = FuncAnimation(
        fig_sm, _update_sm, frames=range(n_steps), blit=True, interval=1000 // max(fps, 1)
    )
    _save_mp4(ani_sm, output_dir / "soil_moisture.mp4", fps)
    plt.close(fig_sm)
    print(f"Saved {output_dir / 'soil_moisture.mp4'}")

    # -----------------------------------------------------------------------
    # 2.  Overland-flow animation (flow depth + quiver)
    # -----------------------------------------------------------------------
    fig_fl, ax_fl = plt.subplots(figsize=(7, 6), tight_layout=True)
    ax_fl.imshow(hs, cmap="gray", extent=extent, vmin=0, vmax=1, origin="upper")
    depth0 = _load("flow_depth", 0)
    if depth0 is None:
        depth0 = np.zeros((nrows, ncols))
    im_fl = ax_fl.imshow(
        depth0,
        cmap="Blues",
        vmin=0.0,
        vmax=0.05,
        alpha=0.75,
        extent=extent,
        origin="upper",
    )
    cb_fl = fig_fl.colorbar(im_fl, ax=ax_fl, label="Flow depth (m)")

    # Initial quiver
    stride = max(1, quiver_stride)
    Y, X = np.mgrid[0:nrows:stride, 0:ncols:stride]
    vx0 = _load("flow_vx", 0)
    vy0 = _load("flow_vy", 0)
    if vx0 is None:
        vx0 = np.zeros_like(depth0)
    if vy0 is None:
        vy0 = np.zeros_like(depth0)
    U = vx0[::stride, ::stride]
    V = vy0[::stride, ::stride]
    qv = ax_fl.quiver(
        X * cell_size_m, Y * cell_size_m, U, V,
        scale=1, scale_units="xy", angles="xy",
        color="darkblue", alpha=0.7, width=0.003,
    )
    title_fl = ax_fl.set_title("Overland Flow  |  Step 0")
    ax_fl.set_xlabel("Easting (m)")
    ax_fl.set_ylabel("Northing (m)")

    def _update_fl(step: int) -> list:
        depth = _load("flow_depth", step)
        if depth is not None:
            im_fl.set_data(depth)
        vx = _load("flow_vx", step)
        vy = _load("flow_vy", step)
        if vx is not None and vy is not None:
            qv.set_UVC(vx[::stride, ::stride], vy[::stride, ::stride])
        title_fl.set_text(f"Overland Flow  |  Step {step:d}")
        return [im_fl, qv, title_fl]

    ani_fl = FuncAnimation(
        fig_fl, _update_fl, frames=range(n_steps), blit=False, interval=1000 // max(fps, 1)
    )
    _save_mp4(ani_fl, output_dir / "overland_flow.mp4", fps)
    plt.close(fig_fl)
    print(f"Saved {output_dir / 'overland_flow.mp4'}")

    # -----------------------------------------------------------------------
    # 3.  Combined side-by-side animation
    # -----------------------------------------------------------------------
    fig_co, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)
    for ax in (ax_l, ax_r):
        ax.imshow(hs, cmap="gray", extent=extent, vmin=0, vmax=1, origin="upper")

    im_l = ax_l.imshow(
        theta0, cmap="YlGnBu", vmin=theta_vmin, vmax=theta_vmax,
        alpha=0.75, extent=extent, origin="upper",
    )
    fig_co.colorbar(im_l, ax=ax_l, label="Mean θ (−)")
    ax_l.set_title("Soil Moisture")
    ax_l.set_xlabel("Easting (m)")
    ax_l.set_ylabel("Northing (m)")

    im_r = ax_r.imshow(
        depth0, cmap="Blues", vmin=0.0, vmax=0.05,
        alpha=0.75, extent=extent, origin="upper",
    )
    fig_co.colorbar(im_r, ax=ax_r, label="Flow depth (m)")
    qv2 = ax_r.quiver(
        X * cell_size_m, Y * cell_size_m, U, V,
        scale=1, scale_units="xy", angles="xy",
        color="darkblue", alpha=0.7, width=0.003,
    )
    ax_r.set_title("Overland Flow")
    ax_r.set_xlabel("Easting (m)")
    title_co = fig_co.suptitle("Step 0", fontsize=12)

    def _update_co(step: int) -> list:
        theta = _load("theta_mean", step)
        if theta is not None:
            im_l.set_data(theta)
        depth = _load("flow_depth", step)
        if depth is not None:
            im_r.set_data(depth)
        vx = _load("flow_vx", step)
        vy = _load("flow_vy", step)
        if vx is not None and vy is not None:
            qv2.set_UVC(vx[::stride, ::stride], vy[::stride, ::stride])
        title_co.set_text(f"Step {step:d}")
        return [im_l, im_r, qv2, title_co]

    ani_co = FuncAnimation(
        fig_co, _update_co, frames=range(n_steps), blit=False, interval=1000 // max(fps, 1)
    )
    _save_mp4(ani_co, output_dir / "combined.mp4", fps)
    plt.close(fig_co)
    print(f"Saved {output_dir / 'combined.mp4'}")

    # -----------------------------------------------------------------------
    # 4.  Static snapshots
    # -----------------------------------------------------------------------
    if snapshot_interval and snapshot_interval > 0:
        snap_dir = output_dir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        for step in range(0, n_steps, snapshot_interval):
            _save_snapshot(step, hs, dem, cell_size_m, output_dir, snap_dir,
                           theta_vmin, theta_vmax, quiver_stride)
        print(f"Saved static snapshots to {snap_dir}")


def _save_mp4(animation: FuncAnimation, path: Path, fps: int) -> None:
    """Save a FuncAnimation as an MP4 using FFMpegWriter."""
    try:
        writer = FFMpegWriter(fps=max(fps, 1), metadata={"title": path.stem}, bitrate=1800)
        animation.save(str(path), writer=writer)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not save {path} as MP4 ({exc}). "
              "Ensure ffmpeg is installed (e.g. conda install ffmpeg or apt-get install ffmpeg).")


def _save_snapshot(
    step: int,
    hs: np.ndarray,
    dem: np.ndarray,
    cell_size_m: float,
    data_dir: Path,
    snap_dir: Path,
    theta_vmin: float,
    theta_vmax: float,
    stride: int,
) -> None:
    nrows, ncols = dem.shape
    extent = [0, ncols * cell_size_m, nrows * cell_size_m, 0]

    def _load(prefix: str) -> Optional[np.ndarray]:
        p = data_dir / f"{prefix}_{step:05d}.npy"
        return np.load(str(p)) if p.exists() else None

    theta = _load("theta_mean")
    depth = _load("flow_depth")
    vx = _load("flow_vx")
    vy = _load("flow_vy")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)
    for ax in (ax_l, ax_r):
        ax.imshow(hs, cmap="gray", extent=extent, vmin=0, vmax=1, origin="upper")

    if theta is not None:
        im = ax_l.imshow(theta, cmap="YlGnBu", vmin=theta_vmin, vmax=theta_vmax,
                         alpha=0.75, extent=extent, origin="upper")
        fig.colorbar(im, ax=ax_l, label="Mean θ (−)")
    ax_l.set_title(f"Soil Moisture – Step {step:d}")
    ax_l.set_xlabel("Easting (m)")
    ax_l.set_ylabel("Northing (m)")

    if depth is not None:
        im2 = ax_r.imshow(depth, cmap="Blues", vmin=0.0, vmax=0.05,
                          alpha=0.75, extent=extent, origin="upper")
        fig.colorbar(im2, ax=ax_r, label="Flow depth (m)")
    if vx is not None and vy is not None:
        Y, X = np.mgrid[0:nrows:stride, 0:ncols:stride]
        ax_r.quiver(X * cell_size_m, Y * cell_size_m,
                    vx[::stride, ::stride], vy[::stride, ::stride],
                    scale=1, scale_units="xy", angles="xy",
                    color="darkblue", alpha=0.7, width=0.003)
    ax_r.set_title(f"Overland Flow – Step {step:d}")
    ax_r.set_xlabel("Easting (m)")

    out_path = snap_dir / f"snapshot_{step:05d}.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
