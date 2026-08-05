#!/usr/bin/env python3
"""Richards-equation soil column simulation for a 1D vertical soil column."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class VegetationType:
    """Physical characteristics of a vegetation type that affect the soil water balance."""

    name: str
    rooting_depth_m: float  # depth to which roots actively extract water
    # Fraction of PET that this vegetation type can achieve at field conditions (0–1).
    # Used to scale PET to actual plant water demand relative to a reference crop.
    pet_scale: float = 1.0
    # Optional short description
    description: str = ""


# ---------------------------------------------------------------------------
# Built-in vegetation library
# ---------------------------------------------------------------------------
VEGETATION_LIBRARY: dict[str, VegetationType] = {
    "grass": VegetationType(
        name="grass",
        rooting_depth_m=0.30,
        pet_scale=1.0,
        description="Short managed or natural grass sward.",
    ),
    "broadleaf_woodland": VegetationType(
        name="broadleaf_woodland",
        rooting_depth_m=1.50,
        pet_scale=1.1,
        description="Deciduous broadleaf trees (oak, ash, beech, etc.).",
    ),
    "coniferous_woodland": VegetationType(
        name="coniferous_woodland",
        rooting_depth_m=1.20,
        pet_scale=1.05,
        description="Evergreen coniferous plantation or native forest.",
    ),
    "moorland": VegetationType(
        name="moorland",
        rooting_depth_m=0.20,
        pet_scale=0.7,
        description="Upland heath and blanket bog dominated by heather and sedge.",
    ),
    "wheat": VegetationType(
        name="wheat",
        rooting_depth_m=1.00,
        pet_scale=1.0,
        description="Winter or spring wheat cereal crop.",
    ),
    "maize": VegetationType(
        name="maize",
        rooting_depth_m=1.20,
        pet_scale=1.15,
        description="Silage or grain maize.",
    ),
    "oilseed_rape": VegetationType(
        name="oilseed_rape",
        rooting_depth_m=0.90,
        pet_scale=1.0,
        description="Winter oilseed rape (canola).",
    ),
    "sugar_beet": VegetationType(
        name="sugar_beet",
        rooting_depth_m=1.00,
        pet_scale=1.0,
        description="Sugar beet root crop.",
    ),
    "potato": VegetationType(
        name="potato",
        rooting_depth_m=0.60,
        pet_scale=1.05,
        description="Potato tuber crop.",
    ),
    "bare_soil": VegetationType(
        name="bare_soil",
        rooting_depth_m=0.0,
        pet_scale=0.3,
        description="No vegetation; evaporation from soil surface only.",
    ),
    "urban": VegetationType(
        name="urban",
        rooting_depth_m=0.10,
        pet_scale=0.2,
        description="Urban or suburban mix with mostly impervious surfaces.",
    ),
}


@dataclass
class SoilProperties:
    theta_r: float
    theta_s: float
    alpha_per_m: float
    n: float
    ks_m_per_s: float
    pore_connectivity: float = 0.5

    @property
    def m(self) -> float:
        return 1.0 - 1.0 / self.n


@dataclass
class ColumnConfig:
    nz: int
    dz_m: float


@dataclass
class SimulationConfig:
    initial_head_m: float
    default_dt_hours: float
    max_substep_seconds: float
    min_theta_buffer: float
    max_dtheta_per_substep: float = 0.002
    min_substep_seconds: float = 1.0
    green_ampt_wetting_front_suction_m: float = 0.11
    slope_angle_deg: float = 6.0
    hillslope_flow_path_m: float = 1.0


@dataclass
class SimulationState:
    head_m: np.ndarray
    theta: np.ndarray


def effective_saturation_from_head(head_m: np.ndarray, soil: SoilProperties) -> np.ndarray:
    suction = np.maximum(-head_m, 0.0)
    se_unsat = (1.0 + (soil.alpha_per_m * suction) ** soil.n) ** (-soil.m)
    return np.where(head_m >= 0.0, 1.0, se_unsat)


def theta_from_head(head_m: np.ndarray, soil: SoilProperties) -> np.ndarray:
    se = effective_saturation_from_head(head_m, soil)
    return soil.theta_r + se * (soil.theta_s - soil.theta_r)


def hydraulic_conductivity(head_m: np.ndarray, soil: SoilProperties) -> np.ndarray:
    se = effective_saturation_from_head(head_m, soil)
    se = np.clip(se, 1.0e-9, 1.0)
    term = 1.0 - (1.0 - se ** (1.0 / soil.m)) ** soil.m
    return soil.ks_m_per_s * se ** soil.pore_connectivity * term**2


def head_from_theta(theta: np.ndarray, soil: SoilProperties) -> np.ndarray:
    theta = np.clip(theta, soil.theta_r + 1.0e-12, soil.theta_s - 1.0e-12)
    se = (theta - soil.theta_r) / (soil.theta_s - soil.theta_r)
    suction = ((se ** (-1.0 / soil.m) - 1.0) ** (1.0 / soil.n)) / soil.alpha_per_m
    return np.where(se >= 0.999999, 0.0, -suction)


def read_config(config_path: Path) -> tuple[SoilProperties, ColumnConfig, SimulationConfig]:
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    soil = SoilProperties(**raw["soil"])
    column = ColumnConfig(**raw["column"])
    simulation = SimulationConfig(**raw["simulation"])
    return soil, column, simulation


def _process_forcing_df(forcing: pd.DataFrame, default_dt_hours: float, source_name: str = "<dataframe>") -> pd.DataFrame:
    """Add ``dt_seconds`` and ``timestamp`` columns to a forcing DataFrame in-place."""
    expected = {"rainfall_mm_h", "pet_mm_h"}
    missing = expected.difference(forcing.columns)
    if missing:
        raise ValueError(f"Forcing data is missing required columns: {sorted(missing)}")

    if "timestamp" in forcing.columns:
        forcing["timestamp"] = pd.to_datetime(forcing["timestamp"], utc=True)
        dt_seconds = (forcing["timestamp"].shift(-1) - forcing["timestamp"]).dt.total_seconds()
        fallback_dt_s = default_dt_hours * 3600.0
        positive_dt = dt_seconds[dt_seconds > 0]
        inferred_dt_s = float(positive_dt.median()) if not positive_dt.empty else fallback_dt_s
        invalid = dt_seconds <= 0
        invalid_count = int(invalid.fillna(False).sum())
        if invalid_count > 0:
            print(
                "Warning: found "
                f"{invalid_count} non-positive timestamp interval(s) in {source_name}; "
                f"replacing with {inferred_dt_s:.1f} s."
            )
        dt_seconds = dt_seconds.where(dt_seconds > 0, inferred_dt_s)
        forcing["dt_seconds"] = dt_seconds.fillna(inferred_dt_s)
    elif "dt_hours" in forcing.columns:
        forcing["dt_seconds"] = forcing["dt_hours"] * 3600.0
        forcing["timestamp"] = pd.NaT
    else:
        forcing["dt_seconds"] = default_dt_hours * 3600.0
        forcing["timestamp"] = pd.NaT

    if np.any(forcing["dt_seconds"] <= 0):
        raise ValueError("All forcing time steps must have positive duration.")

    return forcing


def read_forcing(forcing_csv: Path, default_dt_hours: float) -> pd.DataFrame:
    forcing = pd.read_csv(forcing_csv)
    return _process_forcing_df(forcing, default_dt_hours, source_name=forcing_csv.name)


def build_column_geometry(nz: int, dz_m: float) -> tuple[np.ndarray, np.ndarray]:
    column_nodes = np.arange(nz) * 2
    depth_m = np.arange(nz, dtype=float) * dz_m
    return column_nodes, depth_m


def _root_uptake_sink(
    theta: np.ndarray,
    soil: SoilProperties,
    depth_m: np.ndarray,
    dz_m: float,
    pet_flux: float,
    veg: Optional[VegetationType],
    min_theta_buffer: float,
) -> np.ndarray:
    """Return a per-layer root-water-uptake sink rate (m³/m³/s, positive = removal).

    When vegetation is provided, ET demand is distributed uniformly over the
    rooted layers proportional to available water.  When no vegetation is given
    the old surface-only behaviour is preserved (caller subtracts from the
    surface flux directly and this function returns zeros).
    """
    n = theta.size
    sink = np.zeros(n)

    if veg is None or veg.rooting_depth_m <= 0.0:
        return sink

    effective_pet = pet_flux * veg.pet_scale
    if effective_pet <= 0.0:
        return sink

    # Identify rooted layers (node centres within rooting depth)
    in_root_zone = depth_m <= veg.rooting_depth_m
    if not np.any(in_root_zone):
        return sink

    # Available water above wilting-point buffer in each layer (m/s equivalent)
    avail = np.maximum(theta - (soil.theta_r + min_theta_buffer), 0.0)
    avail_root = avail * in_root_zone
    total_avail = float(np.sum(avail_root) * dz_m)

    if total_avail <= 0.0:
        return sink

    # Distribute demand proportionally to available water in each rooted layer
    actual_et_flux = min(effective_pet, total_avail / 1.0)  # cap at available
    weights = avail_root / (np.sum(avail_root) + 1.0e-30)
    sink = weights * actual_et_flux / dz_m  # m³/m³/s per layer
    return sink


def one_substep(
    state: SimulationState,
    soil: SoilProperties,
    sim: SimulationConfig,
    cumulative_infiltration_m: float,
    dz_m: float,
    dt_s: float,
    rainfall_mm_h: float,
    pet_mm_h: float,
    min_theta_buffer: float,
    depth_m: Optional[np.ndarray] = None,
    vegetation: Optional[VegetationType] = None,
) -> tuple[SimulationState, dict[str, float], float]:
    theta = state.theta.copy()
    head_m = state.head_m.copy()

    rain_flux = rainfall_mm_h / 1000.0 / 3600.0
    pet_flux = max(pet_mm_h, 0.0) / 1000.0 / 3600.0

    n = theta.size

    if vegetation is not None and vegetation.rooting_depth_m > 0.0 and depth_m is not None:
        # Root-zone uptake: distributed across layers; no surface-only ET deduction
        aet_sink = _root_uptake_sink(
            theta, soil, depth_m, dz_m, pet_flux, vegetation, min_theta_buffer
        )
        # Total AET flux for diagnostics (m/s equivalent)
        aet_flux = float(np.sum(aet_sink) * dz_m)
        surface_et_flux = 0.0
    else:
        # Legacy surface-layer evaporation
        available_evap = max((theta[0] - (soil.theta_r + min_theta_buffer)) * dz_m / dt_s, 0.0)
        aet_flux = min(pet_flux, available_evap)
        surface_et_flux = aet_flux
        aet_sink = np.zeros(n)

    conductivity = hydraulic_conductivity(head_m, soil)
    theta_deficit = max(soil.theta_s - theta[0], 1.0e-6)
    cumulative_for_capacity = max(cumulative_infiltration_m, 1.0e-6)
    infiltration_capacity_flux = soil.ks_m_per_s * (
        1.0 + (sim.green_ampt_wetting_front_suction_m * theta_deficit) / cumulative_for_capacity
    )
    infiltration_capacity_flux = max(infiltration_capacity_flux, soil.ks_m_per_s)
    infiltration_flux = min(rain_flux, infiltration_capacity_flux)
    runoff_flux = max(rain_flux - infiltration_flux, 0.0)
    net_downward_flux = infiltration_flux - surface_et_flux
    cumulative_infiltration_m_new = cumulative_infiltration_m + infiltration_flux * dt_s

    q_upward = np.zeros(n + 1)

    q_upward[0] = -net_downward_flux

    for j in range(1, n):
        top = j - 1
        bottom = j
        k_int = 0.5 * (conductivity[top] + conductivity[bottom])
        dh_dz = (head_m[top] - head_m[bottom]) / dz_m
        q_upward[j] = -k_int * (dh_dz + 1.0)

    q_upward[n] = -conductivity[-1]

    vertical_dtheta_rate = (q_upward[1:] - q_upward[:-1]) / dz_m

    # Represent lateral throughflow for a hillslope block as gravity-driven flow along slope.
    slope_angle_rad = np.deg2rad(sim.slope_angle_deg)
    lateral_flux_layers = conductivity * np.sin(slope_angle_rad)
    lateral_sink_rate = lateral_flux_layers / max(sim.hillslope_flow_path_m, 1.0e-9)

    theta_new = theta + dt_s * (vertical_dtheta_rate - lateral_sink_rate - aet_sink)
    theta_new = np.clip(theta_new, soil.theta_r + 1.0e-8, soil.theta_s - 1.0e-8)
    head_new = head_from_theta(theta_new, soil)

    lateral_throughflow_flux = float(np.sum(lateral_flux_layers) * dz_m / max(sim.hillslope_flow_path_m, 1.0e-9))

    diagnostics = {
        "rain_flux_m_per_s": rain_flux,
        "pet_flux_m_per_s": pet_flux,
        "aet_flux_m_per_s": aet_flux,
        "infiltration_flux_m_per_s": infiltration_flux,
        "runoff_flux_m_per_s": runoff_flux,
        "infiltration_capacity_flux_m_per_s": infiltration_capacity_flux,
        "cumulative_infiltration_m": cumulative_infiltration_m_new,
        "surface_net_downward_flux_m_per_s": net_downward_flux,
        "bottom_downward_flux_m_per_s": -q_upward[n],
        "lateral_throughflow_flux_m_per_s": lateral_throughflow_flux,
    }

    return SimulationState(head_m=head_new, theta=theta_new), diagnostics, cumulative_infiltration_m_new


def estimate_stable_dt_seconds(
    state: SimulationState,
    soil: SoilProperties,
    sim: SimulationConfig,
    cumulative_infiltration_m: float,
    dz_m: float,
    rainfall_mm_h: float,
    pet_mm_h: float,
    min_theta_buffer: float,
    max_dtheta_per_substep: float,
    max_dt_seconds: float,
    min_dt_seconds: float,
    depth_m: Optional[np.ndarray] = None,
    vegetation: Optional[VegetationType] = None,
) -> float:
    theta = state.theta
    head_m = state.head_m

    rain_flux = rainfall_mm_h / 1000.0 / 3600.0
    pet_flux = max(pet_mm_h, 0.0) / 1000.0 / 3600.0

    if vegetation is not None and vegetation.rooting_depth_m > 0.0 and depth_m is not None:
        aet_sink = _root_uptake_sink(
            theta, soil, depth_m, dz_m, pet_flux, vegetation, min_theta_buffer
        )
        surface_et_flux = 0.0
    else:
        available_evap_rate = max(
            (theta[0] - (soil.theta_r + min_theta_buffer)) * dz_m / max_dt_seconds, 0.0
        )
        aet_flux_surface = min(pet_flux, available_evap_rate)
        surface_et_flux = aet_flux_surface
        aet_sink = np.zeros(theta.size)

    conductivity = hydraulic_conductivity(head_m, soil)
    theta_deficit = max(soil.theta_s - theta[0], 1.0e-6)
    cumulative_for_capacity = max(cumulative_infiltration_m, 1.0e-6)
    infiltration_capacity_flux = soil.ks_m_per_s * (
        1.0 + (sim.green_ampt_wetting_front_suction_m * theta_deficit) / cumulative_for_capacity
    )
    infiltration_capacity_flux = max(infiltration_capacity_flux, soil.ks_m_per_s)
    infiltration_flux = min(rain_flux, infiltration_capacity_flux)
    net_downward_flux = infiltration_flux - surface_et_flux

    n = theta.size
    q_upward = np.zeros(n + 1)
    q_upward[0] = -net_downward_flux

    for j in range(1, n):
        top = j - 1
        bottom = j
        k_int = 0.5 * (conductivity[top] + conductivity[bottom])
        dh_dz = (head_m[top] - head_m[bottom]) / dz_m
        q_upward[j] = -k_int * (dh_dz + 1.0)

    q_upward[n] = -conductivity[-1]

    vertical_dtheta_rate = (q_upward[1:] - q_upward[:-1]) / dz_m
    slope_angle_rad = np.deg2rad(sim.slope_angle_deg)
    lateral_flux_layers = conductivity * np.sin(slope_angle_rad)
    lateral_sink_rate = lateral_flux_layers / max(sim.hillslope_flow_path_m, 1.0e-9)
    dtheta_rate = vertical_dtheta_rate - lateral_sink_rate - aet_sink
    max_abs_rate = float(np.max(np.abs(dtheta_rate)))

    if max_abs_rate <= 1.0e-15:
        return max_dt_seconds

    dt_limit = max_dtheta_per_substep / max_abs_rate
    return float(np.clip(dt_limit, min_dt_seconds, max_dt_seconds))


def run_simulation(
    soil: SoilProperties,
    column: ColumnConfig,
    sim: SimulationConfig,
    forcing: pd.DataFrame,
    output_dir: Path,
    vegetation: Optional[VegetationType] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    column_nodes, depth_m = build_column_geometry(column.nz, column.dz_m)

    initial_head = np.full(column.nz, sim.initial_head_m, dtype=float)
    initial_theta = theta_from_head(initial_head, soil)
    state = SimulationState(head_m=initial_head, theta=initial_theta)

    if vegetation is not None:
        print(
            f"Vegetation: {vegetation.name} "
            f"(rooting depth {vegetation.rooting_depth_m:.2f} m, "
            f"PET scale {vegetation.pet_scale:.2f})"
        )

    profiles: list[dict[str, float]] = []
    diagnostics_rows: list[dict[str, float]] = []

    elapsed_s = 0.0
    cumulative_infiltration_m = 0.0

    for i, row in forcing.iterrows():
        duration_s = float(row["dt_seconds"])
        rain = float(row["rainfall_mm_h"])
        pet = float(row["pet_mm_h"])

        step_diag: Optional[dict[str, float]] = None
        remaining_s = duration_s
        n_iter = 0
        while remaining_s > 0.0:
            n_iter += 1
            if n_iter > 200000:
                raise RuntimeError(
                    "Exceeded maximum adaptive sub-steps in one forcing interval. "
                    "Try increasing min_substep_seconds or relaxing max_dtheta_per_substep."
                )

            candidate_max_dt = min(sim.max_substep_seconds, remaining_s)
            stable_dt = estimate_stable_dt_seconds(
                state,
                soil,
                sim,
                cumulative_infiltration_m,
                column.dz_m,
                rain,
                pet,
                sim.min_theta_buffer,
                sim.max_dtheta_per_substep,
                candidate_max_dt,
                sim.min_substep_seconds,
                depth_m=depth_m,
                vegetation=vegetation,
            )
            dt_s = min(candidate_max_dt, stable_dt, remaining_s)

            state, step_diag, cumulative_infiltration_m = one_substep(
                state,
                soil,
                sim,
                cumulative_infiltration_m,
                column.dz_m,
                dt_s,
                rain,
                pet,
                sim.min_theta_buffer,
                depth_m=depth_m,
                vegetation=vegetation,
            )

            if not (np.all(np.isfinite(state.theta)) and np.all(np.isfinite(state.head_m))):
                raise FloatingPointError(
                    "Non-finite state encountered during integration. "
                    "Reduce max_substep_seconds or max_dtheta_per_substep."
                )

            elapsed_s += dt_s
            remaining_s -= dt_s

        if step_diag is None:
            continue

        conductivity = hydraulic_conductivity(state.head_m, soil)

        timestamp = row["timestamp"] if "timestamp" in forcing.columns else pd.NaT

        for node in range(column.nz):
            profiles.append(
                {
                    "step": i,
                    "time_hours": elapsed_s / 3600.0,
                    "timestamp": timestamp,
                    "depth_m": depth_m[node],
                    "theta": state.theta[node],
                    "head_m": state.head_m[node],
                    "conductivity_m_per_s": conductivity[node],
                }
            )

        diagnostics_rows.append(
            {
                "step": i,
                "time_hours": elapsed_s / 3600.0,
                "timestamp": timestamp,
                "slope_angle_deg": sim.slope_angle_deg,
                "rainfall_mm_h": rain,
                "pet_mm_h": pet,
                "aet_mm_h": step_diag["aet_flux_m_per_s"] * 1000.0 * 3600.0,
                "infiltration_mm_h": step_diag["infiltration_flux_m_per_s"] * 1000.0 * 3600.0,
                "runoff_mm_h": step_diag["runoff_flux_m_per_s"] * 1000.0 * 3600.0,
                "cumulative_infiltration_mm": step_diag["cumulative_infiltration_m"] * 1000.0,
                "surface_net_downward_flux_mm_h": step_diag["surface_net_downward_flux_m_per_s"]
                * 1000.0
                * 3600.0,
                "bottom_drainage_mm_h": step_diag["bottom_downward_flux_m_per_s"] * 1000.0 * 3600.0,
                "lateral_throughflow_mm_h": step_diag["lateral_throughflow_flux_m_per_s"] * 1000.0 * 3600.0,
            }
        )

    profile_df = pd.DataFrame(profiles)
    diagnostics_df = pd.DataFrame(diagnostics_rows)

    profile_csv = output_dir / "soil_moisture_profiles.csv"
    diagnostics_csv = output_dir / "water_balance_diagnostics.csv"

    profile_df.to_csv(profile_csv, index=False)
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    plot_moisture_heatmap(profile_df, diagnostics_df, output_dir / "moisture_heatmap.png")

    print(f"Wrote profile output: {profile_csv}")
    print(f"Wrote diagnostics output: {diagnostics_csv}")
    print(f"Wrote visualization: {output_dir / 'moisture_heatmap.png'}")


def plot_moisture_heatmap(
    profile_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    output_png: Path,
) -> None:
    pivot = profile_df.pivot(index="depth_m", columns="time_hours", values="theta").sort_index(ascending=True)
    mean_theta = profile_df.groupby("time_hours", as_index=False)["theta"].mean()

    fig, (ax_rain, ax_runoff, ax_lateral, ax_mean, ax_moisture) = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(12, 6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 4], "hspace": 0.08},
    )

    rain_x = diagnostics_df["time_hours"].to_numpy(dtype=float)
    rain_y = diagnostics_df["rainfall_mm_h"].to_numpy(dtype=float)
    ax_rain.step(rain_x, rain_y, where="post", color="#1f77b4", linewidth=1.8)
    ax_rain.fill_between(rain_x, rain_y, step="post", alpha=0.25, color="#1f77b4")
    ax_rain.set_ylabel("Rain\n(mm/h)")
    ax_rain.set_title("Rainfall, Runoff, Lateral Throughflow, and Soil Moisture Evolution")
    ax_rain.grid(True, axis="y", alpha=0.25)

    runoff_y = diagnostics_df["runoff_mm_h"].to_numpy(dtype=float)
    ax_runoff.step(rain_x, runoff_y, where="post", color="#d62728", linewidth=1.8)
    ax_runoff.fill_between(rain_x, runoff_y, step="post", alpha=0.2, color="#d62728")
    ax_runoff.set_ylabel("Runoff\n(mm/h)")
    ax_runoff.grid(True, axis="y", alpha=0.25)

    lateral_y = diagnostics_df["lateral_throughflow_mm_h"].to_numpy(dtype=float)
    ax_lateral.plot(rain_x, lateral_y, color="#9467bd", linewidth=1.8)
    ax_lateral.fill_between(rain_x, lateral_y, alpha=0.2, color="#9467bd")
    ax_lateral.set_ylabel("Lateral\n(mm/h)")
    ax_lateral.grid(True, axis="y", alpha=0.25)

    mean_x = mean_theta["time_hours"].to_numpy(dtype=float)
    mean_y = mean_theta["theta"].to_numpy(dtype=float)
    ax_mean.plot(mean_x, mean_y, color="#2ca02c", linewidth=1.8)
    ax_mean.set_ylabel("Mean\ntheta")
    ax_mean.grid(True, axis="y", alpha=0.25)

    mesh = ax_moisture.pcolormesh(
        pivot.columns.values,
        pivot.index.values,
        pivot.values,
        shading="auto",
        cmap="YlGnBu",
    )
    ax_moisture.invert_yaxis()
    ax_moisture.set_xlabel("Time (hours)")
    ax_moisture.set_ylabel("Depth (m)")
    cbar = plt.colorbar(mesh, ax=ax_moisture)
    cbar.set_label("Volumetric moisture, theta (-)")
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def fetch_openweather_forcing(
    api_key: str,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    """Fetch hourly observed + forecast forcing from the OpenWeatherMap One Call API 3.0.

    Returns a DataFrame with columns ``timestamp``, ``rainfall_mm_h``, and
    ``pet_mm_h`` compatible with :func:`read_forcing`.

    The function combines:
    * Today's observed hourly data (``/timemachine`` for the past 24 h).
    * Up to 8 days of hourly forecast from the standard One Call endpoint.

    PET is estimated from the Hargreaves–Samani equation using temperature and
    the latitude-based extra-terrestrial radiation for the day of year, since
    the free tier of OpenWeatherMap does not provide radiation directly.

    Parameters
    ----------
    api_key:
        OpenWeatherMap API key (v3.0 subscription required).
    lat:
        Site latitude in decimal degrees.
    lon:
        Site longitude in decimal degrees.
    """
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "The 'requests' package is required for OpenWeather integration. "
            "Install it with: pip install requests"
        ) from exc

    import math
    from datetime import datetime, timezone, timedelta

    BASE_URL = "https://api.openweathermap.org/data/3.0/onecall"

    # ------------------------------------------------------------------
    # 1.  Forecast data (hourly, up to 48 h) + daily (up to 8 days)
    # ------------------------------------------------------------------
    params_forecast = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
        "exclude": "current,minutely,alerts",
    }
    resp = requests.get(BASE_URL, params=params_forecast, timeout=30)
    resp.raise_for_status()
    ow = resp.json()

    rows: list[dict] = []

    hourly = ow.get("hourly", [])
    for h in hourly:
        ts = pd.Timestamp(h["dt"], unit="s", tz="UTC")
        rain_mm = h.get("rain", {}).get("1h", 0.0)
        rows.append({"timestamp": ts, "rainfall_mm_h": rain_mm, "_temp_c": h["temp"]})

    # Fill any remaining days from daily data (beyond hourly window)
    daily = ow.get("daily", [])
    hourly_timestamps = {r["timestamp"] for r in rows}
    for d in daily:
        day_start = pd.Timestamp(d["dt"], unit="s", tz="UTC").normalize()
        rain_mm_day = d.get("rain", 0.0)
        rain_mm_h = rain_mm_day / 24.0
        temp_mean = (d["temp"]["max"] + d["temp"]["min"]) / 2.0
        for hour_offset in range(24):
            ts = day_start + pd.Timedelta(hours=hour_offset)
            if ts not in hourly_timestamps:
                rows.append({"timestamp": ts, "rainfall_mm_h": rain_mm_h, "_temp_c": temp_mean})

    # ------------------------------------------------------------------
    # 2.  Yesterday's observed data via timemachine (last 24 h)
    # ------------------------------------------------------------------
    now_utc = datetime.now(timezone.utc)
    yesterday_dt = int((now_utc - timedelta(hours=24)).timestamp())
    params_hist = {
        "lat": lat,
        "lon": lon,
        "dt": yesterday_dt,
        "appid": api_key,
        "units": "metric",
    }
    try:
        resp_hist = requests.get(f"{BASE_URL}/timemachine", params=params_hist, timeout=30)
        resp_hist.raise_for_status()
        hist_data = resp_hist.json()
        for h in hist_data.get("data", []):
            ts = pd.Timestamp(h["dt"], unit="s", tz="UTC")
            rain_mm = h.get("rain", {}).get("1h", 0.0)
            rows.append({"timestamp": ts, "rainfall_mm_h": rain_mm, "_temp_c": h["temp"]})
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not fetch historical weather data: {exc}")

    if not rows:
        raise ValueError("OpenWeatherMap returned no usable data.")

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3.  Estimate PET using Hargreaves–Samani simplified equation
    # ------------------------------------------------------------------
    lat_rad = math.radians(lat)

    def _pet_mm_h(row: pd.Series) -> float:
        """Hargreaves–Samani PET, returned as mm/h."""
        ts = row["timestamp"]
        day_of_year = ts.day_of_year
        # Extra-terrestrial radiation (MJ/m²/day)
        dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * day_of_year / 365.0)
        delta = 0.409 * math.sin(2.0 * math.pi * day_of_year / 365.0 - 1.39)
        omega_s = math.acos(-math.tan(lat_rad) * math.tan(delta))
        ra = (
            24.0
            * 60.0
            / math.pi
            * 0.0820
            * dr
            * (
                omega_s * math.sin(lat_rad) * math.sin(delta)
                + math.cos(lat_rad) * math.cos(delta) * math.sin(omega_s)
            )
        )  # MJ/m²/day
        # Hargreaves–Samani: ET0 (mm/day) = 0.0023 * Ra * (Tmean + 17.8) * sqrt(Trange)
        # We don't have Tmax/Tmin per hour so use a conservative Trange estimate of 0
        # (gives minimum PET estimate; still reasonable for soil moisture simulation)
        t_mean = float(row["_temp_c"])
        et0_mm_day = max(0.0023 * ra * (t_mean + 17.8) * 1.0, 0.0)
        return et0_mm_day / 24.0

    df["pet_mm_h"] = df.apply(_pet_mm_h, axis=1)
    df = df.drop(columns=["_temp_c"])

    print(
        f"Fetched {len(df)} hourly records from OpenWeatherMap "
        f"({df['timestamp'].min()} – {df['timestamp'].max()})"
    )
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate vertical soil-water flow with Richards equation.")
    parser.add_argument(
        "--soil-config",
        type=Path,
        default=Path("soil_properties_example.json"),
        help="Path to JSON file with soil, column, and simulation settings.",
    )
    parser.add_argument(
        "--forcing-csv",
        type=Path,
        default=None,
        help=(
            "Path to forcing CSV with rainfall_mm_h, pet_mm_h, and optional timestamp or dt_hours. "
            "Mutually exclusive with --openweather-key."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to write CSV and plot outputs.",
    )

    # Vegetation options
    veg_names = sorted(VEGETATION_LIBRARY.keys())
    parser.add_argument(
        "--vegetation",
        type=str,
        default=None,
        choices=veg_names,
        metavar="TYPE",
        help=(
            "Optional vegetation type for root-zone water uptake. "
            f"Available types: {', '.join(veg_names)}."
        ),
    )
    parser.add_argument(
        "--list-vegetation",
        action="store_true",
        help="Print available vegetation types with their rooting depths and exit.",
    )

    # OpenWeather options
    ow_group = parser.add_argument_group("OpenWeatherMap forcing (alternative to --forcing-csv)")
    ow_group.add_argument(
        "--openweather-key",
        type=str,
        default=None,
        metavar="API_KEY",
        help="OpenWeatherMap API key (One Call API 3.0). Fetches observed + 7-day forecast.",
    )
    ow_group.add_argument(
        "--openweather-lat",
        type=float,
        default=None,
        metavar="LAT",
        help="Site latitude in decimal degrees (required with --openweather-key).",
    )
    ow_group.add_argument(
        "--openweather-lon",
        type=float,
        default=None,
        metavar="LON",
        help="Site longitude in decimal degrees (required with --openweather-key).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_vegetation:
        print(f"{'Name':<25} {'Rooting depth (m)':>18} {'PET scale':>10}  Description")
        print("-" * 80)
        for name, veg in sorted(VEGETATION_LIBRARY.items()):
            print(f"{name:<25} {veg.rooting_depth_m:>18.2f} {veg.pet_scale:>10.2f}  {veg.description}")
        return

    soil, column, sim = read_config(args.soil_config)

    # Resolve vegetation
    vegetation: Optional[VegetationType] = None
    if args.vegetation is not None:
        vegetation = VEGETATION_LIBRARY[args.vegetation]

    # Resolve forcing source
    if args.openweather_key is not None:
        if args.openweather_lat is None or args.openweather_lon is None:
            raise SystemExit(
                "Error: --openweather-lat and --openweather-lon are required when using --openweather-key."
            )
        if args.forcing_csv is not None:
            raise SystemExit(
                "Error: --forcing-csv and --openweather-key are mutually exclusive."
            )
        forcing = fetch_openweather_forcing(
            api_key=args.openweather_key,
            lat=args.openweather_lat,
            lon=args.openweather_lon,
        )
        forcing = _process_forcing_df(forcing, sim.default_dt_hours, source_name="OpenWeatherMap")
    else:
        forcing_csv = args.forcing_csv if args.forcing_csv is not None else Path("forcing_example.csv")
        forcing = read_forcing(forcing_csv, sim.default_dt_hours)

    run_simulation(soil, column, sim, forcing, args.output_dir, vegetation=vegetation)


if __name__ == "__main__":
    main()
