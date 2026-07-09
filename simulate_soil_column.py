#!/usr/bin/env python3
"""Richards-equation soil column simulation for a 1D vertical soil column."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def read_forcing(forcing_csv: Path, default_dt_hours: float) -> pd.DataFrame:
    forcing = pd.read_csv(forcing_csv)
    expected = {"rainfall_mm_h", "pet_mm_h"}
    missing = expected.difference(forcing.columns)
    if missing:
        raise ValueError(f"Forcing CSV is missing required columns: {sorted(missing)}")

    if "timestamp" in forcing.columns:
        forcing["timestamp"] = pd.to_datetime(forcing["timestamp"])
        dt_seconds = (forcing["timestamp"].shift(-1) - forcing["timestamp"]).dt.total_seconds()
        fallback_dt_s = default_dt_hours * 3600.0
        positive_dt = dt_seconds[dt_seconds > 0]
        inferred_dt_s = float(positive_dt.median()) if not positive_dt.empty else fallback_dt_s
        invalid = dt_seconds <= 0
        invalid_count = int(invalid.fillna(False).sum())
        if invalid_count > 0:
            print(
                "Warning: found "
                f"{invalid_count} non-positive timestamp interval(s) in {forcing_csv.name}; "
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


def build_column_geometry(nz: int, dz_m: float) -> tuple[np.ndarray, np.ndarray]:
    column_nodes = np.arange(nz) * 2
    depth_m = np.arange(nz, dtype=float) * dz_m
    return column_nodes, depth_m


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
) -> tuple[SimulationState, dict[str, float], float]:
    theta = state.theta.copy()
    head_m = state.head_m.copy()

    rain_flux = rainfall_mm_h / 1000.0 / 3600.0
    pet_flux = max(pet_mm_h, 0.0) / 1000.0 / 3600.0

    available_evap = max((theta[0] - (soil.theta_r + min_theta_buffer)) * dz_m / dt_s, 0.0)
    aet_flux = min(pet_flux, available_evap)

    conductivity = hydraulic_conductivity(head_m, soil)
    theta_deficit = max(soil.theta_s - theta[0], 1.0e-6)
    cumulative_for_capacity = max(cumulative_infiltration_m, 1.0e-6)
    infiltration_capacity_flux = soil.ks_m_per_s * (
        1.0 + (sim.green_ampt_wetting_front_suction_m * theta_deficit) / cumulative_for_capacity
    )
    infiltration_capacity_flux = max(infiltration_capacity_flux, soil.ks_m_per_s)
    infiltration_flux = min(rain_flux, infiltration_capacity_flux)
    runoff_flux = max(rain_flux - infiltration_flux, 0.0)
    net_downward_flux = infiltration_flux - aet_flux
    cumulative_infiltration_m_new = cumulative_infiltration_m + infiltration_flux * dt_s

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

    # Represent lateral throughflow for a hillslope block as gravity-driven flow along slope.
    slope_angle_rad = np.deg2rad(sim.slope_angle_deg)
    lateral_flux_layers = conductivity * np.sin(slope_angle_rad)
    lateral_sink_rate = lateral_flux_layers / max(sim.hillslope_flow_path_m, 1.0e-9)

    theta_new = theta + dt_s * (vertical_dtheta_rate - lateral_sink_rate)
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
) -> float:
    theta = state.theta
    head_m = state.head_m

    rain_flux = rainfall_mm_h / 1000.0 / 3600.0
    pet_flux = max(pet_mm_h, 0.0) / 1000.0 / 3600.0
    available_evap_rate = max((theta[0] - (soil.theta_r + min_theta_buffer)) * dz_m / max_dt_seconds, 0.0)
    aet_flux = min(pet_flux, available_evap_rate)

    conductivity = hydraulic_conductivity(head_m, soil)
    theta_deficit = max(soil.theta_s - theta[0], 1.0e-6)
    cumulative_for_capacity = max(cumulative_infiltration_m, 1.0e-6)
    infiltration_capacity_flux = soil.ks_m_per_s * (
        1.0 + (sim.green_ampt_wetting_front_suction_m * theta_deficit) / cumulative_for_capacity
    )
    infiltration_capacity_flux = max(infiltration_capacity_flux, soil.ks_m_per_s)
    infiltration_flux = min(rain_flux, infiltration_capacity_flux)
    net_downward_flux = infiltration_flux - aet_flux

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
    dtheta_rate = vertical_dtheta_rate - lateral_sink_rate
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
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    column_nodes, depth_m = build_column_geometry(column.nz, column.dz_m)

    initial_head = np.full(column.nz, sim.initial_head_m, dtype=float)
    initial_theta = theta_from_head(initial_head, soil)
    state = SimulationState(head_m=initial_head, theta=initial_theta)

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
        default=Path("forcing_example.csv"),
        help="Path to forcing CSV with rainfall_mm_h, pet_mm_h, and optional timestamp or dt_hours.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to write CSV and plot outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    soil, column, sim = read_config(args.soil_config)
    forcing = read_forcing(args.forcing_csv, sim.default_dt_hours)
    run_simulation(soil, column, sim, forcing, args.output_dir)


if __name__ == "__main__":
    main()
