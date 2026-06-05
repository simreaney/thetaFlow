# ThetaFlow: A LandLab-based Richards equation simulator for vadose-zone dynamics.


ThetaFlow simulates 1D vertical water movement in a soil column using:

- A LandLab `RasterModelGrid` to hold column state and fields
- The mixed-form Richards equation (Darcy flux + mass conservation)
- Van Genuchten-Mualem hydraulic functions
- Time-series forcing from rainfall and PET

The model is designed as a transparent, configurable research/teaching workflow for
unsaturated flow in a vertical soil profile.

## What the Model Solves

The code solves 1D vertical unsaturated flow by combining:

- Darcy's law for interface water fluxes
- Water mass conservation in each control volume
- Nonlinear constitutive relationships for `theta(h)` and `K(h)`

State variables at each soil layer are:

- `h` (pressure head, m)
- `theta` (volumetric water content, m3/m3)

## Model Equations

### 1) Mixed-form Richards equation (1D vertical)

In continuous form, the model follows:

$$
\frac{\partial \theta}{\partial t} = -\frac{\partial q}{\partial z}
$$

with Darcy flux:

$$
q = -K(h)\left(\frac{\partial h}{\partial z} + 1\right)
$$

where:

- $z$ is positive downward (m)
- $q$ is vertical flux (m/s), positive downward in this sign convention
- $K(h)$ is unsaturated hydraulic conductivity (m/s)

### 2) van Genuchten water retention curve

Effective saturation:

$$
S_e = \frac{\theta - \theta_r}{\theta_s - \theta_r}
$$

For unsaturated conditions (`h < 0`):

$$
S_e(h) = \left[1 + (\alpha |h|)^n\right]^{-m}, \quad m = 1 - \frac{1}{n}
$$

Then:

$$
	heta(h) = \theta_r + S_e(h)(\theta_s - \theta_r)
$$

For saturated/ponded conditions (`h >= 0`), the implementation caps to $S_e = 1$ and
$\theta = \theta_s$ (subject to numerical clipping near exact bounds).

### 3) Mualem-van Genuchten hydraulic conductivity

$$
K(S_e) = K_s S_e^l\left[1 - \left(1 - S_e^{1/m}\right)^m\right]^2
$$

where:

- $K_s$ is saturated conductivity (m/s)
- $l$ is the pore-connectivity parameter (`pore_connectivity`, often 0.5)

## Numerical Implementation

The script uses an explicit finite-volume style update over layer thickness `dz`:

$$
	heta_i^{t+\Delta t} = \theta_i^t + \frac{\Delta t}{\Delta z}\left(q_{i+1/2} - q_{i-1/2}\right)
$$

Interface fluxes are computed from averaged conductivity and local head gradient.

To improve stability, each forcing interval is split into adaptive sub-steps bounded by:

- `max_substep_seconds` (upper cap)
- `min_substep_seconds` (lower cap)
- `max_dtheta_per_substep` (target maximum moisture change per layer per sub-step)

After each moisture update, `theta` is converted back to `h` via the inverse
van Genuchten relation.

## Boundary Conditions and Forcing

### Upper boundary

At the surface:

$$
q_{top} = I - ET_a
$$

where:

- $P$ is rainfall flux from `rainfall_mm_h`
- $I$ is actual infiltration flux into soil
- $ET_a$ is actual evaporation demand limited by available water in the top layer

Overland flow is generated when rainfall exceeds infiltration capacity:

$$
I = \min(P, I_{cap}), \quad R = P - I
$$

where $R$ is overland flow (Hortonian excess runoff).

In ThetaFlow, infiltration capacity is computed with the Green-Ampt form:

$$
I_{cap} = K_s\left(1 + \frac{\psi_f\,\Delta\theta}{F}\right)
$$

with:

- $\psi_f$: wetting-front suction head (`green_ampt_wetting_front_suction_m`)
- $\Delta\theta = \theta_s - \theta_{surface}$ (surface moisture deficit)
- $F$: cumulative infiltration depth since simulation start

Potential ET input (`pet_mm_h`) is converted to flux and constrained so the top layer
does not drop below approximately `theta_r + min_theta_buffer` during a sub-step.

### Lower boundary

The bottom boundary is free drainage:

$$
q_{bottom} = K(h_{bottom})
$$

(downward drainage under unit hydraulic gradient).

## Worked Example (One Time Step)

This section shows one approximate hand-calculation using the provided example settings.
Values are rounded for readability.

Given from [soil_properties_example.json](soil_properties_example.json):

- $\theta_r = 0.065$
- $\theta_s = 0.41$
- $\alpha = 3.6\ \text{m}^{-1}$
- $n = 1.56 \Rightarrow m = 1 - 1/n \approx 0.359$
- $K_s = 1.5\times10^{-5}\ \text{m/s}$
- $l = 0.5$

Initial condition:

- $h = -1.0\ \text{m}$ everywhere

Forcing during a rainy hour from [forcing_example.csv](forcing_example.csv):

- rainfall $= 4\ \text{mm/h}$
- PET $= 0.10\ \text{mm/h}$

Converted fluxes:

$$
P = \frac{4}{1000\times3600} \approx 1.11\times10^{-6}\ \text{m/s}
$$

$$
PET = \frac{0.10}{1000\times3600} \approx 2.78\times10^{-8}\ \text{m/s}
$$

At this wetness, PET demand is usually fully met at the surface, so:

$$
q_{top} = P - ET_a \approx 1.11\times10^{-6} - 2.78\times10^{-8}
\approx 1.08\times10^{-6}\ \text{m/s}
$$

### Step 1: Convert pressure head to moisture

At $h=-1$ m:

$$
S_e = [1 + (\alpha|h|)^n]^{-m} = [1 + 3.6^{1.56}]^{-0.359} \approx 0.66
$$

$$
	heta = \theta_r + S_e(\theta_s-\theta_r)
= 0.065 + 0.66\times(0.41-0.065)
\approx 0.29
$$

### Step 2: Compute unsaturated conductivity

$$
K = K_s S_e^l\left[1-(1-S_e^{1/m})^m\right]^2
\approx 1.5\times10^{-5}\times 0.066
\approx 9.9\times10^{-7}\ \text{m/s}
$$

This is close to the imposed surface inflow, so infiltration can proceed without
large immediate ponding in this simple case.

### Step 3: Flux divergence and moisture update

For top layer thickness $\Delta z=0.025$ m and a sub-step $\Delta t=60$ s,
the explicit update is:

$$
\Delta\theta \approx \frac{\Delta t}{\Delta z}(q_{in} - q_{out})
$$

Using $q_{in}\approx q_{top}=1.08\times10^{-6}$ m/s and
$q_{out}\approx K\approx 9.9\times10^{-7}$ m/s:

$$
\Delta\theta \approx \frac{60}{0.025}(8.7\times10^{-8})
\approx 2.1\times10^{-4}
$$

So the top-layer moisture changes from about $0.2930$ to $0.2932$ in this sub-step,
then the model repeats over all sub-steps and layers.

### Interpretation

- Rainfall raises near-surface moisture first.
- Conductivity increases as the soil wets, accelerating downward redistribution.
- After rainfall ends, PET and gravity drainage reduce near-surface moisture.
- These dynamics appear clearly in the generated moisture heatmap output.

## Inputs

### 1) Soil and model configuration (JSON)

Use `soil_properties_example.json` as a template.

Key fields:

- `soil.theta_r`: residual volumetric water content
- `soil.theta_s`: saturated volumetric water content
- `soil.alpha_per_m`: van Genuchten alpha (1/m)
- `soil.n`: van Genuchten n
- `soil.ks_m_per_s`: saturated hydraulic conductivity (m/s)
- `soil.pore_connectivity`: Mualem pore-connectivity parameter (often ~0.5)
- `column.nz`: number of soil layers (nodes)
- `column.dz_m`: layer thickness (m)
- `simulation.initial_head_m`: initial pressure head (m)
- `simulation.default_dt_hours`: used when forcing does not provide explicit timing
- `simulation.max_substep_seconds`: explicit integration sub-step limit
- `simulation.min_substep_seconds`: smallest adaptive sub-step allowed
- `simulation.max_dtheta_per_substep`: adaptive cap on moisture increment per sub-step
- `simulation.min_theta_buffer`: moisture buffer above residual for evaporation limiting
- `simulation.green_ampt_wetting_front_suction_m`: Green-Ampt wetting-front suction head

Recommended checks when supplying parameters:

- Ensure `theta_r < theta_s`
- Typical `n > 1`
- `ks_m_per_s` should be physically consistent with texture/structure
- `dz_m` and `max_substep_seconds` should be small enough for stable explicit updates
- If oscillations appear, reduce `max_substep_seconds` and/or `max_dtheta_per_substep`

### 2) Forcing time series (CSV)

Use `forcing_example.csv` as a template.

Required columns:

- `rainfall_mm_h`
- `pet_mm_h`

Optional timing columns:

- `timestamp` (preferred)
- `dt_hours` (if `timestamp` is not provided)

If neither is provided, `default_dt_hours` from JSON is used.

Unit conventions:

- Rainfall and PET are in mm/h
- Internally converted to m/s
- Time output is in elapsed hours

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are using conda, create/activate your environment first, then run:

```bash
pip install -r requirements.txt
```

## Run

```bash
python simulate_soil_column.py \
  --soil-config soil_properties_example.json \
  --forcing-csv forcing_example.csv \
  --output-dir outputs
```

Custom paths are fully supported via these command-line arguments.

## Outputs

Generated in `outputs/`:

- `soil_moisture_profiles.csv`: depth-wise moisture/head/conductivity at each forcing step
- `water_balance_diagnostics.csv`: rainfall, PET, estimated AET, net surface flux, bottom drainage
- `moisture_heatmap.png`: time-depth moisture visualization

### `soil_moisture_profiles.csv` columns

- `step`: forcing step index
- `time_hours`: elapsed simulation time (h)
- `timestamp`: timestamp if provided in forcing
- `depth_m`: node depth from surface (m)
- `theta`: volumetric water content (m3/m3)
- `head_m`: pressure head (m)
- `conductivity_m_per_s`: unsaturated hydraulic conductivity (m/s)

### `water_balance_diagnostics.csv` columns

- `rainfall_mm_h`
- `pet_mm_h`
- `aet_mm_h` (actual ET achieved under moisture limitation)
- `infiltration_mm_h` (rainfall that enters the soil column)
- `runoff_mm_h` (rainfall excess routed to overland flow)
- `cumulative_infiltration_mm` (Green-Ampt cumulative infiltration depth)
- `surface_net_downward_flux_mm_h`
- `bottom_drainage_mm_h`

## Visualization

`moisture_heatmap.png` is a depth-time map of `theta`:

- x-axis: simulation time (h)
- y-axis: depth (m, increasing downward)
- color: volumetric moisture

This quickly shows infiltration fronts, redistribution, and drying periods.

## Assumptions and Limitations

- 1D vertical flow only (no lateral flow)
- Homogeneous soil hydraulic properties with depth
- Explicit time integration (can require small sub-steps for high conductivity / sharp fronts)
- PET is treated as near-surface evaporative demand (no explicit root profile yet)
- No hysteresis in the retention curve

## Notes on Physics and Stability

The solver uses:

- Darcy flux in 1D vertical form for each layer interface
- Continuity equation to update volumetric moisture
- Van Genuchten retention to map between pressure head and moisture
- Mualem conductivity model for unsaturated hydraulic conductivity

This is an explicit mixed-form implementation, so stability depends on chosen `max_substep_seconds`, soil conductivity, and layer spacing.

If runs become noisy or unstable, reduce:

- `simulation.max_substep_seconds`
- `column.dz_m`

and re-test mass-balance diagnostics.

## Suggested Extensions

- Add depth-distributed root water uptake function
- Add alternative lower boundary (fixed head / fluctuating water table)
- Add calibration workflow against observed profile moisture
- Export additional plots (profile snapshots, cumulative infiltration, cumulative ET)
