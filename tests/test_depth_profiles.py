import numpy as np
import pytest

from simulate_soil_column import SoilProperties
from soil_depth import (
    Horizon,
    RegressionRule,
    SafeExpression,
    base_grids_from_soil_list,
    group_horizon_grids,
    parse_horizon_name,
    profiles_from_horizons,
    profiles_from_regression,
    soil_list_from_profiles,
    validate_profiles,
)

LOAM = SoilProperties(theta_r=0.078, theta_s=0.43, alpha_per_m=3.6, n=1.56, ks_m_per_s=2.89e-6)
DEPTH = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])


def _base(shape=(2, 3)):
    return base_grids_from_soil_list([LOAM] * (shape[0] * shape[1]), shape)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("ks_0-5cm", ("ks_m_per_s", 0.0, 0.05)),
        ("theta_s_15-30cm", ("theta_s", 0.15, 0.30)),
        ("alpha_0-0.3m", ("alpha_per_m", 0.0, 0.3)),
        ("Ksat_5-15cm_mean", ("ks_m_per_s", 0.05, 0.15)),
        ("n_30-60cm", ("n", 0.30, 0.60)),
    ],
)
def test_parse_horizon_name(name, expected):
    prop, top, bottom = parse_horizon_name(name)
    assert prop == expected[0]
    assert top == pytest.approx(expected[1])
    assert bottom == pytest.approx(expected[2])


@pytest.mark.parametrize("name", ["dem", "clay_0-5cm_mean", "ks-0-5cm"])
def test_parse_horizon_name_rejects(name):
    assert parse_horizon_name(name) is None


def test_step_profiles_assign_containing_horizon_and_extend_below():
    base = _base()
    hz = group_horizon_grids([
        ("ks", 0.0, 0.15, np.full((2, 3), 1e-5)),
        ("ks", 0.15, 0.35, np.full((2, 3), 1e-6)),
    ])
    prof = profiles_from_horizons(hz, DEPTH, base, interp="step")
    ks = prof["ks_m_per_s"][0, 0]
    np.testing.assert_allclose(ks, [1e-5, 1e-5, 1e-6, 1e-6, 1e-6, 1e-6])
    # Property without a horizon grid stays uniform at the base value
    np.testing.assert_allclose(prof["theta_s"], 0.43)
    assert prof["ks_m_per_s"].shape == (2, 3, DEPTH.size)


def test_linear_profiles_interpolate_between_mid_depths():
    base = _base()
    hz = [
        Horizon(0.0, 0.2, {"theta_s": np.full((2, 3), 0.5)}),   # mid 0.1
        Horizon(0.2, 0.6, {"theta_s": np.full((2, 3), 0.3)}),   # mid 0.4
    ]
    prof = profiles_from_horizons(hz, DEPTH, base, interp="linear")
    np.testing.assert_allclose(
        prof["theta_s"][1, 2], [0.5, 0.5, 0.5 - 0.2 / 3, 0.5 - 0.4 / 3, 0.3, 0.3]
    )


def test_horizon_nodata_falls_back_to_base():
    base = _base()
    grid = np.full((2, 3), 0.5)
    grid[0, 1] = np.nan
    prof = profiles_from_horizons([Horizon(0, 1, {"theta_s": grid})], DEPTH, base)
    np.testing.assert_allclose(prof["theta_s"][0, 1], 0.43)
    np.testing.assert_allclose(prof["theta_s"][0, 0], 0.5)


def test_regression_forms():
    base = _base()
    coef_grid = np.full((2, 3), 0.25)
    rules = [
        RegressionRule("ks", "exp", coef_grid),
        RegressionRule("theta_s", "linear", -0.1),
        RegressionRule("alpha", "power", -1.0),
        RegressionRule("n", "expr", expression="p0 + 0.1 * where(z > 0.25, 1, 0)"),
    ]
    prof = profiles_from_regression(rules, DEPTH, base)
    np.testing.assert_allclose(prof["ks_m_per_s"][0, 0], 2.89e-6 * np.exp(-DEPTH / 0.25))
    np.testing.assert_allclose(prof["theta_s"][1, 1], 0.43 - 0.1 * DEPTH)
    np.testing.assert_allclose(prof["alpha_per_m"][0, 2], 3.6 / (1.0 + DEPTH))
    np.testing.assert_allclose(prof["n"][0, 0], 1.56 + 0.1 * (DEPTH > 0.25))
    np.testing.assert_allclose(prof["theta_r"], 0.078)


def test_regression_expression_with_raster_variable():
    base = _base()
    depth_to_bedrock = np.array([[0.2, 0.2, 0.2], [1.0, 1.0, 1.0]])
    rules = [RegressionRule("ks", "expr", expression="where(z < dtb, p0, p0 * 0.01)")]
    prof = profiles_from_regression(rules, DEPTH, base, variables={"dtb": depth_to_bedrock})
    np.testing.assert_allclose(prof["ks_m_per_s"][0, 0, :2], 2.89e-6)
    np.testing.assert_allclose(prof["ks_m_per_s"][0, 0, 2:], 2.89e-8)
    np.testing.assert_allclose(prof["ks_m_per_s"][1, 0], 2.89e-6)


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os').system('echo hi')",
        "p0.__class__",
        "z[0]",
        "open('x')",
        "lambda: 1",
        "exp(z, out=z)",
        "'abc'",
        "0 < z < 1",
    ],
)
def test_safe_expression_rejects_unsafe_syntax(text):
    with pytest.raises(ValueError):
        SafeExpression(text)


def test_safe_expression_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown name"):
        SafeExpression("p0 * foo").evaluate({"p0": 1.0, "z": 0.0})


def test_validate_profiles_clips_and_fills():
    base = _base()
    prof = profiles_from_regression([RegressionRule("theta_s", "linear", -2.0)], DEPTH, base)
    prof["n"][0, 0, 0] = 0.5
    prof["ks_m_per_s"][1, 1, 3] = np.nan
    fixed, report = validate_profiles(prof, base)
    assert fixed["theta_s"].min() >= 0.01
    assert np.all(fixed["theta_r"] <= fixed["theta_s"] - 0.01 + 1e-12)
    assert fixed["n"][0, 0, 0] == pytest.approx(1.01)
    assert fixed["ks_m_per_s"][1, 1, 3] == pytest.approx(2.89e-6)
    assert report["n"] == 1 and report["ks_m_per_s"] == 1 and report["theta_s"] > 0


def test_soil_list_from_profiles_row_major():
    base = _base()
    base["ks_m_per_s"] = np.arange(6, dtype=float).reshape(2, 3) + 1.0
    prof = profiles_from_regression([], DEPTH, base)
    soils = soil_list_from_profiles(prof)
    assert len(soils) == 6
    np.testing.assert_allclose(soils[4].ks_m_per_s, 5.0)
    assert soils[4].ks_m_per_s.shape == DEPTH.shape
