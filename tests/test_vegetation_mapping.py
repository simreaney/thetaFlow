import json
from pathlib import Path

import numpy as np
import pytest

from spatial_utils import (
    build_friction_factor_grid,
    load_code_map,
    load_vegetation_map,
    load_vegetation_types,
)

ROOT = Path(__file__).resolve().parents[1]
VEG_JSON = ROOT / "vegetation_types.json"


def test_json_entries_take_precedence_over_builtin_library():
    veg = {"vegetation_types": [{"name": "grass", "rooting_depth_m": 0.9, "pet_scale": 0.5}]}
    lib = load_vegetation_types(veg)
    assert lib["grass"].rooting_depth_m == pytest.approx(0.9)
    assert lib["grass"].pet_scale == pytest.approx(0.5)


def test_missing_fields_fall_back_to_library():
    lib = load_vegetation_types({"vegetation_types": [{"name": "wheat"}]})
    assert lib["wheat"].rooting_depth_m == pytest.approx(1.0)


def test_json_only_types_are_available():
    lib = load_vegetation_types(VEG_JSON)
    assert {"shrubland", "open_water"} <= set(lib)


def test_default_vegetation_for_unmapped_and_nodata_codes():
    codes = np.array([[1, 2], [99, -1]])
    grid = load_vegetation_map(codes, VEG_JSON, {1: "grass", 2: "urban"}, default_name="moorland")
    assert grid[0][0].name == "grass" and grid[0][1].name == "urban"
    assert grid[1][0].name == "moorland" and grid[1][1].name == "moorland"
    # Without a default the previous behaviour (no vegetation) is kept
    assert load_vegetation_map(codes, VEG_JSON, {1: "grass"})[1][0] is None


def test_friction_grid_uses_default_and_overrides():
    codes = np.array([[1, 5]])
    f = build_friction_factor_grid(codes, VEG_JSON, {1: "grass"}, overrides={"grass": 0.7},
                                   default_name="urban")
    np.testing.assert_allclose(f, [[0.7, 0.1]])


@pytest.mark.parametrize("path", sorted((ROOT / "landcover_maps").glob("*.json")), ids=lambda p: p.name)
def test_preset_maps_only_use_known_vegetation_names(path):
    names = {e["name"] for e in json.loads(VEG_JSON.read_text())["vegetation_types"]}
    mapping = load_code_map(path)
    assert mapping
    assert set(mapping.values()) <= names


def test_corine_preset_covers_codes_and_grid_codes():
    mapping = load_code_map(ROOT / "landcover_maps" / "corine.json")
    assert mapping[311] == mapping[23] == "broadleaf_woodland"
    assert mapping[523] == mapping[44] == "open_water"
