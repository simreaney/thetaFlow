#!/usr/bin/env python3
"""Depth-varying soil hydraulic properties for thetaFlow spatial runs.

Two ways of describing how van Genuchten–Mualem properties change with depth
are supported, both producing per-cell :class:`SoilProperties` whose fields
are arrays with one value per column layer:

* **Horizon grids** – a set of rasters, each giving one property for one
  depth horizon (e.g. ``ks_0-5cm``, ``ks_5-15cm``, ``theta_s_0-30cm``).
  Values are assigned to layers stepwise or interpolated linearly between
  horizon mid-depths.
* **Regression** – per-property depth functions applied to the surface
  value ``p0`` taken from the soil map:

  ======== ==========================
  form     p(z)
  ======== ==========================
  exp      p0 · exp(−z / a)
  linear   p0 + a · z
  power    p0 · (1 + z) ^ a
  expr     any expression in z, p0, a
  ======== ==========================

  ``a`` may be a constant or a 2-D grid, and ``expr`` may also reference
  extra named grids.  Expressions are evaluated by a restricted AST walker
  (arithmetic, comparisons and a whitelist of NumPy functions only).

This module has no QGIS dependency so it can be used and tested standalone.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

from simulate_soil_column import SoilProperties

SOIL_PROPERTIES: tuple[str, ...] = (
    "theta_r",
    "theta_s",
    "alpha_per_m",
    "n",
    "ks_m_per_s",
    "pore_connectivity",
)

# Accepted spellings (lower case) → canonical SoilProperties field name
PROPERTY_ALIASES: dict[str, str] = {
    "theta_r": "theta_r",
    "thetar": "theta_r",
    "thr": "theta_r",
    "theta_s": "theta_s",
    "thetas": "theta_s",
    "ths": "theta_s",
    "porosity": "theta_s",
    "alpha": "alpha_per_m",
    "alpha_per_m": "alpha_per_m",
    "n": "n",
    "vg_n": "n",
    "ks": "ks_m_per_s",
    "ksat": "ks_m_per_s",
    "ks_m_per_s": "ks_m_per_s",
    "l": "pore_connectivity",
    "pore_connectivity": "pore_connectivity",
}

# Multiply a Ks value in the given unit by this factor to obtain m/s
KS_UNIT_FACTORS: dict[str, float] = {
    "m/s": 1.0,
    "cm/day": 1.0 / 100.0 / 86400.0,
    "mm/h": 1.0 / 1000.0 / 3600.0,
}

Grid = np.ndarray
Coefficient = Union[float, np.ndarray]


def canonical_property(name: str) -> str:
    """Map a property alias (case-insensitive) to its SoilProperties field name."""
    key = name.strip().lower()
    if key not in PROPERTY_ALIASES:
        raise ValueError(
            f"Unknown soil property '{name}'. Expected one of: "
            f"{', '.join(sorted(PROPERTY_ALIASES))}."
        )
    return PROPERTY_ALIASES[key]


# ---------------------------------------------------------------------------
# Horizon rasters
# ---------------------------------------------------------------------------

_HORIZON_NAME_RE = re.compile(
    r"^(?P<prop>[a-z_]+?)_(?P<top>\d+(?:\.\d+)?)-(?P<bottom>\d+(?:\.\d+)?)(?P<unit>cm|m)(?:_.*)?$",
    re.IGNORECASE,
)


def parse_horizon_name(name: str) -> Optional[tuple[str, float, float]]:
    """Parse ``<property>_<top>-<bottom><cm|m>[_suffix]`` layer names.

    Examples: ``ks_0-5cm``, ``theta_s_15-30cm``, ``alpha_0-0.3m``,
    ``ksat_5-15cm_mean`` (SoilGrids-style suffix).  Returns
    ``(property, top_m, bottom_m)`` or ``None`` if the name does not match or
    the property is not a hydraulic property.
    """
    m = _HORIZON_NAME_RE.match(name.strip())
    if not m:
        return None
    prop = PROPERTY_ALIASES.get(m.group("prop").lower())
    if prop is None:
        return None
    scale = 0.01 if m.group("unit").lower() == "cm" else 1.0
    top = float(m.group("top")) * scale
    bottom = float(m.group("bottom")) * scale
    if bottom <= top:
        raise ValueError(f"Horizon '{name}': bottom depth must be greater than top depth.")
    return prop, top, bottom


@dataclass
class Horizon:
    """One depth horizon with any subset of property grids (NaN = no data)."""

    top_m: float
    bottom_m: float
    grids: dict[str, Grid] = field(default_factory=dict)

    @property
    def mid_m(self) -> float:
        return 0.5 * (self.top_m + self.bottom_m)


def group_horizon_grids(items: Sequence[tuple[str, float, float, Grid]]) -> list[Horizon]:
    """Group ``(property, top_m, bottom_m, grid)`` items into :class:`Horizon` objects."""
    by_depth: dict[tuple[float, float], Horizon] = {}
    for prop, top, bottom, grid in items:
        prop = canonical_property(prop)
        key = (float(top), float(bottom))
        hz = by_depth.setdefault(key, Horizon(top_m=key[0], bottom_m=key[1]))
        if prop in hz.grids:
            raise ValueError(
                f"Duplicate '{prop}' grid for horizon {key[0]:g}–{key[1]:g} m."
            )
        hz.grids[prop] = np.asarray(grid, dtype=float)
    return sorted(by_depth.values(), key=lambda h: (h.top_m, h.bottom_m))


def base_grids_from_soil_list(
    soil_list: Sequence[Optional[SoilProperties]],
    shape: tuple[int, int],
    fallback: Optional[SoilProperties] = None,
) -> dict[str, Grid]:
    """Turn a row-major per-cell list of (scalar) SoilProperties into 2-D grids."""
    if fallback is None:
        fallback = next((sp for sp in soil_list if sp is not None), None)
    if fallback is None:
        raise ValueError("No soil properties available to build base grids.")
    grids: dict[str, Grid] = {}
    for prop in SOIL_PROPERTIES:
        values = [
            float(np.ravel(getattr(sp if sp is not None else fallback, prop))[0])
            for sp in soil_list
        ]
        grids[prop] = np.asarray(values, dtype=float).reshape(shape)
    return grids


def profiles_from_horizons(
    horizons: Sequence[Horizon],
    depth_m: np.ndarray,
    base_grids: dict[str, Grid],
    interp: str = "step",
) -> dict[str, np.ndarray]:
    """Build (nrows, ncols, nz) property profiles from horizon grids.

    * ``step``: each layer takes the value of the horizon whose top is the
      deepest one not below the layer depth (gaps use the horizon above;
      layers above the first horizon use the first one).
    * ``linear``: linear interpolation between horizon mid-depths.

    Layers below the deepest horizon take its values.  Properties with no
    horizon grid, and NaN (no-data) cells, fall back to *base_grids*.
    """
    if interp not in ("step", "linear"):
        raise ValueError("interp must be 'step' or 'linear'.")
    depth_m = np.asarray(depth_m, dtype=float)
    profiles: dict[str, np.ndarray] = {}

    for prop in SOIL_PROPERTIES:
        base = np.asarray(base_grids[prop], dtype=float)
        hz = [h for h in horizons if prop in h.grids]
        if not hz:
            profiles[prop] = np.repeat(base[:, :, None], depth_m.size, axis=2)
            continue

        stack = np.stack(
            [np.where(np.isfinite(h.grids[prop]), h.grids[prop], base) for h in hz],
            axis=-1,
        )  # (nrows, ncols, H)

        if interp == "step" or len(hz) == 1:
            tops = np.array([h.top_m for h in hz])
            idx = np.clip(np.searchsorted(tops, depth_m, side="right") - 1, 0, len(hz) - 1)
            profiles[prop] = stack[:, :, idx]
        else:
            mids = np.array([h.mid_m for h in hz])
            pos = np.interp(depth_m, mids, np.arange(len(hz), dtype=float))
            i0 = np.floor(pos).astype(int)
            i1 = np.minimum(i0 + 1, len(hz) - 1)
            w = pos - i0
            profiles[prop] = stack[:, :, i0] * (1.0 - w) + stack[:, :, i1] * w

    return profiles


# ---------------------------------------------------------------------------
# Regression with depth
# ---------------------------------------------------------------------------

REGRESSION_FORMS: tuple[str, ...] = ("exp", "linear", "power", "expr")


@dataclass
class RegressionRule:
    """Depth function for one property (see module docstring for the forms)."""

    prop: str
    form: str
    coefficient: Coefficient = 0.0
    expression: str = ""

    def __post_init__(self) -> None:
        self.prop = canonical_property(self.prop)
        self.form = self.form.strip().lower()
        if self.form not in REGRESSION_FORMS:
            raise ValueError(
                f"Unknown regression form '{self.form}' for {self.prop}; "
                f"expected one of {', '.join(REGRESSION_FORMS)}."
            )
        if self.form == "expr":
            if not self.expression.strip():
                raise ValueError(f"Regression for {self.prop}: 'expr' form needs an expression.")
            SafeExpression(self.expression)  # validate early


_SAFE_FUNCTIONS = {
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "min": np.minimum,
    "max": np.maximum,
    "clip": np.clip,
    "where": np.where,
    "tanh": np.tanh,
}
_SAFE_CONSTANTS = {"pi": np.pi, "e": np.e}
_BIN_OPS = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
    ast.Mod: np.mod,
}
_CMP_OPS = {
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
}


class SafeExpression:
    """A restricted, vectorised arithmetic expression.

    Allowed: numeric literals, variable names supplied at evaluation time,
    ``pi``/``e``, ``+ - * / ** %``, unary ``+``/``-``, single comparisons, and
    calls to ``exp log log10 sqrt abs minimum maximum min max clip where tanh``.
    Anything else (attribute access, subscripts, keyword arguments, lambdas,
    other names, ...) is rejected when the expression is parsed.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        try:
            tree = ast.parse(text.strip(), mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid expression '{text}': {exc.msg}") from exc
        self._tree = tree
        self.names: set[str] = set()
        self._check(tree.body)

    def _check(self, node: ast.AST) -> None:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ValueError(f"Only numeric constants are allowed in '{self.text}'.")
        elif isinstance(node, ast.Name):
            if node.id in _SAFE_FUNCTIONS:
                raise ValueError(f"Function '{node.id}' must be called in '{self.text}'.")
            if node.id not in _SAFE_CONSTANTS:
                self.names.add(node.id)
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BIN_OPS:
                raise ValueError(f"Operator not allowed in '{self.text}'.")
            self._check(node.left)
            self._check(node.right)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                raise ValueError(f"Operator not allowed in '{self.text}'.")
            self._check(node.operand)
        elif isinstance(node, ast.Compare):
            if len(node.ops) != 1 or type(node.ops[0]) not in _CMP_OPS:
                raise ValueError(f"Only single comparisons are allowed in '{self.text}'.")
            self._check(node.left)
            self._check(node.comparators[0])
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCTIONS:
                raise ValueError(
                    f"Only these functions are allowed in expressions: "
                    f"{', '.join(sorted(_SAFE_FUNCTIONS))}."
                )
            if node.keywords:
                raise ValueError(f"Keyword arguments are not allowed in '{self.text}'.")
            for arg in node.args:
                self._check(arg)
        else:
            raise ValueError(
                f"Unsupported syntax ({type(node).__name__}) in expression '{self.text}'."
            )

    def evaluate(self, variables: dict[str, object]) -> np.ndarray:
        missing = self.names.difference(variables)
        if missing:
            raise ValueError(
                f"Unknown name(s) {', '.join(sorted(missing))} in expression '{self.text}'. "
                f"Available: {', '.join(sorted(variables))}."
            )
        with np.errstate(all="ignore"):
            return np.asarray(self._eval(self._tree.body, variables), dtype=float)

    def _eval(self, node: ast.AST, env: dict[str, object]):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return _SAFE_CONSTANTS[node.id] if node.id in _SAFE_CONSTANTS else env[node.id]
        if isinstance(node, ast.BinOp):
            return _BIN_OPS[type(node.op)](self._eval(node.left, env), self._eval(node.right, env))
        if isinstance(node, ast.UnaryOp):
            value = self._eval(node.operand, env)
            return np.negative(value) if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Compare):
            return _CMP_OPS[type(node.ops[0])](
                self._eval(node.left, env), self._eval(node.comparators[0], env)
            )
        if isinstance(node, ast.Call):
            return _SAFE_FUNCTIONS[node.func.id](*(self._eval(a, env) for a in node.args))
        raise ValueError(f"Unsupported syntax in '{self.text}'.")  # pragma: no cover


def _as_layer_field(value: Coefficient) -> Union[float, np.ndarray]:
    """Broadcast a scalar or (nrows, ncols) grid against (nrows, ncols, nz)."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 2:
        return arr[:, :, None]
    raise ValueError("Regression coefficients must be scalars or 2-D grids.")


def profiles_from_regression(
    rules: Sequence[RegressionRule],
    depth_m: np.ndarray,
    base_grids: dict[str, Grid],
    variables: Optional[dict[str, Grid]] = None,
) -> dict[str, np.ndarray]:
    """Build (nrows, ncols, nz) property profiles from depth regressions.

    Properties without a rule are uniform with depth (the base grid value).
    Expression variables: ``z`` (layer depth, m), ``p0`` (surface value),
    ``a`` (the rule's coefficient) and every name in *variables*.
    """
    depth_m = np.asarray(depth_m, dtype=float)
    z = depth_m[None, None, :]
    by_prop: dict[str, RegressionRule] = {}
    for rule in rules:
        if rule.prop in by_prop:
            raise ValueError(f"More than one regression rule for '{rule.prop}'.")
        by_prop[rule.prop] = rule

    extra = {k: _as_layer_field(v) for k, v in (variables or {}).items()}
    for reserved in ("z", "p0", "a"):
        if reserved in extra:
            raise ValueError(f"Variable name '{reserved}' is reserved.")

    profiles: dict[str, np.ndarray] = {}
    for prop in SOIL_PROPERTIES:
        base = np.asarray(base_grids[prop], dtype=float)
        p0 = base[:, :, None]
        rule = by_prop.get(prop)
        if rule is None:
            profiles[prop] = np.repeat(p0, depth_m.size, axis=2)
            continue
        a = _as_layer_field(rule.coefficient)
        with np.errstate(all="ignore"):
            if rule.form == "exp":
                out = p0 * np.exp(-z / a)
            elif rule.form == "linear":
                out = p0 + a * z
            elif rule.form == "power":
                out = p0 * (1.0 + z) ** a
            else:
                out = SafeExpression(rule.expression).evaluate({"z": z, "p0": p0, "a": a, **extra})
        profiles[prop] = np.broadcast_to(out, base.shape + (depth_m.size,)).astype(float)
    return profiles


# ---------------------------------------------------------------------------
# Validation and conversion
# ---------------------------------------------------------------------------

def validate_profiles(
    profiles: dict[str, np.ndarray],
    base_grids: Optional[dict[str, Grid]] = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Clip profiles to physically valid ranges.

    Non-finite values are replaced by the base-grid value (or the property's
    median) and counted.  Bounds: ``0.01 ≤ theta_s ≤ 1``,
    ``0 ≤ theta_r ≤ theta_s − 0.01``, ``n ≥ 1.01``, ``alpha ≥ 1e-4 m⁻¹``,
    ``1e-12 ≤ Ks ≤ 1 m/s``, ``−10 ≤ l ≤ 10``.

    Returns the corrected profiles and ``{property: n_values_changed}``.
    """
    out: dict[str, np.ndarray] = {}
    report: dict[str, int] = {}

    def _fix(prop: str, lo, hi) -> None:
        arr = np.array(profiles[prop], dtype=float)
        bad = ~np.isfinite(arr)
        if bad.any():
            if base_grids is not None:
                fill = np.broadcast_to(np.asarray(base_grids[prop], float)[:, :, None], arr.shape)
                arr[bad] = fill[bad]
            still = ~np.isfinite(arr)
            if still.any():
                finite = arr[np.isfinite(arr)]
                arr[still] = float(np.median(finite)) if finite.size else float(lo)
        clipped = np.clip(arr, lo, hi)
        report[prop] = int(np.count_nonzero(bad | (clipped != arr)))
        out[prop] = clipped

    _fix("theta_s", 0.01, 1.0)
    _fix("theta_r", 0.0, out["theta_s"] - 0.01)
    _fix("n", 1.01, np.inf)
    _fix("alpha_per_m", 1.0e-4, np.inf)
    _fix("ks_m_per_s", 1.0e-12, 1.0)
    _fix("pore_connectivity", -10.0, 10.0)
    return out, report


def soil_list_from_profiles(profiles: dict[str, np.ndarray]) -> list[SoilProperties]:
    """Convert (nrows, ncols, nz) profiles to a row-major per-cell list of
    SoilProperties whose fields are (nz,) arrays."""
    shape = profiles["theta_s"].shape
    nrows, ncols, nz = shape
    flat = {p: np.ascontiguousarray(profiles[p]).reshape(nrows * ncols, nz) for p in SOIL_PROPERTIES}
    return [
        SoilProperties(
            theta_r=flat["theta_r"][i].copy(),
            theta_s=flat["theta_s"][i].copy(),
            alpha_per_m=flat["alpha_per_m"][i].copy(),
            n=flat["n"][i].copy(),
            ks_m_per_s=flat["ks_m_per_s"][i].copy(),
            pore_connectivity=flat["pore_connectivity"][i].copy(),
        )
        for i in range(nrows * ncols)
    ]
