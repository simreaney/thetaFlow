#!/usr/bin/env python3
"""thetaFlow launcher: pick a model mode with one word.

Usage
-----
python thetaflow.py column     [options]   # bare soil column (no root uptake)
python thetaflow.py vegetation [options]   # soil column with root-zone uptake
python thetaflow.py 2d         [options]   # 2-D spatial model on a DEM

Options after the mode are passed to the underlying model:

* ``column`` / ``vegetation`` → ``simulate_soil_column.py``
* ``2d`` → ``simulate_spatial.py``

Run ``python thetaflow.py <mode> --help`` for the options of each mode.

Examples
--------
python thetaflow.py column --forcing-csv forcing_example.csv
python thetaflow.py vegetation --vegetation wheat
python thetaflow.py 2d --dem dem.tif --forcing forcing_example.csv
python thetaflow.py 2d --synthetic
"""

from __future__ import annotations

import sys
from typing import Optional

MODES = {
    "column": "1-D soil column without vegetation",
    "vegetation": "1-D soil column with root-zone water uptake (default: grass)",
    "2d": "2-D spatial model; topography from a DEM GeoTIFF (--dem)",
}
ALIASES = {"soil": "column", "veg": "vegetation", "spatial": "2d"}
DEFAULT_VEGETATION = "grass"


def _usage() -> str:
    lines = ["usage: python thetaflow.py {column,vegetation,2d} [options]", "", "modes:"]
    lines += [f"  {name:<11} {desc}" for name, desc in MODES.items()]
    lines += ["", "Run 'python thetaflow.py <mode> --help' for the options of each mode."]
    return "\n".join(lines)


def _has_option(argv: list[str], option: str) -> bool:
    return any(a == option or a.startswith(option + "=") for a in argv)


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return

    mode, rest = ALIASES.get(argv[0].lower(), argv[0].lower()), argv[1:]
    if mode not in MODES:
        sys.exit(f"Unknown mode {argv[0]!r}.\n\n{_usage()}")

    if mode == "column":
        if _has_option(rest, "--vegetation"):
            sys.exit("The 'column' mode runs without vegetation; use 'vegetation' mode instead.")
        import simulate_soil_column
        simulate_soil_column.main(rest)
    elif mode == "vegetation":
        if not _has_option(rest, "--vegetation") and not _has_option(rest, "--list-vegetation"):
            print(f"No --vegetation given; using '{DEFAULT_VEGETATION}'.")
            rest += ["--vegetation", DEFAULT_VEGETATION]
        import simulate_soil_column
        simulate_soil_column.main(rest)
    else:
        import simulate_spatial
        simulate_spatial.main(rest)


if __name__ == "__main__":
    main()
