"""
Cloudy input files for the grid points of an emission table.

Each point is one optically thin zone (1 cm thick) at fixed n_H and T in the
configured radiation field, run twice (`iterate`) so the line transfer uses
the thin optical depths; Cloudy saves the linear volume emissivity
[erg cm^-3 s^-1] of each line to <name>.ems and the gas-phase element
densities (log10 n_X [cm^-3]) to <name>.abn.

    python -m emtab.cloudy_input              # print the grid size and one input
    python -m emtab.cloudy_input -4 5.5       # input for log nH = -4, log T = 5.5
"""

from __future__ import annotations

if not __package__:  # run as a script (python emtab/<module>.py): import from the repo root
    import os
    import sys
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    __package__ = "emtab"

import argparse

import numpy as np


def axis(start: float, stop: float, step: float) -> np.ndarray:
    """Evenly spaced values from start to stop inclusive, rounded to drop float noise."""
    n = int(round((stop - start) / step)) + 1
    return np.round(start + step * np.arange(n), 6)


def grid(cfg) -> tuple[np.ndarray, np.ndarray]:
    """(log_nH, log_T) axes of the table."""
    return axis(**cfg.GRID["log_nH"]), axis(**cfg.GRID["log_T"])


def point_name(i: int, j: int) -> str:
    """File stem of grid point (i along log_nH, j along log_T)."""
    return f"p_{i:03d}_{j:03d}"


def cloudy_line(label: str) -> str:
    """Line label as Cloudy c25 parses it: species quoted, wavelength after,
    e.g. "H  1 6562.81A" -> '"H  1" 6562.81A'."""
    species, wavelength = label.rsplit(" ", 1)
    return f'"{species}" {wavelength}'


def make_input(log_nH: float, log_T: float, name: str, cfg) -> str:
    """Cloudy input text for one grid point, saving emissivities to <name>.ems."""
    rad = cfg.RADIATION
    z = rad["redshift"]
    lines = [f"title lognH={log_nH:.4f} logT={log_T:.4f}"]
    if rad.get("uvb"):
        lines.append(f"table {rad['uvb']} redshift {z}")
    if rad.get("cmb"):
        lines.append(f"cmb redshift {z}")
    if rad.get("cosmic_rays"):
        lines.append("cosmic rays background")
    lines += [
        f"hden {log_nH:.4f}",
        f"constant temperature {log_T:.4f}",  # values <= 10 are log10 T
        f"metals {cfg.METALS_LOG:.4f} log",
        "set dr 0",                            # 1 cm zone: optically thin
        "stop zone 1",
        "iterate",                             # 2nd pass uses the thin optical depths
        f'save line emissivity "{name}.ems" last',
        *(cloudy_line(label) for label, _ in cfg.LINES.values()),
        "end of lines",
        f'save abundances "{name}.abn" last',  # for X_H
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_nH", nargs="?", type=float, help="log10 n_H [cm^-3]")
    parser.add_argument("log_T", nargs="?", type=float, help="log10 T [K]")
    args = parser.parse_args(argv)

    import config as cfg
    log_nH, log_T = grid(cfg)
    print(f"# grid: {log_nH.size} x {log_T.size} = {log_nH.size * log_T.size} points; "
          f"log_nH {log_nH[0]}..{log_nH[-1]}, log_T {log_T[0]}..{log_T[-1]}")
    n = log_nH[0] if args.log_nH is None else args.log_nH
    t = log_T[0] if args.log_T is None else args.log_T
    print(make_input(n, t, "example", cfg), end="")


if __name__ == "__main__":
    main()
