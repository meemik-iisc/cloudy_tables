"""
Collect the Cloudy runs of a table grid into an emission table file.

Reads <OUTPUT_DIR>/<TABLE_NAME>/p_iii_jjj.{ems,abn} for every grid point,
converts the emissivities to log10(eps/n_H^2), works out the hydrogen mass
fraction X_H from the abundances, and writes <OUTPUT_DIR>/<TABLE_NAME>.h5
(format v1, see table_io), validates it and draws the heatmaps of
config.HEATMAPS next to it (see heatmaps).

    python -m emtab.collect
"""

from __future__ import annotations

if not __package__:  # run as a script (python emtab/<module>.py): import from the repo root
    import os
    import sys
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    __package__ = "emtab"

import argparse
import re

import numpy as np

from .cloudy_input import grid, point_name
from .heatmaps import draw_heatmaps
from .run_grid import env_path, run_dir
from .table_io import summary, validate, write_table

# Stored log10(eps/n_H^2) is clipped below at this (also where eps = 0); far
# below any line that matters (Ha in warm gas is ~ -24.6).
LOG_FLOOR = -50.0

# Atomic weights (amu) under Cloudy's 4-letter element names, H..Zn.
ATOMIC_WEIGHT = {
    "H": 1.008, "HELI": 4.0026, "LITH": 6.94, "BERY": 9.0122, "BORO": 10.81,
    "CARB": 12.011, "NITR": 14.007, "OXYG": 15.999, "FLUO": 18.998, "NEON": 20.180,
    "SODI": 22.990, "MAGN": 24.305, "ALUM": 26.982, "SILI": 28.085, "PHOS": 30.974,
    "SULP": 32.06, "CHLO": 35.45, "ARGO": 39.948, "POTA": 39.098, "CALC": 40.078,
    "SCAN": 44.956, "TITA": 47.867, "VANA": 50.942, "CHRO": 51.996, "MANG": 54.938,
    "IRON": 55.845, "COBA": 58.933, "NICK": 58.693, "COPP": 63.546, "ZINC": 65.38,
}


def _norm(label: str) -> str:
    """Line label with runs of spaces collapsed, for matching .ems headers."""
    return re.sub(r"\s+", " ", label.strip())


def read_ems(path, labels: list[str]) -> np.ndarray:
    """Emissivities [erg cm^-3 s^-1] of `labels` from a Cloudy .ems file (last row)."""
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [r for r in f.read().splitlines() if r.strip()]
    if not rows:
        raise ValueError(f"{path}: no data rows")
    columns = {_norm(h): k for k, h in enumerate(header)}
    values = rows[-1].split("\t")
    try:
        return np.array([float(values[columns[_norm(lab)]]) for lab in labels])
    except KeyError as e:
        raise ValueError(f"{path}: line {e} not in header {header[1:]}") from None


def read_abundances(path) -> dict[str, float]:
    """Element -> log10 n_X [cm^-3] from a Cloudy `save abundances` file."""
    with open(path) as f:
        header = f.readline().lstrip("#").replace("abund", "", 1).split()
        values = [r for r in f.read().splitlines() if r.strip()][-1].split()
    return dict(zip(header, map(float, values)))


def hydrogen_mass_fraction(log_abund: dict[str, float]) -> float:
    """X_H = m_H n_H / sum_X m_X n_X from log10 n_X (any common scale)."""
    unknown = set(log_abund) - set(ATOMIC_WEIGHT)
    if unknown:
        raise ValueError(f"no atomic weight for {sorted(unknown)}")
    ref = log_abund["H"]
    total = sum(ATOMIC_WEIGHT[el] * 10.0 ** (a - ref) for el, a in log_abund.items())
    return ATOMIC_WEIGHT["H"] / total


def cloudy_version(out_path) -> str:
    """'Cloudy <version>' from the banner of a Cloudy .out file."""
    with open(out_path) as f:
        for line in f:
            m = re.search(r"Cloudy\s+(\d+\.\d+)", line)
            if m:
                return f"Cloudy {m.group(1)}"
    return "Cloudy"


def radiation_text(rad: dict) -> str:
    parts = []
    if rad.get("uvb"):
        parts.append(f"{rad['uvb']} z={rad['redshift']}")
    if rad.get("cmb"):
        parts.append("CMB")
    if rad.get("cosmic_rays"):
        parts.append("CR background")
    return " + ".join(parts) or "none"


def collect(cfg):
    """Read every grid point; returns (log_nH, log_T, {key: log eps/n_H^2}, X_H, source)."""
    folder = run_dir(cfg)
    log_nH, log_T = grid(cfg)
    keys = list(cfg.LINES)
    labels = [cfg.LINES[k][0] for k in keys]
    eps = np.empty((log_nH.size, log_T.size, len(keys)))
    missing, x_h = [], set()
    for i in range(log_nH.size):
        for j in range(log_T.size):
            name = point_name(i, j)
            try:
                eps[i, j] = read_ems(folder / f"{name}.ems", labels)
                x_h.add(round(hydrogen_mass_fraction(read_abundances(folder / f"{name}.abn")), 6))
            except FileNotFoundError:
                missing.append(name)
    if missing:
        raise SystemExit(f"{len(missing)} grid points have no results (e.g. {missing[:5]}); "
                         "run `python -m emtab.run_grid` first")
    if len(x_h) != 1:
        raise SystemExit(f"abundances differ between grid points: X_H in {sorted(x_h)}")
    nH2 = (10.0 ** log_nH)[:, None] ** 2
    with np.errstate(divide="ignore"):
        lines = {k: np.maximum(np.log10(eps[..., n] / nH2), LOG_FLOOR)
                 for n, k in enumerate(keys)}
    source = cloudy_version(folder / f"{point_name(0, 0)}.out")
    return log_nH, log_T, lines, x_h.pop(), source


def main(argv=None):
    import config as cfg

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)

    log_nH, log_T, lines, x_h, source = collect(cfg)
    path = env_path("OUTPUT_DIR") / f"{cfg.TABLE_NAME}.h5"
    write_table(
        path, log_nH, log_T, lines,
        wavelengths={k: w for k, (_, w) in cfg.LINES.items()},
        labels={k: lab for k, (lab, _) in cfg.LINES.items()},
        X_H=x_h, log_floor=LOG_FLOOR,
        source=source,
        radiation=radiation_text(cfg.RADIATION),
        metallicity_Zsun=cfg.METALLICITY_ZSUN,
        abundances=f"Cloudy default solar, metals scaled by {cfg.METALLICITY_ZSUN}",
    )
    problems = validate(path)
    if problems:
        raise SystemExit("written table is INVALID:\n" + "\n".join(problems))
    print(summary(path))
    for p in draw_heatmaps(path, cfg.HEATMAPS):
        print(f"saved {p}")


if __name__ == "__main__":
    main()
