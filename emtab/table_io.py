"""
Write and check emission tables, format version 1.

One HDF5 file per table:

    attrs   format="emission_table", format_version=1, X_H, log_floor, source,
            radiation, metallicity_Zsun, abundances, created
    axes/   log_nH (cm^-3), log_T (K): 1D, strictly increasing;
            attrs axis_order = ["log_nH", "log_T"]
    lines/  <key>: (n_nH, n_T) log10(eps / n_H^2) [erg cm^3 s^-1];
            attrs wavelength_A, label (Cloudy label), units

eps is the line's volume emissivity [erg cm^-3 s^-1] and n_H the hydrogen
number density.  Values below `log_floor` (including eps = 0) are stored as
`log_floor`.  The reader in vis_athenak (emission/table.py) defines the
format; `validate` checks the same things.

    python -m emtab.table_io <table.h5>      # validate a table and print a summary
"""

from __future__ import annotations

if not __package__:  # run as a script (python emtab/<module>.py): import from the repo root
    import os
    import sys
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    __package__ = "emtab"

import argparse
from datetime import datetime, timezone

import h5py
import numpy as np

FORMAT, VERSION = "emission_table", 1
AXES = ["log_nH", "log_T"]
UNITS = "log10(erg cm^3 s^-1)"


def write_table(path, log_nH, log_T, lines: dict, wavelengths: dict, labels: dict,
                X_H: float, log_floor: float, **meta) -> None:
    """
    Write a table.  ``lines[key]`` is log10(eps/n_H^2) of shape
    (len(log_nH), len(log_T)); ``wavelengths``/``labels`` are keyed the same;
    ``meta`` (source, radiation, ...) goes into the file attributes.
    """
    with h5py.File(path, "w") as f:
        f.attrs["format"] = FORMAT
        f.attrs["format_version"] = VERSION
        f.attrs["X_H"] = float(X_H)
        f.attrs["log_floor"] = float(log_floor)
        f.attrs["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for k, v in meta.items():
            f.attrs[k] = v
        axes = f.create_group("axes")
        axes.attrs["axis_order"] = AXES
        axes["log_nH"] = np.asarray(log_nH, dtype=float)
        axes["log_T"] = np.asarray(log_T, dtype=float)
        grp = f.create_group("lines")
        for key, data in lines.items():
            d = grp.create_dataset(key, data=np.asarray(data, dtype=float))
            d.attrs["wavelength_A"] = float(wavelengths[key])
            d.attrs["label"] = labels[key]
            d.attrs["units"] = UNITS


def validate(path) -> list[str]:
    """Problems found in the table at ``path`` (empty if it is a valid v1 table)."""
    problems = []
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != FORMAT:
            return [f"format attribute is {f.attrs.get('format')!r}, not {FORMAT!r}"]
        if int(f.attrs.get("format_version", -1)) != VERSION:
            problems.append(f"format_version is {f.attrs.get('format_version')}, not {VERSION}")
        x_h = float(f.attrs.get("X_H", np.nan))
        if not 0 < x_h <= 1:
            problems.append(f"X_H = {x_h} is not in (0, 1]")
        if "axes" not in f or "lines" not in f:
            return problems + ["missing axes/ or lines/ group"]
        if [str(a) for a in f["axes"].attrs.get("axis_order", [])] != AXES:
            problems.append(f"axis_order is not {AXES}")
        shape = []
        for name in AXES:
            if name not in f["axes"]:
                problems.append(f"missing axes/{name}")
                continue
            a = f["axes"][name][:]
            shape.append(a.size)
            if a.ndim != 1 or a.size < 2 or not np.all(np.diff(a) > 0):
                problems.append(f"axes/{name} is not 1D and strictly increasing")
        if not f["lines"]:
            problems.append("no lines")
        for key, d in f["lines"].items():
            if list(d.shape) != shape:
                problems.append(f"lines/{key} has shape {d.shape}, axes give {tuple(shape)}")
            elif not np.all(np.isfinite(d[:])):
                problems.append(f"lines/{key} has non-finite values")
            for attr in ("wavelength_A", "label", "units"):
                if attr not in d.attrs:
                    problems.append(f"lines/{key} has no {attr!r} attribute")
    return problems


def summary(path) -> str:
    """Short text description of a table."""
    with h5py.File(path, "r") as f:
        nH, T = f["axes/log_nH"][:], f["axes/log_T"][:]
        out = [f"{path}", *(f"  {k} = {v}" for k, v in f.attrs.items()),
               f"  log_nH: {nH.size} values {nH[0]} .. {nH[-1]}",
               f"  log_T:  {T.size} values {T[0]} .. {T[-1]}",
               "  lines (log eps/n_H^2 min .. max):"]
        for key, d in f["lines"].items():
            v = d[:]
            out.append(f"    {key:10s} {d.attrs['wavelength_A']:8.2f} A  "
                       f"{v.min():7.2f} .. {v.max():7.2f}")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", help="emission table .h5 file")
    args = parser.parse_args(argv)
    problems = validate(args.table)
    print(summary(args.table) if not problems else "\n".join(["INVALID:", *problems]))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
