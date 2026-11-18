"""
Heatmaps of an emission table over (log n_H, log T), saved next to the table.

For each line in ``config.HEATMAPS["lines"]``: log10(eps/n_H^2) in colour with
contours of the volume emissivity log10 eps [erg cm^-3 s^-1].  For each ratio
in ``config.HEATMAPS["ratios"]``: log10(num/den) in a diverging colour map,
with contours at the ratio values in ``marks`` (e.g. observed ones).  Cells
where a line is at the table's floor (eps ~ 0) are grey.

    python -m emtab.heatmaps                   # the configured table
    python -m emtab.heatmaps path/to/table.h5  # any table
"""

from __future__ import annotations

if not __package__:  # run as a script (python emtab/<module>.py): import from the repo root
    import os
    import sys
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    __package__ = "emtab"

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

UNITS_LABEL = r"erg cm$^3$ s$^{-1}$"


def load(path):
    """(log_nH, log_T, {key: log eps/n_H^2}, {key: wavelength}, attrs) of a table."""
    with h5py.File(path, "r") as f:
        lines = {k: d[:] for k, d in f["lines"].items()}
        waves = {k: float(d.attrs["wavelength_A"]) for k, d in f["lines"].items()}
        return f["axes/log_nH"][:], f["axes/log_T"][:], lines, waves, dict(f.attrs)


def _title(attrs) -> str:
    return (f"{attrs.get('radiation', '')}, Z = {attrs.get('metallicity_Zsun', '?')} Z$_\\odot$"
            f"  ({attrs.get('source', '')})")


def _axes(ax):
    ax.set_xlabel(r"log $n_{\rm H}$ [cm$^{-3}$]")
    ax.set_ylabel(r"log $T$ [K]")


def plot_line(log_nH, log_T, data, key, wavelength, attrs, dynamic_range, path, dpi):
    """Heatmap of one line's log eps/n_H^2 with log eps contours."""
    floor = float(attrs.get("log_floor", -np.inf))
    vmax = float(data.max())
    masked = np.ma.masked_where(data <= floor, data)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.75")

    fig, ax = plt.subplots(figsize=(7, 5.2), layout="constrained")
    im = ax.pcolormesh(log_nH, log_T, masked.T, shading="nearest", cmap=cmap,
                       vmin=vmax - dynamic_range, vmax=vmax)
    fig.colorbar(im, ax=ax, extend="min",
                 label=rf"log $\varepsilon / n_{{\rm H}}^2$ [{UNITS_LABEL}]")

    log_eps = np.ma.masked_where(data <= floor, data + 2 * log_nH[:, None])
    levels = np.arange(np.floor(log_eps.min()), np.ceil(log_eps.max()) + 1, 2.0)
    cs = ax.contour(log_nH, log_T, log_eps.T, levels=levels, colors="w",
                    linewidths=0.6, alpha=0.8)
    ax.clabel(cs, fmt="%d", fontsize=7)

    _axes(ax)
    ax.set_title(f"{key} ({wavelength:.0f} Å): colour log ε/n$_H^2$, "
                 f"contours log ε [erg cm$^{{-3}}$ s$^{{-1}}$]\n{_title(attrs)}", fontsize=9)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_ratio(log_nH, log_T, num, den, num_key, den_key, marks, attrs, path, dpi):
    """Heatmap of log10(num/den) with contours at the ratio values `marks`."""
    floor = float(attrs.get("log_floor", -np.inf))
    ratio = np.ma.masked_where((num <= floor) | (den <= floor), num - den)
    lim = min(float(np.abs(ratio).max()), 4.0)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.75")

    fig, ax = plt.subplots(figsize=(7, 5.2), layout="constrained")
    im = ax.pcolormesh(log_nH, log_T, ratio.T, shading="nearest", cmap=cmap,
                       vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=ax, extend="both", label=f"log ({num_key} / {den_key})")
    if marks:
        cs = ax.contour(log_nH, log_T, ratio.T, levels=np.log10(sorted(marks)),
                        colors="k", linewidths=1.2, linestyles="solid")
        ax.clabel(cs, fmt={np.log10(m): f"{m:g}" for m in marks}, fontsize=8)
    _axes(ax)
    marks_text = f", contours at {', '.join(f'{m:g}' for m in marks)}" if marks else ""
    ax.set_title(f"{num_key} / {den_key}{marks_text}\n{_title(attrs)}", fontsize=9)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def draw_heatmaps(path, settings: dict) -> list[Path]:
    """Draw the configured heatmaps of the table at `path`; returns the image paths."""
    path = Path(path)
    log_nH, log_T, lines, waves, attrs = load(path)
    dpi = settings.get("dpi", 150)
    saved = []
    for key in settings.get("lines", []):
        if key not in lines:
            print(f"skipping {key}: not in table (has {list(lines)})")
            continue
        out = path.with_name(f"{path.stem}_{key}.png")
        plot_line(log_nH, log_T, lines[key], key, waves[key], attrs,
                  settings.get("dynamic_range", 10.0), out, dpi)
        saved.append(out)
    for r in settings.get("ratios", []):
        num, den = r["num"], r["den"]
        if num not in lines or den not in lines:
            print(f"skipping {num}/{den}: not in table (has {list(lines)})")
            continue
        out = path.with_name(f"{path.stem}_{num}_{den}.png")
        plot_ratio(log_nH, log_T, lines[num], lines[den], num, den,
                   r.get("marks", []), attrs, out, dpi)
        saved.append(out)
    return saved


def main(argv=None):
    import config as cfg

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", nargs="?",
                        help="emission table .h5 (default <OUTPUT_DIR>/<TABLE_NAME>.h5)")
    args = parser.parse_args(argv)
    if args.table:
        table = Path(args.table)
    else:
        from .run_grid import env_path
        table = env_path("OUTPUT_DIR") / f"{cfg.TABLE_NAME}.h5"
    for p in draw_heatmaps(table, cfg.HEATMAPS):
        print(f"saved {p}")


if __name__ == "__main__":
    main()
