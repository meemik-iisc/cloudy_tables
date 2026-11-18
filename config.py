"""
Settings for one emission table: the (n_H, T) grid, the gas and radiation
field Cloudy is run with, and the lines saved.  Machine-specific paths
(Cloudy executable, output folder) go in .env, not here.
"""

import math

# Grid axes, inclusive of both ends: log10 n_H [cm^-3] and log10 T [K].
GRID = dict(
    log_nH=dict(start=-4.5, stop=2.0, step=0.25),
    log_T=dict(start=4.0, stop=7.5, step=0.1),
)

# Radiation field and background sources.
RADIATION = dict(
    uvb="KS19",          # Cloudy `table <uvb> redshift <z>`; None = no UV background
    redshift=0.96,
    cmb=True,            # `cmb redshift <z>`
    cosmic_rays=True,    # `cosmic rays background`
)

# Metallicity relative to Cloudy's default solar abundances.
METALLICITY_ZSUN = 0.2
METALS_LOG = math.log10(METALLICITY_ZSUN)

# Name of the table, made from the settings above so that a new metallicity or
# radiation field gets its own files: <OUTPUT_DIR>/<TABLE_NAME>.h5, with the
# Cloudy runs in <OUTPUT_DIR>/<TABLE_NAME>/.  Set a fixed string to override.
TABLE_NAME = (f"emis_{RADIATION['uvb'] or 'noUVB'}_z{RADIATION['redshift']:g}"
              f"_Z{METALLICITY_ZSUN:g}")

# Table key -> (Cloudy c25 line label, rest wavelength in Angstrom, air).
# Only single lines; doublet sums are formed by the reader.
LINES = {
    "Ha":        ("H  1 6562.81A", 6562.81),
    "Hb":        ("H  1 4861.32A", 4861.32),
    "OIII_5007": ("O  3 5006.84A", 5006.84),
    "OIII_4959": ("O  3 4958.91A", 4958.91),
    "NII_6584":  ("N  2 6583.45A", 6583.45),
    "SII_6716":  ("S  2 6716.44A", 6716.44),
    "SII_6731":  ("S  2 6730.82A", 6730.82),
    "SIII_9069": ("S  3 9068.62A", 9068.62),
    "SIII_9531": ("S  3 9530.62A", 9530.62),
}

# Parallel Cloudy runs (each is single-threaded, a few hundred MB).
N_WORKERS = 8

# Heatmaps drawn by `python -m emtab.heatmaps` (and after every collect), saved
# next to the table as <TABLE_NAME>_<line>.png and <TABLE_NAME>_<num>_<den>.png.
HEATMAPS = dict(
    lines=["Ha", "OIII_5007"],         # maps of log eps/n_H^2
    ratios=[                           # maps of log(num/den); marks = contours drawn
        dict(num="OIII_5007", den="Ha", marks=[0.6, 1.5]),  # observed in RBH-1
    ],
    dynamic_range=10.0,                # colour range below the maximum, in dex
    dpi=150,
)
