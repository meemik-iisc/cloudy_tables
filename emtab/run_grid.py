"""
Run Cloudy over every (n_H, T) point of the table grid.

Inputs and outputs go in <OUTPUT_DIR>/<TABLE_NAME>/ as p_iii_jjj.{in,out,ems,abn}
(i along log_nH, j along log_T).  Points that already finished with the same
input are skipped, so an interrupted run resumes where it stopped, and points
whose input changed (edited config) are rerun; failed points are listed in
failed.txt there.

    python -m emtab.run_grid                 # whole grid
    python -m emtab.run_grid -w 4            # with 4 parallel Cloudy runs
    python -m emtab.run_grid --point 18 0    # one point (here log nH = 0, log T = 4)
    python -m emtab.run_grid --force         # rerun finished points too
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if not __package__:  # run as a script: python emtab/run_grid.py
    import sys
    sys.path[0] = str(Path(__file__).resolve().parent.parent)
    __package__ = "emtab"

from .cloudy_input import grid, make_input, point_name

ROOT = Path(__file__).resolve().parent.parent
OK_MARKER = "Cloudy exited OK"


def load_env(path: Path) -> None:
    """Set KEY=VALUE lines of `path` (# comments allowed) as environment
    variables, unless already set in the shell."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.split(" #", 1)[0].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (s.strip() for s in line.split("=", 1))
        os.environ.setdefault(key, value.strip("'\""))


def env_path(name: str) -> Path:
    """Path from environment variable `name` (from .env or the shell), ~ and $VARS expanded."""
    load_env(ROOT / ".env")
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set; copy .env.example to .env and set it")
    return Path(os.path.expandvars(os.path.expanduser(value)))


def run_dir(cfg) -> Path:
    """Folder holding the Cloudy runs of the configured table."""
    return env_path("OUTPUT_DIR") / cfg.TABLE_NAME


def is_done(folder: Path, name: str, text: str) -> bool:
    """True if point `name` ran input `text` to the end and saved its emissivities
    and abundances; a run of a different input (changed settings) is not done."""
    inp, out = folder / f"{name}.in", folder / f"{name}.out"
    if not (inp.is_file() and inp.read_text() == text):
        return False
    saved = [folder / f"{name}.ems", folder / f"{name}.abn"]
    if not (out.is_file() and all(f.is_file() and f.stat().st_size > 0 for f in saved)):
        return False
    with open(out, "rb") as f:  # the marker is on the last lines
        f.seek(max(0, out.stat().st_size - 2000))
        return OK_MARKER in f.read().decode(errors="replace")


def run_point(cloudy: Path, folder: Path, name: str, text: str) -> tuple[str, bool, float]:
    """Write <name>.in, run Cloudy on it; returns (name, success, seconds)."""
    (folder / f"{name}.in").write_text(text)
    for ext in ("ems", "abn"):
        (folder / f"{name}.{ext}").unlink(missing_ok=True)
    start = time.time()
    subprocess.run([str(cloudy), "-r", name], cwd=folder,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return name, is_done(folder, name, text), time.time() - start


def main(argv=None):
    import config as cfg

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-w", "--workers", type=int, default=cfg.N_WORKERS,
                        help=f"parallel Cloudy runs (default {cfg.N_WORKERS})")
    parser.add_argument("--point", type=int, nargs=2, metavar=("I", "J"),
                        help="run only grid point (i along log_nH, j along log_T)")
    parser.add_argument("--force", action="store_true", help="rerun finished points too")
    args = parser.parse_args(argv)

    cloudy = env_path("CLOUDY_EXE")
    if not cloudy.is_file():
        raise SystemExit(f"Cloudy executable not found: {cloudy}")
    folder = run_dir(cfg)
    folder.mkdir(parents=True, exist_ok=True)

    log_nH, log_T = grid(cfg)
    points = [(i, j) for i in range(log_nH.size) for j in range(log_T.size)]
    if args.point:
        i, j = args.point
        if not (0 <= i < log_nH.size and 0 <= j < log_T.size):
            raise SystemExit(f"point out of range: grid is {log_nH.size} x {log_T.size}")
        points = [(i, j)]
    def text(i, j):
        return make_input(log_nH[i], log_T[j], point_name(i, j), cfg)

    todo = [(i, j) for i, j in points
            if args.force or not is_done(folder, point_name(i, j), text(i, j))]
    print(f"{cfg.TABLE_NAME}: {len(points)} points, {len(points) - len(todo)} already done, "
          f"running {len(todo)} with {args.workers} workers in {folder}")

    failed = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_point, cloudy, folder, point_name(i, j), text(i, j))
                for i, j in todo]
        for k, job in enumerate(as_completed(jobs), 1):
            name, ok, secs = job.result()
            if not ok:
                failed.append(name)
            print(f"[{k}/{len(todo)}] {name} {'ok' if ok else 'FAILED'} ({secs:.1f} s)",
                  flush=True)

    if not args.point:  # a one-point run leaves the grid's list alone
        (folder / "failed.txt").write_text("".join(f"{n}\n" for n in sorted(failed)))
    print(f"done in {time.time() - start:.0f} s; {len(failed)} failed"
          + (f" (see {folder / 'failed.txt'})" if failed and not args.point else ""))


if __name__ == "__main__":
    main()
