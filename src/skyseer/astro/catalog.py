"""Star catalogue download and plate-solver database generation.

The solver needs a pattern database built for the camera's field of view. Databases are
built once from the Hipparcos catalogue (ESA, 118,218 stars) and cached on disk.
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._vendor import tetra3
from .frames import radec_to_vec

log = logging.getLogger(__name__)

HIPPARCOS_URL = "https://cdsarc.cds.unistra.fr/ftp/cats/I/239/hip_main.dat"
DEFAULT_DATA_DIR = Path("data/astro")


def hipparcos_path(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    return data_dir / "hip_main.dat"


def ensure_hipparcos(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download the Hipparcos main catalogue (~51 MB) if it isn't cached yet."""
    path = hipparcos_path(data_dir)
    if path.exists() and path.stat().st_size > 50_000_000:
        return path
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    log.warning("Downloading Hipparcos catalogue from %s (~51 MB)...", HIPPARCOS_URL)
    with urllib.request.urlopen(HIPPARCOS_URL, timeout=120) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(path)
    return path


@dataclass(frozen=True)
class StarCatalog:
    hip: np.ndarray  # Hipparcos number
    ra: np.ndarray  # degrees, ICRS, at the requested epoch
    dec: np.ndarray
    mag: np.ndarray  # Johnson V

    def __len__(self) -> int:
        return len(self.hip)

    def vectors(self) -> np.ndarray:
        return radec_to_vec(self.ra, self.dec)

    def brighter_than(self, mag: float) -> StarCatalog:
        m = self.mag <= mag
        return StarCatalog(self.hip[m], self.ra[m], self.dec[m], self.mag[m])


def load_stars(
    max_mag: float = 7.0, epoch_year: float = 2026.0, data_dir: Path = DEFAULT_DATA_DIR
) -> StarCatalog:
    """Hipparcos stars brighter than ``max_mag``, with proper motion applied to ``epoch_year``.

    The parsed catalogue is cached as a small .npz next to the raw file.
    """
    cache = data_dir / "hip_parsed.npz"
    if not cache.exists():
        # Columns: HIP number, RA, Dec, V magnitude, proper motion in RA and Dec.
        rows: list[tuple[float, ...]] = []
        with open(ensure_hipparcos(data_dir), encoding="ascii", errors="replace") as fh:
            for line in fh:
                f = line.split("|")
                try:
                    rows.append(tuple(float(f[i]) for i in (1, 8, 9, 5, 12, 13)))
                except ValueError:  # a few entries lack astrometry or photometry
                    continue
        cols = np.array(rows).T
        np.savez(
            cache, hip=cols[0], ra=cols[1], dec=cols[2], mag=cols[3], pmra=cols[4], pmdec=cols[5]
        )
    d = np.load(cache)
    keep = d["mag"] <= max_mag
    dt = epoch_year - 1991.25  # Hipparcos catalogue epoch
    dec = d["dec"][keep]
    # Proper motions are in mas/yr; pmRA already includes the cos(dec) factor.
    ra = d["ra"][keep] + d["pmra"][keep] * dt / 3.6e6 / np.cos(np.radians(dec))
    dec = dec + d["pmdec"][keep] * dt / 3.6e6
    return StarCatalog(d["hip"][keep].astype(int), ra % 360.0, dec, d["mag"][keep])


def database_path(
    min_fov: float, max_fov: float, max_mag: float, data_dir: Path = DEFAULT_DATA_DIR
) -> Path:
    # No dots in the stem: tetra3 swaps everything after the last dot for ".npz".
    tag = f"fov{min_fov:g}-{max_fov:g}_mag{max_mag:g}".replace(".", "p")
    return data_dir / f"tetra3_{tag}.npz"


def build_database(
    min_fov: float = 20.0,
    max_fov: float = 70.0,
    max_mag: float = 6.5,
    data_dir: Path = DEFAULT_DATA_DIR,
    force: bool = False,
) -> Path:
    """Build (or reuse) a solver database covering horizontal FOVs in [min_fov, max_fov] deg.

    Wide-angle sky cameras see fewer, brighter stars per square degree, so a magnitude
    limit around 6-6.5 matches what they actually record and keeps the database small.
    """
    out = database_path(min_fov, max_fov, max_mag, data_dir)
    if out.exists() and not force:
        return out
    catalog = ensure_hipparcos(data_dir)
    log.warning(
        "Building plate-solver database for FOV %g-%g deg (a few minutes)...", min_fov, max_fov
    )
    solver = tetra3.Tetra3(load_database=None)
    solver.generate_database(
        max_fov=max_fov,
        min_fov=min_fov,
        star_max_magnitude=max_mag,
        star_catalog="hip_main",
        catalog_path=catalog,
        save_as=out.with_suffix(""),
    )
    return out
