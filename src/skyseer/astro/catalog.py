"""Star catalogue download and plate-solver database generation.

The solver needs a pattern database built for the camera's field of view. Databases are
built once from the Hipparcos catalogue (ESA, 118,218 stars) and cached on disk.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._vendor import tetra3
from .frames import radec_to_vec

log = logging.getLogger(__name__)

# The CDS archive in Strasbourg, then its mirror at Harvard (CfA): the same file, byte for
# byte. Any copy is checked against this SHA-256 before it is used.
HIPPARCOS_URLS = (
    "https://cdsarc.cds.unistra.fr/ftp/cats/I/239/hip_main.dat",
    "https://vizier.cfa.harvard.edu/ftp/cats/I/239/hip_main.dat",
)
HIPPARCOS_SHA256 = "58ceabb104d647160d9437ce6e513a02a036bb4ad9f8879a5a22fd52943616e0"
DEFAULT_DATA_DIR = Path("data/astro")


def hipparcos_path(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    return data_dir / "hip_main.dat"


def _download(url: str, out: Path) -> str:
    """Save ``url`` to ``out`` and return the file's SHA-256."""
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as resp, open(out, "wb") as fh:
        while chunk := resp.read(1 << 20):
            digest.update(chunk)
            fh.write(chunk)
    return digest.hexdigest()


def ensure_hipparcos(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download the Hipparcos main catalogue (~51 MB) if it isn't cached yet.

    Tries each server in turn, so one archive being down doesn't stop a first run.
    """
    path = hipparcos_path(data_dir)
    if path.exists() and path.stat().st_size > 50_000_000:
        return path
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    problems = []
    for url in HIPPARCOS_URLS:
        log.warning("Downloading Hipparcos catalogue from %s (~51 MB)...", url)
        try:
            sha = _download(url, tmp)
        except OSError as err:  # URLError, timeouts and refused connections are OSErrors
            problems.append(f"{url}: {err}")
            continue
        if sha == HIPPARCOS_SHA256:
            tmp.replace(path)
            return path
        problems.append(f"{url}: the file differs from the published catalogue")
    tmp.unlink(missing_ok=True)
    raise OSError("Could not download the Hipparcos catalogue:\n  " + "\n  ".join(problems))


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
