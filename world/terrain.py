"""P2 — sensor-neutral terrain: elevation field + concealment layers + landmarks.

Two physical layers so ANY later sensor combination can be projected:
  (a) elevation field  -> ground line-of-sight (viewshed) for ground sensors
  (b) concealment map  -> canopy / trench / reverse-slope for overhead occlusion
The GT never assumes a sensor; it only records these physical facts.

Procedural model (documented approximation, NOT real Hill-395 terrain): each
sector carries a 2-D Gaussian hill centred on its crest landmark (crest high,
sloping down to the valley/north tip), plus low-frequency seeded noise. The
crest sits north of the MLR, so the far (northern) face is the reverse slope
relative to a defender/ground sensor to the south.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .common import BASE_E, BASE_N, GRID_COLS, LANDMARK_OFFSETS

CONCEAL_LAYERS = ["canopy", "trench", "reverse_slope"]


@dataclass
class Terrain:
    origin_e: float
    origin_n: float
    res: float
    elevation: np.ndarray          # (ny, nx) float32, metres
    concealment: np.ndarray        # (3, ny, nx) uint8  (canopy, trench, reverse_slope)
    base_elev: float
    # tiled mode: one representative sector tile reused for every sector (keeps
    # the array small/fast at 1000s of sectors). Global coords are folded onto
    # the tile via the sector pitch before lookup.
    tiled: bool = False
    pitch: float = 0.0
    tile_base_e: float = 0.0       # sector-0 origin the tile is anchored on
    tile_base_n: float = 0.0

    @property
    def shape(self):
        return self.elevation.shape

    def _fold(self, e: float, n: float) -> tuple[float, float]:
        if not self.tiled:
            return e, n
        col = round((e - self.tile_base_e) / self.pitch)
        row = round((n - self.tile_base_n) / self.pitch)
        return e - col * self.pitch, n - row * self.pitch

    def _ij(self, e: float, n: float) -> tuple[int, int]:
        e, n = self._fold(e, n)
        j = int(round((e - self.origin_e) / self.res))
        i = int(round((n - self.origin_n) / self.res))
        ny, nx = self.elevation.shape
        return min(max(i, 0), ny - 1), min(max(j, 0), nx - 1)

    def elev_at(self, e: float, n: float) -> float:
        i, j = self._ij(e, n)
        return float(self.elevation[i, j])


def build_terrain(cfg: dict) -> tuple[Terrain, dict]:
    """Return (Terrain, meta). meta carries grid geometry for serialisation."""
    tcfg = cfg.get("terrain", {})
    res = float(tcfg.get("grid_resolution_m", 50.0))
    amp = float(tcfg.get("ridge_amplitude_m", 120.0))
    sig = float(tcfg.get("ridge_sigma_m", 700.0))
    base = float(tcfg.get("base_elevation_m", 50.0))
    canopy_frac = float(tcfg.get("canopy_frac", 0.12))
    noise_amp = float(tcfg.get("noise_amplitude_m", 12.0))
    trench_radius = float(tcfg.get("trench_radius_m", 250.0))

    scope = cfg["world_scope"]
    n_sectors = int(scope.get("n_sectors", 1)) if scope.get("mode", "tiled") == "tiled" else 1
    pitch = float(scope.get("sector_pitch_m", 6000.0))
    # tile mode: build ONE representative sector tile (periodic terrain) so the
    # array stays small/fast for thousands of sectors. Global lookups fold onto
    # the tile via the pitch. "global" builds one array spanning the whole grid.
    mode = tcfg.get("mode", "global")
    n_terrain = 1 if mode == "tile" else n_sectors
    cols = min(GRID_COLS, n_terrain)
    rows = math.ceil(n_terrain / GRID_COLS)

    # bounding box covers every (terrain) sector footprint plus a margin.
    e_lo, e_hi = BASE_E - 1000.0, BASE_E + (cols - 1) * pitch + 3000.0
    n_lo, n_hi = BASE_N - 2000.0, BASE_N + (rows - 1) * pitch + 2500.0
    nx = int(math.ceil((e_hi - e_lo) / res)) + 1
    ny = int(math.ceil((n_hi - n_lo) / res)) + 1

    ee = e_lo + res * np.arange(nx)
    nn = n_lo + res * np.arange(ny)
    E, N = np.meshgrid(ee, nn)                     # (ny, nx)

    rng = np.random.default_rng(int(cfg["global_seed"]) + 777)   # terrain sub-stream

    elev = np.full((ny, nx), base, dtype=np.float64)
    crest_centres = []
    for s in range(n_terrain):
        ox = BASE_E + (s % GRID_COLS) * pitch
        oy = BASE_N + (s // GRID_COLS) * pitch
        cx = ox + LANDMARK_OFFSETS["crest"][0]
        cy = oy + LANDMARK_OFFSETS["crest"][1]
        crest_centres.append((cx, cy))
        elev += amp * np.exp(-(((E - cx) ** 2) + ((N - cy) ** 2)) / (2.0 * sig ** 2))

    # low-frequency seeded noise (smooth): sum of a few random sinusoids.
    for _ in range(6):
        kx = rng.uniform(0.3, 1.5) / sig
        ky = rng.uniform(0.3, 1.5) / sig
        ph = rng.uniform(0, 2 * math.pi)
        elev += (noise_amp / 6.0) * np.sin(kx * (E - e_lo) + ky * (N - n_lo) + ph)

    # --- concealment layers ---
    conceal = np.zeros((3, ny, nx), dtype=np.uint8)
    # canopy: low ground (below base + small rise), random patches
    low = elev < (base + 0.15 * amp)
    canopy_noise = rng.random((ny, nx)) < canopy_frac
    conceal[0] = (low & canopy_noise).astype(np.uint8)
    # trench: within trench_radius of any MLR landmark (defender dug-in band)
    trench = np.zeros((ny, nx), dtype=bool)
    for s in range(n_terrain):
        ox = BASE_E + (s % GRID_COLS) * pitch
        oy = BASE_N + (s // GRID_COLS) * pitch
        mx = ox + LANDMARK_OFFSETS["mlr"][0]; my = oy + LANDMARK_OFFSETS["mlr"][1]
        trench |= (((E - mx) ** 2 + (N - my) ** 2) <= trench_radius ** 2)
    conceal[1] = trench.astype(np.uint8)
    # reverse slope: raised ground on the far (north) side of the nearest crest
    rev = np.zeros((ny, nx), dtype=bool)
    raised = elev > (base + 0.20 * amp)
    for (cx, cy) in crest_centres:
        near = ((E - cx) ** 2 + (N - cy) ** 2) <= (2.0 * sig) ** 2
        rev |= (near & (N > cy) & raised)
    conceal[2] = rev.astype(np.uint8)

    terrain = Terrain(e_lo, n_lo, res, elev.astype(np.float32), conceal, base,
                      tiled=(mode == "tile"), pitch=pitch,
                      tile_base_e=BASE_E, tile_base_n=BASE_N)
    full_cols = min(GRID_COLS, n_sectors); full_rows = math.ceil(n_sectors / GRID_COLS)
    meta = {"origin_e": e_lo, "origin_n": n_lo, "res_m": res, "nx": nx, "ny": ny,
            "base_elevation_m": base, "ridge_amplitude_m": amp, "ridge_sigma_m": sig,
            "concealment_layers": CONCEAL_LAYERS,
            "elevation_min": float(elev.min()), "elevation_max": float(elev.max()),
            "mode": mode, "tiled": mode == "tile", "sector_pitch_m": pitch,
            "tile_base_e": BASE_E, "tile_base_n": BASE_N,
            "full_grid_cols": full_cols, "full_grid_rows": full_rows, "n_sectors": n_sectors,
            "note": ("terrain is a single representative sector tile reused for every sector "
                     "(periodic); fold global coords by pitch. Procedural approximation."
                     if mode == "tile" else "single array spanning the whole sector grid")}
    return terrain, meta
