"""Real-scale single global terrain for the Hill 395 battlefield.

Built over ONE continuous ~6 km (E-W) x ~8 km (N-S) area (config), NOT a tiled/
periodic field. Features are a procedural approximation of the qualitative
source description (exact dimensions are not in the sources):
  - central peak ~395 m (Hill 395), south-central
  - a ridge extending NORTH from the peak (the "camel-back" ridge)
  - Yeokgok stream: a linear low ~5 km north of the hill (E-W depression)
  - eastern lowland plain (Cheorwon plain, tank country) — kept low/flat
  - the MLR defensive band to the south
Concealment layers: canopy (forested mid-slopes), trench (MLR band),
reverse_slope (north-facing ridge slopes). Documented as approximation in meta.
"""
from __future__ import annotations

import math

import numpy as np

from .common import BASE_E, BASE_N
from .terrain import Terrain, CONCEAL_LAYERS


def build_terrain_real(cfg: dict) -> tuple[Terrain, dict]:
    tcfg = cfg.get("terrain", {})
    res = float(tcfg.get("grid_resolution_m", 50.0))
    base = float(tcfg.get("base_elevation_m", 50.0))
    width_e = float(tcfg.get("area_width_m", 6000.0))       # E-W extent
    width_n = float(tcfg.get("area_height_m", 8000.0))      # N-S extent
    peak_h = float(tcfg.get("peak_elevation_m", 395.0))     # summit AMSL
    peak_sig = float(tcfg.get("peak_sigma_m", 900.0))
    ridge_h = float(tcfg.get("ridge_amplitude_m", 220.0))
    ridge_sig_e = float(tcfg.get("ridge_sigma_e_m", 350.0))
    stream_depth = float(tcfg.get("stream_depth_m", 25.0))
    stream_north_m = float(tcfg.get("stream_north_offset_m", 5000.0))
    canopy_frac = float(tcfg.get("canopy_frac", 0.14))
    noise_amp = float(tcfg.get("noise_amplitude_m", 10.0))
    trench_radius = float(tcfg.get("trench_radius_m", 400.0))

    e_lo, n_lo = BASE_E, BASE_N
    e_hi, n_hi = BASE_E + width_e, BASE_N + width_n
    nx = int(math.ceil(width_e / res)) + 1
    ny = int(math.ceil(width_n / res)) + 1
    ee = e_lo + res * np.arange(nx)
    nn = n_lo + res * np.arange(ny)
    E, N = np.meshgrid(ee, nn)                              # (ny, nx)

    rng = np.random.default_rng(int(cfg["global_seed"]) + 777)

    # peak: south-central. ridge runs north from it.
    peak_e = e_lo + 0.55 * width_e
    peak_n = n_lo + 0.35 * width_n
    ridge_top_n = peak_n + 0.45 * width_n                   # ridge extends north

    # central peak (2-D gaussian to ~peak_h) and the north ridge as SEPARATE
    # bumps combined with max() so the ridge (lower) never stacks onto the summit.
    peak_bump = (peak_h - base) * np.exp(-(((E - peak_e) ** 2) + ((N - peak_n) ** 2)) / (2.0 * peak_sig ** 2))
    north_mask = np.clip((ridge_top_n - N) / (ridge_top_n - peak_n), 0.0, 1.0) * (N >= peak_n)
    ridge_bump = ridge_h * np.exp(-((E - peak_e) ** 2) / (2.0 * ridge_sig_e ** 2)) * north_mask
    elev = base + np.maximum(peak_bump, ridge_bump)
    # eastern plain: gently pull the far-east lowland down toward base
    east_frac = np.clip((E - (e_lo + 0.7 * width_e)) / (0.3 * width_e), 0.0, 1.0)
    elev = elev - east_frac * (elev - base) * 0.6
    # Yeokgok stream: E-W linear depression ~stream_north_m north of the peak
    stream_n = peak_n + stream_north_m
    if stream_n <= n_hi:
        elev -= stream_depth * np.exp(-((N - stream_n) ** 2) / (2.0 * (150.0) ** 2))
    # low-frequency noise
    for _ in range(6):
        kx = rng.uniform(0.3, 1.5) / peak_sig; ky = rng.uniform(0.3, 1.5) / peak_sig
        ph = rng.uniform(0, 2 * math.pi)
        elev += (noise_amp / 6.0) * np.sin(kx * (E - e_lo) + ky * (N - n_lo) + ph)
    elev = np.maximum(elev, base - stream_depth)            # floor

    # --- concealment ---
    conceal = np.zeros((3, ny, nx), dtype=np.uint8)
    # canopy: forested mid-slopes (moderate elevation), random patches
    mid = (elev > base + 40) & (elev < base + 220)
    conceal[0] = (mid & (rng.random((ny, nx)) < canopy_frac)).astype(np.uint8)
    # trench: MLR band just SOUTH of the peak (defender line)
    mlr_n = peak_n - 1000.0
    trench = (np.abs(N - mlr_n) <= trench_radius) & (np.abs(E - peak_e) <= 0.35 * width_e)
    conceal[1] = trench.astype(np.uint8)
    # reverse slope: north-facing (elevation decreasing as northing increases) & raised
    dz_dn = np.gradient(elev, res, axis=0)
    conceal[2] = ((dz_dn < -0.02) & (elev > base + 60) & (N > peak_n)).astype(np.uint8)

    terrain = Terrain(e_lo, n_lo, res, elev.astype(np.float32), conceal, base,
                      tiled=False)
    meta = {"origin_e": e_lo, "origin_n": n_lo, "res_m": res, "nx": nx, "ny": ny,
            "mode": "global_real", "tiled": False,
            "area_width_m": width_e, "area_height_m": width_n,
            "base_elevation_m": base, "peak_elevation_m": float(elev.max()),
            "peak_e": peak_e, "peak_n": peak_n, "ridge_top_n": ridge_top_n,
            "stream_north_m": stream_north_m,
            "concealment_layers": CONCEAL_LAYERS,
            "elevation_min": float(elev.min()), "elevation_max": float(elev.max()),
            "features": ["central_peak_~395m", "north_ridge", "yeokgok_stream_linear_low",
                         "eastern_plain_lowland", "mlr_trench_band"],
            "note": "procedural approximation of qualitative source description; "
                    "not a reproduction of real terrain/dimensions."}
    return terrain, meta
