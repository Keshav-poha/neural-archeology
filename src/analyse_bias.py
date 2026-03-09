"""
analyse_bias.py
===============
Phase 3: Cross-sensor comparison, hierarchical clustering, and spatial
frequency analysis of bias maps.

Requires ``results/bias_maps/`` to be populated by ``extract_bias.py``.

Analyses performed
------------------
1. **Pairwise sensor similarity** — Pearson r between every pair of sensor
   bias maps, separately for each scene and averaged across scenes.

2. **Hierarchical clustering** — sensors grouped by their mean bias
   fingerprint using Ward linkage on spatially downsampled bias maps.

3. **Power Spectral Density (PSD)** — 1-D PSD along rows and columns of
   each sensor's mean bias map, revealing periodic banding or structured
   noise patterns.

4. **FPN magnitude summary** — per-sensor RMS FPN across scenes and the
   cross-scene variance of the FPN (a measure of scene-dependence).

Output files
------------
results/analysis/sensor_similarity.npz
    similarity   : (14, 14, 5) float32  — Pearson r per sensor pair per scene
    mean_sim     : (14, 14)    float32  — scene-averaged similarity
    sensor_ids   : (14,)       int32

results/analysis/clustering.npz
    linkage_matrix : scipy linkage matrix (14-1, 4) float64
    sensor_ids     : (14,) int32
    feature_matrix : (14, D) float32  — downsampled bias fingerprints used

results/analysis/psd.npz
    row_psd    : (14, W//2+1) float32  — mean row-direction PSD per sensor
    col_psd    : (14, H//2+1) float32  — mean col-direction PSD per sensor
    sensor_ids : (14,) int32
    row_freqs  : (W//2+1,) float32
    col_freqs  : (H//2+1,) float32

results/analysis/fpn_summary.npz
    rms_fpn          : (14, 5) float32  — per-sensor, per-scene RMS FPN
    mean_rms_fpn     : (14,)   float32  — averaged across scenes
    scene_var_fpn    : (14,)   float32  — variance of RMS FPN across scenes
    sensor_ids       : (14,)  int32

Usage
-----
    python src/analyse_bias.py [options]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.signal import periodogram

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_SCENES: list[str] = [
    "color_1lx_120",
    "hardware_1lx_120",
    "spring_1lx_120",
    "toys_1lx_120",
    "yarn_1lx_120",
]
NUM_SENSORS: int = 14

BIAS_DIR: Path = Path("results") / "bias_maps"
OUTPUT_DIR: Path = Path("results") / "analysis"

# Downsampling factor for clustering features (reduces 2736×3648 → ~137×182)
DOWNSAMPLE: int = 20

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_bias_map(
    scene: str,
    sensor_id: int,
    bias_dir: Path,
    key: str = "bias_map",
) -> np.ndarray | None:
    """Load a single array from a bias_maps .npz file, or return None."""
    path = bias_dir / scene / f"sensor_{sensor_id}.npz"
    if not path.exists():
        return None
    return np.load(path)[key].astype(np.float32)


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two flattened arrays."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-average downsample a 2-D array by *factor* in each dimension."""
    h, w = arr.shape
    h_trim = (h // factor) * factor
    w_trim = (w // factor) * factor
    return (
        arr[:h_trim, :w_trim]
        .reshape(h_trim // factor, factor, w_trim // factor, factor)
        .mean(axis=(1, 3))
    )


def row_psd(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean row-direction PSD: average the 1-D PSD of each row.

    Returns (frequencies, power) both as 1-D float32 arrays.
    """
    freqs, powers = None, []
    for row in arr:
        f, p = periodogram(row.astype(np.float64), window="hann")
        powers.append(p)
        if freqs is None:
            freqs = f
    return freqs.astype(np.float32), np.mean(powers, axis=0).astype(np.float32)


def col_psd(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean column-direction PSD: average the 1-D PSD of each column.
    """
    return row_psd(arr.T)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def compute_pairwise_similarity(
    sensors: list[int],
    scenes: list[str],
    bias_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (S, S, N_scenes) Pearson r tensor and scene-averaged (S, S) matrix.
    """
    n = len(sensors)
    ns = len(scenes)
    sim_3d = np.full((n, n, ns), np.nan, dtype=np.float32)

    for si, scene in enumerate(scenes):
        log.info("Pairwise similarity — scene: %s", scene)
        maps = {sid: load_bias_map(scene, sid, bias_dir) for sid in sensors}
        # Downsample every map consistently so cross-sensor shapes are comparable.
        ds_maps: dict[int, np.ndarray | None] = {
            sid: (downsample(m, DOWNSAMPLE) if m is not None else None)
            for sid, m in maps.items()
        }
        for i in range(n):
            sim_3d[i, i, si] = 1.0
            for j in range(i + 1, n):
                a = ds_maps[sensors[i]]
                b = ds_maps[sensors[j]]
                if a is not None and b is not None:
                    if a.shape != b.shape:
                        # Different sensor resolutions — crop to common overlap.
                        h = min(a.shape[0], b.shape[0])
                        w = min(a.shape[1], b.shape[1])
                        a = a[:h, :w]
                        b = b[:h, :w]
                    r = pearson_r(a, b)
                    sim_3d[i, j, si] = r
                    sim_3d[j, i, si] = r

    mean_sim = np.nanmean(sim_3d, axis=2)
    return sim_3d, mean_sim


def compute_clustering(
    sensors: list[int],
    scenes: list[str],
    bias_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a feature matrix (one row per sensor) from scene-averaged,
    spatially downsampled bias maps, then compute Ward linkage.

    Returns (linkage_matrix, feature_matrix).
    """
    # Collect per-sensor downsampled maps (lists of 2-D arrays per scene).
    ds_by_sensor: dict[int, list[np.ndarray]] = {}
    for sensor_id in sensors:
        maps_s = []
        for scene in scenes:
            m = load_bias_map(scene, sensor_id, bias_dir)
            if m is not None:
                maps_s.append(downsample(m, DOWNSAMPLE))
        if maps_s:
            ds_by_sensor[sensor_id] = maps_s

    if len(ds_by_sensor) < 2:
        log.warning("Not enough sensors with data for clustering.")
        return np.array([]), np.array([])

    # Find minimum common (h, w) in case sensors have different native resolutions.
    min_h = min(m.shape[0] for maps_s in ds_by_sensor.values() for m in maps_s)
    min_w = min(m.shape[1] for maps_s in ds_by_sensor.values() for m in maps_s)

    features: list[np.ndarray] = []
    valid_sensors: list[int] = []
    for sensor_id, maps_s in ds_by_sensor.items():
        cropped = [m[:min_h, :min_w].ravel() for m in maps_s]
        features.append(np.mean(cropped, axis=0).astype(np.float32))
        valid_sensors.append(sensor_id)

    feature_matrix = np.stack(features, axis=0)   # (S, D)
    Z = linkage(feature_matrix, method="ward", metric="euclidean")
    log.info("Clustering: %d sensors, feature dim = %d", len(valid_sensors), feature_matrix.shape[1])
    return Z, feature_matrix


def compute_psd(
    sensors: list[int],
    scenes: list[str],
    bias_dir: Path,
) -> dict:
    """
    Compute mean row- and column-direction PSDs per sensor, averaged across
    available scenes.  Returns a dict ready for np.savez_compressed.
    """
    row_psds: list[np.ndarray] = []
    col_psds: list[np.ndarray] = []
    row_freqs_list: list[np.ndarray] = []
    col_freqs_list: list[np.ndarray] = []
    valid_sensors: list[int] = []

    for sensor_id in sensors:
        r_powers, c_powers = [], []
        rf_sensor = cf_sensor = None
        for scene in scenes:
            m = load_bias_map(scene, sensor_id, bias_dir)
            if m is None:
                continue
            rf, rp = row_psd(m)
            cf, cp = col_psd(m)
            r_powers.append(rp)
            c_powers.append(cp)
            if rf_sensor is None:
                rf_sensor = rf
                cf_sensor = cf

        if r_powers:
            row_psds.append(np.mean(r_powers, axis=0))
            col_psds.append(np.mean(c_powers, axis=0))
            row_freqs_list.append(rf_sensor)
            col_freqs_list.append(cf_sensor)
            valid_sensors.append(sensor_id)
            log.info("PSD computed for sensor %d (%d scene(s))", sensor_id, len(r_powers))

    if not row_psds:
        return {"row_psd": np.array([]), "col_psd": np.array([]),
                "row_freqs": np.array([]), "col_freqs": np.array([]),
                "sensor_ids": np.array([], dtype=np.int32)}

    # Truncate to minimum PSD length across sensors (sensors with different
    # resolutions produce PSDs of different lengths).
    min_row = min(p.shape[0] for p in row_psds)
    min_col = min(p.shape[0] for p in col_psds)

    return {
        "row_psd": np.stack([p[:min_row] for p in row_psds], axis=0).astype(np.float32),
        "col_psd": np.stack([p[:min_col] for p in col_psds], axis=0).astype(np.float32),
        "row_freqs": row_freqs_list[np.argmin([p.shape[0] for p in row_psds])][:min_row],
        "col_freqs": col_freqs_list[np.argmin([p.shape[0] for p in col_psds])][:min_col],
        "sensor_ids": np.array(valid_sensors, dtype=np.int32),
    }


def compute_fpn_summary(
    sensors: list[int],
    scenes: list[str],
    bias_dir: Path,
) -> dict:
    """
    Compute per-sensor, per-scene RMS FPN and summarise scene-dependence.
    """
    n, ns = len(sensors), len(scenes)
    rms_fpn = np.full((n, ns), np.nan, dtype=np.float32)

    for i, sensor_id in enumerate(sensors):
        for j, scene in enumerate(scenes):
            path = bias_dir / scene / f"sensor_{sensor_id}.npz"
            if path.exists():
                fpn = np.load(path)["fpn"]
                rms_fpn[i, j] = float(np.sqrt(np.mean(fpn ** 2)))

    mean_rms = np.nanmean(rms_fpn, axis=1)
    scene_var = np.nanvar(rms_fpn, axis=1)

    for i, sensor_id in enumerate(sensors):
        log.info(
            "FPN  sensor_%-2d  mean_rms=%.5f  scene_var=%.2e",
            sensor_id,
            mean_rms[i],
            scene_var[i],
        )

    return {
        "rms_fpn": rms_fpn,
        "mean_rms_fpn": mean_rms.astype(np.float32),
        "scene_var_fpn": scene_var.astype(np.float32),
        "sensor_ids": np.array(sensors, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-sensor similarity, hierarchical clustering, PSD, "
            "and FPN magnitude analysis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=ALL_SCENES,
        choices=ALL_SCENES,
        metavar="SCENE",
        help="Scenes to include. Defaults to all 5.",
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        type=int,
        default=list(range(1, NUM_SENSORS + 1)),
        metavar="N",
        help="Sensor numbers (1–14). Defaults to all 14.",
    )
    parser.add_argument(
        "--bias-dir",
        type=Path,
        default=BIAS_DIR,
        metavar="DIR",
        help=f"Bias maps input directory. Default: {BIAS_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        metavar="DIR",
        help=f"Analysis output directory. Default: {OUTPUT_DIR}",
    )
    parser.add_argument(
        "--skip-psd",
        action="store_true",
        help="Skip PSD computation (slow for large images).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pairwise similarity
    log.info("=== Pairwise sensor similarity ===")
    sim_3d, mean_sim = compute_pairwise_similarity(
        args.sensors, args.scenes, args.bias_dir
    )
    np.savez_compressed(
        args.output_dir / "sensor_similarity.npz",
        similarity=sim_3d,
        mean_sim=mean_sim.astype(np.float32),
        sensor_ids=np.array(args.sensors, dtype=np.int32),
    )
    log.info("Saved → %s", args.output_dir / "sensor_similarity.npz")

    # 2. Hierarchical clustering
    log.info("=== Hierarchical clustering ===")
    Z, feat = compute_clustering(args.sensors, args.scenes, args.bias_dir)
    if Z.size > 0:
        np.savez_compressed(
            args.output_dir / "clustering.npz",
            linkage_matrix=Z,
            feature_matrix=feat,
            sensor_ids=np.array(args.sensors, dtype=np.int32),
        )
        log.info("Saved → %s", args.output_dir / "clustering.npz")

    # 3. PSD
    if not args.skip_psd:
        log.info("=== Power spectral density ===")
        psd_data = compute_psd(args.sensors, args.scenes, args.bias_dir)
        np.savez_compressed(args.output_dir / "psd.npz", **psd_data)
        log.info("Saved → %s", args.output_dir / "psd.npz")
    else:
        log.info("PSD skipped (--skip-psd).")

    # 4. FPN summary
    log.info("=== FPN magnitude summary ===")
    fpn_data = compute_fpn_summary(args.sensors, args.scenes, args.bias_dir)
    np.savez_compressed(args.output_dir / "fpn_summary.npz", **fpn_data)
    log.info("Saved → %s", args.output_dir / "fpn_summary.npz")

    log.info("Analysis complete.")


if __name__ == "__main__":
    main()
