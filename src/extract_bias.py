"""
extract_bias.py
===============
Phase 2: Extract Fixed-Pattern Noise (FPN) bias maps and test scene independence.

Requires ``results/statistics/`` to be populated by ``compute_statistics.py``.

Bias definition
---------------
The **bias map** for a sensor/scene pair is the temporal mean of the 10 noisy
frames (``pixel_mean`` from the statistics step).  This captures the persistent,
signal-independent fixed-pattern component.

The **FPN residual** is defined as::

    fpn = pixel_mean − gt

where ``gt`` is the ground-truth frame.  A non-zero FPN residual indicates
systematic offset beyond the true scene signal.

Row/column FPN (structural banding)::

    row_fpn[r] = mean over columns of bias_map[r, :]
    col_fpn[c] = mean over rows   of bias_map[:, c]

Scene-independence test
-----------------------
For each sensor, one of two signals is pairwise correlated across all 5 scenes
(Pearson r on the flattened spatial vectors):

  Default (``--use-fpn-residual`` not set):
      Uses **bias_map** (temporal mean of 10 noisy frames).  Scene content
      is still present, so r values are diluted by between-scene variation
      in scene brightness.

  With ``--use-fpn-residual``:
      Uses **fpn** = bias_map − gt.  The ground-truth frame is subtracted,
      removing the true scene signal and leaving only the sensor offset.
      This gives a much cleaner read on whether the fixed pattern is a
      stable sensor property.

A high cross-scene correlation indicates that the pattern is an intrinsic
sensor property rather than a scene-specific artefact.

Output files
------------
results/bias_maps/{scene}/sensor_{n}.npz
    bias_map, fpn, row_fpn, col_fpn

results/bias_maps/scene_independence.npz
    correlations   : (14, 5, 5) float32  — per-sensor Pearson r matrix
    mean_corr      : (14,)      float32  — mean off-diagonal correlation per sensor
    scenes         : list[str]           — ordered scene names
    sensor_ids     : list[int]           — sensor indices

Usage
-----
    python src/extract_bias.py [options]
"""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

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

STATS_DIR: Path = Path("results") / "statistics"
OUTPUT_DIR: Path = Path("results") / "bias_maps"

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


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson r between two flattened arrays."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def extract_and_save_bias(
    scene: str,
    sensor_id: int,
    stats_dir: Path,
    output_dir: Path,
) -> np.ndarray | None:
    """
    Load statistics for one sensor/scene, compute bias maps, and save.

    Returns the bias_map array (for later cross-scene comparison), or None
    if the statistics file is missing.
    """
    stats_path = stats_dir / scene / f"sensor_{sensor_id}.npz"
    if not stats_path.exists():
        log.warning("Statistics missing: %s", stats_path)
        return None

    stats = np.load(stats_path)
    bias_map: np.ndarray = stats["pixel_mean"]           # (H, W)
    gt: np.ndarray = stats["gt"]                          # (H, W)

    fpn = bias_map - gt                                   # (H, W)
    row_fpn = bias_map.mean(axis=1).astype(np.float32)   # (H,)
    col_fpn = bias_map.mean(axis=0).astype(np.float32)   # (W,)

    out_path = output_dir / scene / f"sensor_{sensor_id}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        bias_map=bias_map.astype(np.float32),
        fpn=fpn.astype(np.float32),
        row_fpn=row_fpn,
        col_fpn=col_fpn,
    )
    log.info("Saved bias maps → %s", out_path)
    return bias_map.astype(np.float32)


def compute_scene_independence(
    sensor_id: int,
    scenes: list[str],
    output_dir: Path,
    key: str = "bias_map",
) -> tuple[np.ndarray, float] | None:
    """
    Load *key* arrays for *sensor_id* across all *scenes* and compute the
    pairwise Pearson correlation matrix.

    Parameters
    ----------
    key:
        Which array to correlate — ``"bias_map"`` (default, includes scene
        content) or ``"fpn"`` (gt-subtracted residual, cleaner fingerprint).

    Returns (corr_matrix, mean_off_diagonal_r) or None if fewer than 2
    scenes have data for this sensor.
    """
    bias_maps: dict[str, np.ndarray] = {}
    for scene in scenes:
        bp = output_dir / scene / f"sensor_{sensor_id}.npz"
        if bp.exists():
            bias_maps[scene] = np.load(bp)[key]

    available = list(bias_maps.keys())
    n = len(scenes)
    if len(available) < 2:
        log.warning(
            "Sensor %d: only %d scene(s) available — cannot compute cross-scene correlation.",
            sensor_id,
            len(available),
        )
        return None

    # Build full n×n matrix (NaN for missing pairs)
    corr = np.full((n, n), np.nan, dtype=np.float32)
    for i in range(n):
        corr[i, i] = 1.0
    for i, j in combinations(range(n), 2):
        si, sj = scenes[i], scenes[j]
        if si in bias_maps and sj in bias_maps:
            r = pearson_r(bias_maps[si], bias_maps[sj])
            corr[i, j] = corr[j, i] = r

    # Mean of off-diagonal elements (ignore NaN)
    mask = ~np.eye(n, dtype=bool)
    off_diag = corr[mask]
    mean_r = float(np.nanmean(off_diag))

    log.info(
        "Sensor %2d — mean cross-scene Pearson r = %.4f  (%d / %d scenes)",
        sensor_id,
        mean_r,
        len(available),
        n,
    )
    return corr, mean_r


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract FPN bias maps and compute cross-scene independence.",
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
        "--stats-dir",
        type=Path,
        default=STATS_DIR,
        metavar="DIR",
        help=f"Statistics input directory. Default: {STATS_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        metavar="DIR",
        help=f"Bias maps output directory. Default: {OUTPUT_DIR}",
    )
    parser.add_argument(
        "--use-fpn-residual",
        action="store_true",
        default=False,
        help=(
            "Correlate FPN residuals (bias_map − gt) across scenes instead of raw "
            "bias maps. Removes true scene signal before testing independence, "
            "giving higher and more meaningful cross-scene r values."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # --- Step 1: extract bias maps per sensor/scene -------------------------
    log.info(
        "Extracting bias maps for %d scene(s) × %d sensor(s).",
        len(args.scenes),
        len(args.sensors),
    )
    for scene in args.scenes:
        for sensor_id in args.sensors:
            extract_and_save_bias(scene, sensor_id, args.stats_dir, args.output_dir)

    # --- Step 2: cross-scene independence test ------------------------------
    corr_key = "fpn" if args.use_fpn_residual else "bias_map"
    log.info(
        "Computing cross-scene independence for %d sensor(s) using key='%s'.",
        len(args.sensors),
        corr_key,
    )

    all_corr: list[np.ndarray] = []
    mean_corrs: list[float] = []
    valid_sensor_ids: list[int] = []

    for sensor_id in args.sensors:
        result = compute_scene_independence(
            sensor_id, args.scenes, args.output_dir, key=corr_key
        )
        if result is not None:
            corr, mean_r = result
            all_corr.append(corr)
            mean_corrs.append(mean_r)
            valid_sensor_ids.append(sensor_id)

    if all_corr:
        out_path = args.output_dir / "scene_independence.npz"
        np.savez_compressed(
            out_path,
            correlations=np.stack(all_corr, axis=0),      # (S, 5, 5)
            mean_corr=np.array(mean_corrs, dtype=np.float32),
            sensor_ids=np.array(valid_sensor_ids, dtype=np.int32),
        )
        log.info("Scene independence results → %s", out_path)

        # Summary table
        log.info("")
        log.info("  %-10s  %s", "Sensor", "Mean cross-scene Pearson r")
        log.info("  %-10s  %s", "-" * 9, "-" * 26)
        for sid, r in zip(valid_sensor_ids, mean_corrs):
            log.info("  sensor_%-4d  %.4f", sid, r)

    log.info("Done.")


if __name__ == "__main__":
    main()
