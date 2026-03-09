"""
compute_statistics.py
=====================
Phase 1: Compute per-pixel temporal statistics for each sensor/scene pair.

For each sensor .npz the following statistics are derived from the 10 noisy
frames (frame_00 … frame_09) and saved to a companion .npz under
``results/statistics/{scene}/sensor_{n}.npz``.

Output arrays
-------------
pixel_mean     : (H, W) float32  — temporal mean across noisy frames
pixel_var      : (H, W) float32  — temporal variance (Bessel-corrected)
pixel_std      : (H, W) float32  — temporal standard deviation
gt             : (H, W) float32  — ground-truth frame (unmodified copy)
frame_means    : (10,)  float32  — per-frame spatial mean
frame_stds     : (10,)  float32  — per-frame spatial standard deviation
extra_mean     : (H, W) float32  — temporal mean across extra-noisy frames
extra_var      : (H, W) float32  — temporal variance across extra-noisy frames
n_noisy_frames : scalar int      — number of noisy frames accumulated
n_extra_frames : scalar int      — number of extra-noisy frames accumulated

Usage
-----
    python src/compute_statistics.py [options]

Examples
--------
    python src/compute_statistics.py
    python src/compute_statistics.py --scenes color_1lx_120 --sensors 1 2 3
    python src/compute_statistics.py --data-dir data --output-dir results/statistics
"""

from __future__ import annotations

import argparse
import logging
import sys
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
NOISY_KEYS: list[str] = [f"frame_{i:02d}" for i in range(10)]
EXTRA_NOISY_KEYS: list[str] = [f"extra_noisy_09_{i:02d}" for i in range(20)]

DATA_DIR: Path = Path("data")
OUTPUT_DIR: Path = Path("results") / "statistics"

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
# Core helpers
# ---------------------------------------------------------------------------


def online_mean_var(
    arrays: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pixel-wise mean and Bessel-corrected variance using Welford's
    online algorithm; avoids stacking all frames into memory at once.

    Parameters
    ----------
    arrays:
        Sequence of 2-D float arrays of identical shape.

    Returns
    -------
    mean, var : pair of (H, W) float64 arrays
    """
    n = 0
    mean = np.zeros(arrays[0].shape, dtype=np.float64)
    M2 = np.zeros(arrays[0].shape, dtype=np.float64)

    for arr in arrays:
        n += 1
        delta = arr.astype(np.float64) - mean
        mean += delta / n
        delta2 = arr.astype(np.float64) - mean
        M2 += delta * delta2

    var = M2 / (n - 1) if n > 1 else np.zeros_like(M2)
    return mean, var


def process_sensor(
    scene: str,
    sensor_id: int,
    data_dir: Path,
    output_dir: Path,
) -> bool:
    """
    Compute and save statistics for one sensor/scene pair.

    Returns True on success, False if the source file is missing.
    """
    npz_path = data_dir / scene / f"sensor_{sensor_id}.npz"
    if not npz_path.exists():
        log.warning("Missing: %s", npz_path)
        return False

    data = np.load(npz_path)
    available = set(data.files)

    # --- noisy frames -------------------------------------------------------
    noisy_keys = [k for k in NOISY_KEYS if k in available]
    if not noisy_keys:
        log.warning("No noisy frames found in %s", npz_path)
        return False

    noisy_frames = [data[k] for k in noisy_keys]
    pixel_mean, pixel_var = online_mean_var(noisy_frames)
    pixel_std = np.sqrt(pixel_var)

    frame_means = np.array([f.mean() for f in noisy_frames], dtype=np.float32)
    frame_stds = np.array([f.std() for f in noisy_frames], dtype=np.float32)

    # --- extra-noisy frames -------------------------------------------------
    extra_keys = [k for k in EXTRA_NOISY_KEYS if k in available]
    if extra_keys:
        extra_frames = [data[k] for k in extra_keys]
        extra_mean, extra_var = online_mean_var(extra_frames)
    else:
        log.warning("No extra-noisy frames in %s — skipping extra stats.", npz_path)
        extra_mean = np.full_like(pixel_mean, np.nan)
        extra_var = np.full_like(pixel_var, np.nan)

    # --- ground truth -------------------------------------------------------
    gt = data["gt"] if "gt" in available else np.full_like(pixel_mean, np.nan)

    # --- save ---------------------------------------------------------------
    out_path = output_dir / scene / f"sensor_{sensor_id}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        pixel_mean=pixel_mean.astype(np.float32),
        pixel_var=pixel_var.astype(np.float32),
        pixel_std=pixel_std.astype(np.float32),
        gt=gt.astype(np.float32),
        frame_means=frame_means,
        frame_stds=frame_stds,
        extra_mean=extra_mean.astype(np.float32),
        extra_var=extra_var.astype(np.float32),
        n_noisy_frames=np.int32(len(noisy_keys)),
        n_extra_frames=np.int32(len(extra_keys)),
    )
    log.info("Saved statistics → %s", out_path)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-pixel temporal statistics from RAW noisy frames "
            "and save to results/statistics/."
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
        help="Scene(s) to process. Defaults to all 5.",
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        type=int,
        default=list(range(1, NUM_SENSORS + 1)),
        metavar="N",
        help="Sensor number(s) to process (1–14). Defaults to all 14.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        metavar="DIR",
        help=f"Root data directory. Default: {DATA_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        metavar="DIR",
        help=f"Root output directory. Default: {OUTPUT_DIR}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    log.info(
        "Computing statistics for %d scene(s) × %d sensor(s).",
        len(args.scenes),
        len(args.sensors),
    )

    total = success = 0
    for scene in args.scenes:
        for sensor_id in args.sensors:
            total += 1
            if process_sensor(scene, sensor_id, args.data_dir, args.output_dir):
                success += 1

    log.info("Done. %d / %d pairs processed successfully.", success, total)


if __name__ == "__main__":
    main()
