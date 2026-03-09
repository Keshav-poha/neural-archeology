"""
decode_and_save.py
==================
Decode RAW sensor data from .npz archives and save grayscale visualization PNGs.

Each .npz file contains pre-ISP RAW frames for one sensor in one scene.
The output PNGs are for visual inspection only — pixel values are linearly
scaled from the native sensor bit-depth range to 8-bit grayscale.  The
original RAW values stored in the .npz files are never modified.

Usage
-----
    python src/decode_and_save.py [options]

Examples
--------
    # Process all scenes and sensors (default)
    python src/decode_and_save.py

    # Single scene
    python src/decode_and_save.py --scenes color_1lx_120

    # Specific sensors across two scenes
    python src/decode_and_save.py --scenes color_1lx_120 spring_1lx_120 --sensors 1 3 5

    # Override data / output directories
    python src/decode_and_save.py --data-dir /path/to/data --output-dir /path/to/out
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

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

# Default paths (relative to the repo root, i.e. the working directory when
# the script is invoked as  `python src/decode_and_save.py`).
DATA_DIR: Path = Path("data")
OUTPUT_DIR: Path = Path("results") / "processed_images"

# Mapping: output filename stem  →  key inside the .npz archive
FRAME_KEYS: dict[str, str] = {
    "gt": "gt",
    "normal": "frame_00",
    "extra_noisy": "extra_noisy_09_00",
}

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


def to_uint8_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Linearly scale *img* into [0, 255] and return as ``uint8``.

    This is a lossy, display-only transformation.  It is applied exclusively
    to produce human-readable PNGs and must never be used on data intended
    for statistical analysis.

    Parameters
    ----------
    img:
        Array of any numeric dtype (2-D or 3-D; only the first channel is
        used if 3-D).

    Returns
    -------
    np.ndarray
        2-D ``uint8`` array suitable for grayscale PNG export.
    """
    arr = img.astype(np.float64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)


def save_png(img: np.ndarray, path: Path) -> None:
    """Write *img* as a grayscale PNG, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8_grayscale(img), mode="L").save(path)


def process_sensor(
    scene: str,
    sensor_id: int,
    data_dir: Path,
    output_dir: Path,
) -> bool:
    """
    Load one sensor .npz and write the configured visualization PNGs.

    Parameters
    ----------
    scene:
        Scene folder name, e.g. ``"color_1lx_120"``.
    sensor_id:
        Integer sensor index (1–14).
    data_dir:
        Repo-relative root that contains the scene sub-folders.
    output_dir:
        Root directory for PNG output.

    Returns
    -------
    bool
        ``True`` on success, ``False`` if the source file is missing.
    """
    npz_path = data_dir / scene / f"sensor_{sensor_id}.npz"
    if not npz_path.exists():
        log.warning("Missing: %s", npz_path)
        return False

    data = np.load(npz_path)
    sensor_out = output_dir / scene / f"sensor_{sensor_id}"

    for out_stem, npz_key in FRAME_KEYS.items():
        if npz_key not in data:
            log.warning("Key '%s' not found in %s — skipping.", npz_key, npz_path)
            continue
        save_png(data[npz_key], sensor_out / f"{out_stem}.png")

    log.info("Saved  %s / sensor_%d", scene, sensor_id)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode RAW sensor .npz archives into grayscale visualization PNGs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=ALL_SCENES,
        choices=ALL_SCENES,
        metavar="SCENE",
        help=(
            "Scene(s) to process. "
            f"Choices: {ALL_SCENES}. "
            "Defaults to all 5 scenes."
        ),
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
        help=f"Root output directory for PNGs. Default: {OUTPUT_DIR}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    log.info(
        "Processing %d scene(s) × %d sensor(s).",
        len(args.scenes),
        len(args.sensors),
    )
    log.info("Data dir   : %s", args.data_dir.resolve())
    log.info("Output dir : %s", args.output_dir.resolve())

    total = success = 0
    for scene in args.scenes:
        for sensor_id in args.sensors:
            total += 1
            if process_sensor(scene, sensor_id, args.data_dir, args.output_dir):
                success += 1

    log.info(
        "Done. %d / %d sensor–scene pairs processed successfully.",
        success,
        total,
    )


if __name__ == "__main__":
    main()