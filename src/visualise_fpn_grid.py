"""
visualise_fpn_grid.py
=====================
Generate 2-D grids and strips that expose scene-independent FPN structure.

For each sensor the FPN residual is:

    fpn = bias_map - gt

where ``bias_map`` is the 10-frame temporal mean and ``gt`` is the ground-truth
(200-frame average).  Random shot noise averages toward zero across scenes;
any persistent pattern is the true sensor FPN.

Outputs
-------
results/figures/fpn_grid/
    overview_grid.png
        14 rows (sensors) × 6 cols (5 scenes + cross-scene mean).
        Each panel is the FPN residual, colour-clipped to ±N×RMS of the
        cross-scene mean to keep a consistent scale per sensor.
        An optional display-only Gaussian blur suppresses pixel-level shot
        noise so structural patterns are easier to see.

    snr_grid.png
        14 rows × 2 cols (cross-scene mean  |  cross-scene std).
        Stable FPN shows as signal in the mean and silence in the std.

    sensor_{n:02d}_strip.png  (one per sensor)
        Single-sensor 1×6 strip at higher resolution (downsample ÷ 4).
        Useful for zooming in on individual sensors.

    sensor_{n:02d}_mean.png   (one per sensor)
        Cross-scene mean FPN at full resolution (downsample ÷ 2).

Usage
-----
    python src/visualise_fpn_grid.py [options]

Options
-------
    --sensors        Sensor numbers to process (default: all 14)
    --scenes         Scene names to include   (default: all 5)
    --bias-dir       Input directory           (default: results/bias_maps)
    --output-dir     Output directory          (default: results/figures/fpn_grid)
    --downsample     Display downsample factor (default: 16)
    --smooth-sigma   Gaussian blur sigma for DISPLAY only, 0 = off (default: 3)
    --clip-sigma     Colormap clip in ±N×RMS of per-sensor mean map (default: 3)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

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

SENSOR_NAMES: dict[int, str] = {
    1:  "Fold4-W",    2:  "Fold4-UW",    3:  "Fold4-T",
    4:  "P5a-W",      5:  "P5a-UW",      6:  "P5a-F",
    7:  "P7Pro-W",    8:  "P7Pro-UW",    9:  "P7Pro-T",    10: "P7Pro-F",
    11: "S20-W",      12: "S20-UW",
    13: "POCO-W",     14: "POCO-F",
}
SCENE_LABELS: dict[str, str] = {
    "color_1lx_120":    "color",
    "hardware_1lx_120": "hardware",
    "spring_1lx_120":   "spring",
    "toys_1lx_120":     "toys",
    "yarn_1lx_120":     "yarn",
}

BIAS_DIR:   Path = Path("results") / "bias_maps"
OUTPUT_DIR: Path = Path("results") / "figures" / "fpn_grid"

DPI: int = 150

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


def load_fpn(scene: str, sensor_id: int, bias_dir: Path) -> np.ndarray | None:
    """Load the raw FPN residual (bias_map - gt) for one sensor/scene."""
    path = bias_dir / scene / f"sensor_{sensor_id}.npz"
    if not path.exists():
        log.warning("Missing: %s", path)
        return None
    return np.load(path)["fpn"].astype(np.float32)


def block_downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-average 2-D array by *factor* in each spatial dimension."""
    h, w = arr.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    return (
        arr[:h2, :w2]
        .reshape(h2 // factor, factor, w2 // factor, factor)
        .mean(axis=(1, 3))
    )


def prepare(
    arr: np.ndarray,
    factor: int,
    sigma: float,
) -> np.ndarray:
    """
    Downsample then optionally Gaussian-blur an FPN panel for display.

    Blurring is applied AFTER downsampling (cheap) and is display-only.
    """
    ds = block_downsample(arr, factor)
    if sigma > 0:
        ds = gaussian_filter(ds, sigma=sigma)
    return ds


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


def render_panel(
    ax: plt.Axes,
    data: np.ndarray | None,
    vmin: float,
    vmax: float,
    title: str = "",
) -> None:
    """Render one FPN panel onto *ax*."""
    if data is None:
        ax.set_facecolor("0.15")
        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                transform=ax.transAxes, color="white", fontsize=13)
    else:
        ax.imshow(data, cmap="bwr", vmin=vmin, vmax=vmax,
                  interpolation="nearest", aspect="auto")
    if title:
        ax.set_title(title, fontsize=13, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Overview grid  (14 sensors × 6 cols: 5 scenes + mean)
# ---------------------------------------------------------------------------


def build_overview_grid(
    sensors: list[int],
    scenes: list[str],
    bias_dir: Path,
    output_dir: Path,
    downsample: int,
    smooth_sigma: float,
    clip_sigma: float,
) -> None:
    n_sensors = len(sensors)
    n_cols = len(scenes) + 1          # +1 for cross-scene mean
    col_labels = [SCENE_LABELS.get(s, s) for s in scenes] + ["MEAN"]

    fig, axes = plt.subplots(
        n_sensors, n_cols,
        figsize=(n_cols * 1.5, n_sensors * 1.2),
    )
    if n_sensors == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        "FPN residuals (bias_map − gt)  |  bwr: blue=negative offset, red=positive offset\n"
        "Colour scale = ±3×RMS of the per-sensor cross-scene mean",
        fontsize=14, y=1.01,
    )

    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].set_title(label, fontsize=14, pad=3,
                                   fontweight="bold" if label == "MEAN" else "normal")

    for row_idx, sensor_id in enumerate(sensors):
        sensor_label = SENSOR_NAMES.get(sensor_id, f"S{sensor_id:02d}")

        # Load all available scene panels for this sensor
        raw_panels: list[np.ndarray | None] = []
        for scene in scenes:
            fpn = load_fpn(scene, sensor_id, bias_dir)
            raw_panels.append(fpn)

        # Cross-scene mean (from available scenes only, at original resolution)
        available = [p for p in raw_panels if p is not None]
        if not available:
            mean_raw = None
        else:
            # Different resolutions: crop to smallest before averaging
            min_h = min(p.shape[0] for p in available)
            min_w = min(p.shape[1] for p in available)
            mean_raw = np.mean(
                [p[:min_h, :min_w] for p in available], axis=0
            ).astype(np.float32)

        # Colour scale: ±clip_sigma × RMS of the mean map
        if mean_raw is not None:
            rms = float(np.sqrt((mean_raw ** 2).mean()))
            vabs = max(clip_sigma * rms, 1e-9)
        else:
            vabs = 1e-4
        vmin, vmax = -vabs, vabs

        # Downsample + smooth each scene panel
        ds_panels = [
            (prepare(p, downsample, smooth_sigma) if p is not None else None)
            for p in raw_panels
        ]
        ds_mean = prepare(mean_raw, downsample, smooth_sigma) if mean_raw is not None else None

        # Render row label on the first column
        axes[row_idx, 0].set_ylabel(
            f"S{sensor_id:02d}\n{sensor_label}", fontsize=13, rotation=0,
            labelpad=36, va="center",
        )

        # Render scene columns
        for col_idx, panel in enumerate(ds_panels):
            render_panel(axes[row_idx, col_idx], panel, vmin, vmax)

        # Render mean column (last)
        ax_mean = axes[row_idx, n_cols - 1]
        render_panel(ax_mean, ds_mean, vmin, vmax)
        # Highlight mean column with a coloured border
        for spine in ax_mean.spines.values():
            spine.set_edgecolor("gold")
            spine.set_linewidth(1.2)

    plt.tight_layout(h_pad=0.3, w_pad=0.3)
    savefig(fig, output_dir / "overview_grid.png")


# ---------------------------------------------------------------------------
# SNR grid  (14 sensors × 2 cols: mean | std)
# ---------------------------------------------------------------------------


def build_snr_grid(
    sensors: list[int],
    scenes: list[str],
    bias_dir: Path,
    output_dir: Path,
    downsample: int,
    smooth_sigma: float,
) -> None:
    n_sensors = len(sensors)
    fig, axes = plt.subplots(
        n_sensors, 2, figsize=(4, n_sensors * 1.2),
    )
    if n_sensors == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Cross-scene Mean vs Std of FPN residual\n"
        "Left=Mean (persistent signal), Right=Std (variability/noise)\n"
        "Low std + non-zero mean = stable FPN fingerprint",
        fontsize=14, y=1.01,
    )
    axes[0, 0].set_title("Mean", fontsize=14, fontweight="bold")
    axes[0, 1].set_title("Std", fontsize=14)

    for row_idx, sensor_id in enumerate(sensors):
        available = []
        for scene in scenes:
            fpn = load_fpn(scene, sensor_id, bias_dir)
            if fpn is not None:
                available.append(fpn)

        if not available:
            for col in range(2):
                axes[row_idx, col].set_visible(False)
            continue

        min_h = min(p.shape[0] for p in available)
        min_w = min(p.shape[1] for p in available)
        stack = np.stack([p[:min_h, :min_w] for p in available], axis=0)  # (N,H,W)

        mean_map = stack.mean(axis=0)
        std_map  = stack.std(axis=0)

        ds_mean = prepare(mean_map, downsample, smooth_sigma)
        ds_std  = prepare(std_map,  downsample, 0)           # no blur on std

        axes[row_idx, 0].set_ylabel(
            f"S{sensor_id:02d}\n{SENSOR_NAMES.get(sensor_id, '')}", fontsize=12,
            rotation=0, labelpad=36, va="center",
        )

        # Mean: diverging bwr at ±clip
        vabs = max(3 * float(np.sqrt((mean_map**2).mean())), 1e-9)
        axes[row_idx, 0].imshow(ds_mean, cmap="bwr", vmin=-vabs, vmax=vabs,
                                 interpolation="nearest", aspect="auto")
        axes[row_idx, 0].set_xticks([]); axes[row_idx, 0].set_yticks([])

        # Std: sequential, 0 = stable
        axes[row_idx, 1].imshow(ds_std, cmap="hot_r",
                                 vmin=0, vmax=ds_std.max() or 1e-9,
                                 interpolation="nearest", aspect="auto")
        axes[row_idx, 1].set_xticks([]); axes[row_idx, 1].set_yticks([])

    plt.tight_layout(h_pad=0.3, w_pad=0.3)
    savefig(fig, output_dir / "snr_grid.png")


# ---------------------------------------------------------------------------
# Per-sensor strip  (1×6: 5 scenes + mean, higher resolution)
# ---------------------------------------------------------------------------


def build_sensor_strips(
    sensors: list[int],
    scenes: list[str],
    bias_dir: Path,
    output_dir: Path,
    downsample: int,
    smooth_sigma: float,
    clip_sigma: float,
) -> None:
    strip_factor = max(1, downsample // 4)   # 4× higher res than overview grid
    n_cols = len(scenes) + 1

    for sensor_id in sensors:
        raw_panels: list[np.ndarray | None] = [
            load_fpn(scene, sensor_id, bias_dir) for scene in scenes
        ]
        available = [p for p in raw_panels if p is not None]
        if not available:
            continue

        min_h = min(p.shape[0] for p in available)
        min_w = min(p.shape[1] for p in available)
        mean_raw = np.mean(
            [p[:min_h, :min_w] for p in available], axis=0
        ).astype(np.float32)

        rms  = float(np.sqrt((mean_raw ** 2).mean()))
        vabs = max(clip_sigma * rms, 1e-9)

        col_labels = [SCENE_LABELS.get(s, s) for s in scenes] + ["MEAN"]
        panels     = raw_panels + [mean_raw]

        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.5, 2.2))
        if n_cols == 1:
            axes = [axes]

        sensor_label = SENSOR_NAMES.get(sensor_id, f"S{sensor_id:02d}")
        fig.suptitle(
            f"Sensor {sensor_id:02d}  ({sensor_label})  —  FPN residual  "
            f"|  colour scale ±{clip_sigma:.0f}×RMS = ±{vabs:.5f}  "
            f"|  display smooth σ={smooth_sigma}",
            fontsize=15,
        )

        for ax, panel, label in zip(axes, panels, col_labels):
            bold = label == "MEAN"
            sigma = smooth_sigma if label != "MEAN" else smooth_sigma * 1.5
            ds = prepare(panel, strip_factor, sigma) if panel is not None else None
            render_panel(ax, ds, -vabs, vabs, title=label)
            if bold:
                for spine in ax.spines.values():
                    spine.set_edgecolor("gold")
                    spine.set_linewidth(1.5)

        plt.tight_layout()
        savefig(fig, output_dir / f"sensor_{sensor_id:02d}_strip.png")


# ---------------------------------------------------------------------------
# Per-sensor mean map (full-ish resolution)
# ---------------------------------------------------------------------------


def build_sensor_means(
    sensors: list[int],
    scenes: list[str],
    bias_dir: Path,
    output_dir: Path,
    downsample: int,
    smooth_sigma: float,
    clip_sigma: float,
) -> None:
    mean_factor = max(1, downsample // 2)

    fig_all, axes_all = plt.subplots(
        2, 7, figsize=(14, 5),
    )
    axes_all = axes_all.ravel()

    for idx, sensor_id in enumerate(sensors):
        available = [
            load_fpn(scene, sensor_id, bias_dir) for scene in scenes
        ]
        available = [p for p in available if p is not None]
        if not available:
            continue

        min_h = min(p.shape[0] for p in available)
        min_w = min(p.shape[1] for p in available)
        mean_raw = np.mean(
            [p[:min_h, :min_w] for p in available], axis=0
        ).astype(np.float32)

        rms  = float(np.sqrt((mean_raw ** 2).mean()))
        vabs = max(clip_sigma * rms, 1e-9)
        ds   = prepare(mean_raw, mean_factor, smooth_sigma * 1.5)
        sensor_label = SENSOR_NAMES.get(sensor_id, f"S{sensor_id:02d}")

        # Individual file (higher res)
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        im = ax1.imshow(ds, cmap="bwr", vmin=-vabs, vmax=vabs,
                        interpolation="nearest")
        ax1.set_title(
            f"S{sensor_id:02d} {sensor_label}  —  cross-scene mean FPN\n"
            f"RMS={rms:.5f}   clip=±{vabs:.5f}   smooth σ={smooth_sigma * 1.5:.1f}",
            fontsize=15,
        )
        ax1.set_xticks([]); ax1.set_yticks([])
        plt.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
        plt.tight_layout()
        savefig(fig1, output_dir / f"sensor_{sensor_id:02d}_mean.png")

        # Slot into all-sensors mosaic
        if idx < len(axes_all):
            ax = axes_all[idx]
            ax.imshow(ds, cmap="bwr", vmin=-vabs, vmax=vabs,
                      interpolation="nearest", aspect="auto")
            ax.set_title(f"S{sensor_id:02d} {sensor_label}", fontsize=13)
            ax.set_xticks([]); ax.set_yticks([])

    # Hide any unused mosaic cells
    for ax in axes_all[len(sensors):]:
        ax.set_visible(False)

    fig_all.suptitle(
        "Cross-scene mean FPN residual — all 14 sensors\n"
        "Each panel uses its own ±3×RMS colour scale",
        fontsize=16,
    )
    plt.tight_layout()
    savefig(fig_all, output_dir / "all_sensors_mean.png")


# ---------------------------------------------------------------------------
# Row / column FPN profile overlays
# ---------------------------------------------------------------------------


def build_profile_plots(
    sensors: list[int],
    scenes: list[str],
    bias_dir: Path,
    output_dir: Path,
) -> None:
    """
    For each sensor: plot the row-FPN and col-FPN profiles from all scenes
    overlaid.  Common structure across scenes = true fixed pattern.
    """
    for sensor_id in sensors:
        row_profiles, col_profiles = [], []
        scene_hit = []
        for scene in scenes:
            p = bias_dir / scene / f"sensor_{sensor_id}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            row_profiles.append(d["row_fpn"])
            col_profiles.append(d["col_fpn"])
            scene_hit.append(SCENE_LABELS.get(scene, scene))

        if not row_profiles:
            continue

        # Trim all to minimum length (cross-resolution safety)
        min_r = min(a.shape[0] for a in row_profiles)
        min_c = min(a.shape[0] for a in col_profiles)
        row_profiles = [a[:min_r] for a in row_profiles]
        col_profiles = [a[:min_c] for a in col_profiles]

        mean_row = np.mean(row_profiles, axis=0)
        mean_col = np.mean(col_profiles, axis=0)

        cmap = plt.get_cmap("tab10")
        sensor_label = SENSOR_NAMES.get(sensor_id, f"S{sensor_id:02d}")

        fig, (ax_r, ax_c) = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
        fig.suptitle(
            f"S{sensor_id:02d} {sensor_label}  —  Row & Column FPN profiles  "
            "(overlaid across scenes)",
            fontsize=16,
        )

        for i, (prof, label) in enumerate(zip(row_profiles, scene_hit)):
            ax_r.plot(prof, lw=0.6, alpha=0.6, color=cmap(i), label=label)
        ax_r.plot(mean_row, lw=1.5, color="black", label="mean", zorder=5)
        ax_r.axhline(0, color="grey", lw=0.5, ls="--")
        ax_r.set_ylabel("Row FPN (mean across cols)")
        ax_r.legend(fontsize=13, ncol=6, loc="upper right")
        ax_r.set_xlim(0, len(mean_row) - 1)

        for i, (prof, label) in enumerate(zip(col_profiles, scene_hit)):
            ax_c.plot(prof, lw=0.6, alpha=0.6, color=cmap(i), label=label)
        ax_c.plot(mean_col, lw=1.5, color="black", label="mean", zorder=5)
        ax_c.axhline(0, color="grey", lw=0.5, ls="--")
        ax_c.set_ylabel("Col FPN (mean across rows)")
        ax_c.set_xlabel("Pixel index")
        ax_c.set_xlim(0, len(mean_col) - 1)

        plt.tight_layout()
        savefig(fig, output_dir / f"sensor_{sensor_id:02d}_profiles.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise FPN residuals as grids and strips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sensors", nargs="+", type=int,
                        default=list(range(1, NUM_SENSORS + 1)), metavar="N")
    parser.add_argument("--scenes", nargs="+", default=ALL_SCENES,
                        choices=ALL_SCENES, metavar="SCENE")
    parser.add_argument("--bias-dir", type=Path, default=BIAS_DIR, metavar="DIR")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, metavar="DIR")
    parser.add_argument("--downsample", type=int, default=16, metavar="N",
                        help="Spatial downsample factor for overview grid (default 16)")
    parser.add_argument("--smooth-sigma", type=float, default=3.0, metavar="S",
                        help="Gaussian display blur sigma after downsampling (0=off, default 3)")
    parser.add_argument("--clip-sigma", type=float, default=3.0, metavar="N",
                        help="Colormap clip at ±N×RMS of the per-sensor mean (default 3)")
    parser.add_argument("--skip-strips", action="store_true",
                        help="Skip per-sensor strip figures (faster)")
    parser.add_argument("--skip-profiles", action="store_true",
                        help="Skip row/column FPN profile plots")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Overview grid (%d sensors × %d scenes) …",
             len(args.sensors), len(args.scenes))
    build_overview_grid(
        args.sensors, args.scenes, args.bias_dir, args.output_dir,
        args.downsample, args.smooth_sigma, args.clip_sigma,
    )

    log.info("SNR grid (mean | std across scenes) …")
    build_snr_grid(
        args.sensors, args.scenes, args.bias_dir, args.output_dir,
        args.downsample, args.smooth_sigma,
    )

    log.info("Per-sensor mean maps …")
    build_sensor_means(
        args.sensors, args.scenes, args.bias_dir, args.output_dir,
        args.downsample, args.smooth_sigma, args.clip_sigma,
    )

    if not args.skip_strips:
        log.info("Per-sensor strips …")
        build_sensor_strips(
            args.sensors, args.scenes, args.bias_dir, args.output_dir,
            args.downsample, args.smooth_sigma, args.clip_sigma,
        )

    if not args.skip_profiles:
        log.info("Row/column FPN profiles …")
        build_profile_plots(
            args.sensors, args.scenes, args.bias_dir, args.output_dir,
        )

    log.info("Done. All figures in %s", args.output_dir)


if __name__ == "__main__":
    main()
