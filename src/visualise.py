"""
visualise.py
============
Phase 4: Generate publication-quality figures from bias maps and analysis results.

Requires results from all prior pipeline steps.

Figures produced
----------------
results/figures/bias_heatmaps/
    {scene}_sensor_{n}.png       — bias map heatmap (clipped to ±2σ)

results/figures/
    scene_independence.png       — per-sensor cross-scene Pearson r matrix
    sensor_similarity.png        — 14×14 cross-sensor similarity heatmap
    sensor_dendrogram.png        — hierarchical clustering dendrogram
    row_psd.png                  — row-direction PSD overlay (all sensors)
    col_psd.png                  — column-direction PSD overlay (all sensors)
    fpn_summary.png              — mean RMS FPN per sensor (bar chart)
    temporal_stability.png       — extra-noisy vs noisy frame variance comparison

Usage
-----
    python src/visualise.py [options]

Examples
--------
    python src/visualise.py
    python src/visualise.py --sensors 1 2 3 --scenes color_1lx_120
    python src/visualise.py --skip-heatmaps   # fast: skip per-sensor maps
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.cluster.hierarchy import dendrogram

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

# Sensor numbering inferred from Table 4 column order in the challenge paper:
# Yakovenko et al., 2025 (arXiv:2508.16830).  Not officially published.
SENSOR_NAMES: dict[int, str] = {
    1:  "Fold4-W",
    2:  "Fold4-UW",
    3:  "Fold4-T",
    4:  "Pixel5a-W",
    5:  "Pixel5a-UW",
    6:  "Pixel5a-F",
    7:  "Pixel7Pro-W",
    8:  "Pixel7Pro-UW",
    9:  "Pixel7Pro-T",
    10: "Pixel7Pro-F",
    11: "S20-W",
    12: "S20-UW",
    13: "POCO-W",
    14: "POCO-F",
}

BIAS_DIR: Path = Path("results") / "bias_maps"
STATS_DIR: Path = Path("results") / "statistics"
ANALYSIS_DIR: Path = Path("results") / "analysis"
FIGURES_DIR: Path = Path("results") / "figures"

DPI: int = 200
CMAP_BIAS: str = "bwr"       # diverging — shows positive/negative bias
CMAP_CORR: str = "RdYlGn"   # correlation matrices

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
# Shared utilities
# ---------------------------------------------------------------------------


def savefig(fig: plt.Figure, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", **kwargs)
    plt.close(fig)
    log.info("Saved → %s", path)


def sensor_label(sensor_id: int) -> str:
    """Return a short descriptive label for a sensor, e.g. 'Fold4-T'."""
    return SENSOR_NAMES.get(sensor_id, f"S{sensor_id:02d}")


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------


def plot_bias_heatmaps(
    sensors: list[int],
    scenes: list[str],
    bias_dir: Path,
    out_dir: Path,
) -> None:
    """One heatmap per sensor/scene, colour range clipped to ±2σ of the map."""
    heatmap_dir = out_dir / "bias_heatmaps"
    for scene in scenes:
        for sensor_id in sensors:
            path = bias_dir / scene / f"sensor_{sensor_id}.npz"
            if not path.exists():
                continue
            bias_map = np.load(path)["bias_map"].astype(np.float64)

            sigma = bias_map.std()
            vmax = 2 * sigma
            vmin = -vmax

            fig, ax = plt.subplots(figsize=(10, 7.5))
            im = ax.imshow(bias_map, cmap=CMAP_BIAS, vmin=vmin, vmax=vmax, aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Bias (clipped ±2σ)")
            ax.set_title(
                f"Bias map — {scene} / sensor_{sensor_id}\n"
                f"σ = {sigma:.5f}  |  range [{bias_map.min():.4f}, {bias_map.max():.4f}]",
                fontsize=10,
            )
            ax.axis("off")
            savefig(fig, heatmap_dir / f"{scene}_sensor_{sensor_id}.png")


def plot_scene_independence(
    analysis_dir: Path,
    out_dir: Path,
    scenes: list[str],
) -> None:
    """Grid of cross-scene correlation matrices, one subplot per sensor."""
    si_path = analysis_dir / "scene_independence.npz"
    if not si_path.exists():
        log.warning("scene_independence.npz not found — skipping.")
        return

    d = np.load(si_path)
    correlations = d["correlations"]   # (S, 5, 5)
    sensor_ids = d["sensor_ids"].tolist()
    n = len(sensor_ids)
    scene_labels = [s.replace("_1lx_120", "") for s in scenes]

    ncols = 7
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axes = np.array(axes).ravel()

    for ax in axes[n:]:
        ax.axis("off")

    for idx, sensor_id in enumerate(sensor_ids):
        corr = correlations[idx]
        im = axes[idx].imshow(corr, cmap=CMAP_CORR, vmin=-1, vmax=1)
        axes[idx].set_title(f"S{sensor_id:02d}", fontsize=9)
        axes[idx].set_xticks(range(len(scenes)))
        axes[idx].set_yticks(range(len(scenes)))
        axes[idx].set_xticklabels(scene_labels, rotation=45, ha="right", fontsize=6)
        axes[idx].set_yticklabels(scene_labels, fontsize=6)

    fig.suptitle("Cross-Scene Pearson r of Bias Maps (per Sensor)", fontsize=13, y=1.01)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=-1, vmax=1), cmap=CMAP_CORR),
        ax=axes[:n],
        fraction=0.015,
        pad=0.04,
        label="Pearson r",
    )
    savefig(fig, out_dir / "scene_independence.png")


def plot_sensor_similarity(analysis_dir: Path, out_dir: Path) -> None:
    """14×14 cross-sensor similarity heatmap (scene-averaged Pearson r)."""
    path = analysis_dir / "sensor_similarity.npz"
    if not path.exists():
        log.warning("sensor_similarity.npz not found — skipping.")
        return

    d = np.load(path)
    mean_sim = d["mean_sim"]          # (14, 14)
    sensor_ids = d["sensor_ids"].tolist()
    labels = [sensor_label(s) for s in sensor_ids]

    fig, ax = plt.subplots(figsize=(8, 7.5))
    im = ax.imshow(mean_sim, cmap=CMAP_CORR, vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r (scene avg.)")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Cross-Sensor Bias Similarity (scene-averaged Pearson r)", fontsize=12)

    # Annotate cells
    for i in range(len(sensor_ids)):
        for j in range(len(sensor_ids)):
            val = mean_sim[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="black" if abs(val) < 0.6 else "white",
                )

    savefig(fig, out_dir / "sensor_similarity.png")


def plot_dendrogram(analysis_dir: Path, out_dir: Path) -> None:
    """Hierarchical clustering dendrogram of sensors."""
    path = analysis_dir / "clustering.npz"
    if not path.exists():
        log.warning("clustering.npz not found — skipping.")
        return

    d = np.load(path)
    Z = d["linkage_matrix"]
    sensor_ids = d["sensor_ids"].tolist()
    labels = [sensor_label(s) for s in sensor_ids]

    fig, ax = plt.subplots(figsize=(8, 7))
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=45, leaf_font_size=9)
    ax.set_title("Hierarchical Clustering of Sensors by Bias Fingerprint (Ward linkage)", fontsize=12)
    ax.set_ylabel("Euclidean distance")
    savefig(fig, out_dir / "sensor_dendrogram.png")


def plot_psd(analysis_dir: Path, out_dir: Path) -> None:
    """Row- and column-direction PSD overlay for all sensors."""
    path = analysis_dir / "psd.npz"
    if not path.exists():
        log.warning("psd.npz not found — skipping.")
        return

    d = np.load(path)
    row_psds = d["row_psd"]           # (S, F)
    col_psds = d["col_psd"]           # (S, F)
    row_freqs = d["row_freqs"]
    col_freqs = d["col_freqs"]
    sensor_ids = d["sensor_ids"].tolist()

    cmap = plt.get_cmap("tab20", len(sensor_ids))
    line_styles = ["-", "--", "-.", ":"] * 4  # cycle for grayscale readability

    for direction, freqs, psds, fname in [
        ("Row", row_freqs, row_psds, "row_psd.png"),
        ("Column", col_freqs, col_psds, "col_psd.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6))
        for idx, sensor_id in enumerate(sensor_ids):
            ax.semilogy(freqs, psds[idx], color=cmap(idx),
                        linestyle=line_styles[idx], linewidth=0.9,
                        label=sensor_label(sensor_id))
        ax.set_xlabel("Normalised frequency (cycles/sample)", fontsize=10)
        ax.set_ylabel("Power spectral density (log scale)", fontsize=10)
        ax.set_title(f"{direction}-direction PSD of Bias Maps \u2014 All Sensors", fontsize=11)
        ax.legend(
            ncol=3,
            fontsize=7,
            loc="upper right",
            framealpha=0.85,
            edgecolor="#cccccc",
        )
        ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=9)
        savefig(fig, out_dir / fname)


def plot_fpn_summary(analysis_dir: Path, out_dir: Path) -> None:
    """Bar chart: mean RMS FPN per sensor, error bars = std across scenes."""
    path = analysis_dir / "fpn_summary.npz"
    if not path.exists():
        log.warning("fpn_summary.npz not found — skipping.")
        return

    d = np.load(path)
    rms_fpn = d["rms_fpn"]            # (S, 5)
    sensor_ids = d["sensor_ids"].tolist()
    labels = [sensor_label(s) for s in sensor_ids]

    mean_rms = np.nanmean(rms_fpn, axis=1)
    std_rms = np.nanstd(rms_fpn, axis=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    x = np.arange(len(sensor_ids))
    bars = ax.bar(x, mean_rms, yerr=std_rms, capsize=3, color="#2c5f8a",
                  edgecolor="#1a3a54", linewidth=0.6, error_kw={"elinewidth": 1.0, "capthick": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RMS FPN  (pixel_mean \u2212 gt)", fontsize=10)
    ax.set_title("Mean RMS Fixed-Pattern Noise per Sensor\n(error bars = std across scenes)", fontsize=11)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9)

    # Annotate bar tops
    for bar, val in zip(bars, mean_rms):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std_rms[0] * 0.1,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=7,
        )

    savefig(fig, out_dir / "fpn_summary.png")


def plot_temporal_stability(
    sensors: list[int],
    scenes: list[str],
    stats_dir: Path,
    out_dir: Path,
) -> None:
    """
    Compare per-pixel variance from standard noisy frames vs extra-noisy frames.
    Higher extra-noisy variance confirms the frames are genuinely noisier.
    """
    noisy_vars, extra_vars, labels = [], [], []

    for sensor_id in sensors:
        nv, ev = [], []
        for scene in scenes:
            path = stats_dir / scene / f"sensor_{sensor_id}.npz"
            if not path.exists():
                continue
            d = np.load(path)
            nv.append(float(d["pixel_var"].mean()))
            ev.append(float(d["extra_var"].mean()))
        if nv:
            noisy_vars.append(np.mean(nv))
            extra_vars.append(np.mean(ev))
            labels.append(sensor_label(sensor_id))

    if not labels:
        log.warning("No data for temporal stability plot — skipping.")
        return

    n = len(labels)
    x = np.arange(n)
    w = 0.38

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar(x - w / 2, noisy_vars, width=w, label="Noisy frames", color="#2c5f8a",
           edgecolor="#1a3a54", linewidth=0.6)
    ax.bar(x + w / 2, extra_vars, width=w, label="Extra-noisy frames", color="#c75146",
           edgecolor="#8c2e25", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean pixel-wise temporal variance", fontsize=10)
    ax.set_title("Temporal Variance: Noisy vs Extra-Noisy Frames\n(scene-averaged)", fontsize=11)
    ax.legend(fontsize=9, edgecolor="#cccccc", framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    savefig(fig, out_dir / "temporal_stability.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures from pipeline results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--scenes", nargs="+", default=ALL_SCENES, choices=ALL_SCENES, metavar="SCENE")
    parser.add_argument("--sensors", nargs="+", type=int,
                        default=list(range(1, NUM_SENSORS + 1)), metavar="N")
    parser.add_argument("--bias-dir", type=Path, default=BIAS_DIR, metavar="DIR")
    parser.add_argument("--stats-dir", type=Path, default=STATS_DIR, metavar="DIR")
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR, metavar="DIR")
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR, metavar="DIR")
    parser.add_argument(
        "--skip-heatmaps",
        action="store_true",
        help="Skip individual bias heatmap images (saves time for large datasets).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_heatmaps:
        log.info("=== Bias heatmaps ===")
        plot_bias_heatmaps(args.sensors, args.scenes, args.bias_dir, args.output_dir)

    log.info("=== Scene independence matrix ===")
    plot_scene_independence(args.analysis_dir, args.output_dir, args.scenes)

    log.info("=== Sensor similarity heatmap ===")
    plot_sensor_similarity(args.analysis_dir, args.output_dir)

    log.info("=== Clustering dendrogram ===")
    plot_dendrogram(args.analysis_dir, args.output_dir)

    log.info("=== PSD plots ===")
    plot_psd(args.analysis_dir, args.output_dir)

    log.info("=== FPN summary ===")
    plot_fpn_summary(args.analysis_dir, args.output_dir)

    log.info("=== Temporal stability ===")
    plot_temporal_stability(args.sensors, args.scenes, args.stats_dir, args.output_dir)

    log.info("All figures saved to %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
