<div align="center">

# 🧬 Neural Archaeology

**Sensor-Specific Bias in RAW Low-Light Data**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-AIM%202025-green.svg)](https://www.codabench.org/competitions/8729/)

*Research code for identifying and characterising **scene-independent, sensor-specific biases** in pre-ISP RAW image data captured under extreme low-light conditions.*

**[📄 Paper](paper/main.tex)** · **[📊 Analysis Protocol](docs/analysis_protocol.md)** · **[📝 Blog Walkthrough](docs/blog.txt)**

</div>

---

## Overview

Every camera sensor has a fingerprint — a persistent, structured pattern of Fixed-Pattern Noise (FPN) that is baked into the hardware and has nothing to do with the scene being photographed. At ~1 lux illumination, this deterministic noise often **dominates** the true optical signal.

This repository provides:
- A **4-phase analysis pipeline** that extracts, quantifies, and visualises FPN from RAW sensor data
- **Cross-sensor comparison** revealing unexpected structural affinities across brands
- A **companion research paper** in IEEE format with full graphical analysis

> **Key finding:** FPN severity varies by up to **22×** between modules in the same handset (e.g., Samsung Galaxy S20 Wide vs. Ultrawide), demonstrating that noise is a property of the individual silicon die, not the device firmware.

---

## Repository Structure

```
neural-archeology/
├── paper/                       # Research paper (IEEE LaTeX)
│   ├── main.tex                 # Main manuscript
│   ├── references.bib           # Bibliography (14 references)
│   └── figures/                 # All paper figures (7 PNGs)
│
├── src/                         # Analysis pipeline
│   ├── compute_statistics.py    # Phase 1 — per-pixel temporal statistics
│   ├── extract_bias.py          # Phase 2 — FPN extraction + scene-independence test
│   ├── analyse_bias.py          # Phase 3 — cross-sensor comparison, clustering, PSD
│   ├── visualise.py             # Phase 4 — publication figures
│   ├── visualise_fpn_grid.py    # FPN overview grids and per-sensor strips
│   └── decode_and_save.py       # Utility — .npz frames → grayscale PNGs
│
├── docs/
│   ├── analysis_protocol.md     # Analysis scope, restrictions, methodology
│   └── blog.txt                 # Plain-English research walkthrough
│
├── data/                        # Raw .npz archives (NOT in repo — see below)
├── results/                     # Pipeline outputs (NOT in repo — regenerated)
│
├── requirements.txt             # Python dependencies
├── CITATION.cff                 # Citation metadata
├── LICENSE                      # MIT License
└── README.md
```

---

## Dataset

The raw data comes from the **AIM 2025 Low-Light RAW Video Denoising Challenge**:

> 🔗 **Download:** https://www.codabench.org/competitions/8729/ (registration required)

| Property       | Value                     |
|----------------|---------------------------|
| **Sensors**    | 14 (across 5 smartphones) |
| **Scenes**     | 5 tabletop subjects       |
| **Lighting**   | ~1 lux                    |
| **Frame rate** | 120 fps                   |
| **Exposure**   | 1/120 s                   |
| **Format**     | Pre-ISP RAW (`.npz`)      |

### Camera Sensors

The 14 sensors span five smartphones. Sensor numbering is inferred from the column order in Table 4 of the challenge paper (Yakovenko et al., 2025; [arXiv:2508.16830](https://arxiv.org/abs/2508.16830)).

| # | Device | Module | Resolution (W×H) |
|---|--------|--------|-------------------|
| 1 | Samsung Galaxy Z Fold4 | Wide | 4080 × 3060 |
| 2 | Samsung Galaxy Z Fold4 | Ultrawide | 4080 × 3060 |
| 3 | Samsung Galaxy Z Fold4 | Telephoto | 3648 × 2736 |
| 4 | Google Pixel 5a | Wide | 4032 × 3024 |
| 5 | Google Pixel 5a | Ultrawide | 4032 × 3022 |
| 6 | Google Pixel 5a | Front | 3280 × 2464 |
| 7 | Google Pixel 7 Pro | Wide | 4080 × 3072 |
| 8 | Google Pixel 7 Pro | Ultrawide | 4080 × 3072 |
| 9 | Google Pixel 7 Pro | Telephoto | 4080 × 3072 |
| 10 | Google Pixel 7 Pro | Front | 3440 × 2448 |
| 11 | Samsung Galaxy S20 | Wide | 4000 × 3000 |
| 12 | Samsung Galaxy S20 | Ultrawide | 4000 × 3000 |
| 13 | POCO X3 Pro | Wide | 4000 × 3000 |
| 14 | POCO X3 Pro | Front | 2592 × 1944 |

After downloading, place files at `data/{scene}/sensor_{n}.npz`:
```
data/
├── color_1lx_120/
├── hardware_1lx_120/
├── spring_1lx_120/
├── toys_1lx_120/
└── yarn_1lx_120/
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Keshav-poha/neural-archeology.git
cd neural-archeology

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Phase 1 — Compute per-pixel temporal statistics
python src/compute_statistics.py

# Phase 2 — Extract FPN bias maps and test scene independence
python src/extract_bias.py

# Phase 3 — Cross-sensor similarity, clustering, PSD, FPN summary
python src/analyse_bias.py

# Phase 4 — Generate all publication figures
python src/visualise.py

# (Optional) FPN overview grids
python src/visualise_fpn_grid.py
```

All scripts accept `--scenes`, `--sensors`, `--help`, and directory overrides. See individual script docstrings for details.

### 3. Visual Inspection (Optional)

```bash
# Convert .npz frames to grayscale PNGs
python src/decode_and_save.py

# Specific scenes/sensors
python src/decode_and_save.py --scenes color_1lx_120 --sensors 1 3 5
```

---

## Key Results

| Sensor | Module | RMS FPN | Note |
|--------|--------|---------|------|
| S12 | S20 Ultrawide | 0.0345 | **Highest** — dominates all others |
| S14 | POCO Front | 0.0229 | |
| S08 | Pixel 7 Pro UW | 0.0200 | |
| S11 | S20 Wide | 0.0016 | **Lowest** — same phone as S12 |

> The full 14-sensor breakdown with error bars is in the paper (Table I).

### Research Questions

1. ✅ What persistent statistical patterns characterise each sensor's output?
2. ✅ Are those patterns stable across scenes (scene-independent)?
3. 🔮 Can scene-independent correction functions be derived without ISP operations? *(future work)*

---

## Analysis Protocol

All analyses follow a strict **zero-ISP** protocol:

- ❌ No demosaicking or Bayer interpolation
- ❌ No denoising before measurement
- ❌ No white balance, tone curves, or sharpening
- ❌ No normalisation or rescaling of pixel values
- ❌ No machine learning models
- ✅ Native RAW sensor values only

See [`docs/analysis_protocol.md`](docs/analysis_protocol.md) for the full specification.

---

## Citation

If you use this code or reference this research, please cite:

```bibtex
@article{keshav2025neural,
  title={Reading the Silicon: A Critical Visual Analysis of Hardware Fingerprints in Low-Light RAW Video},
  author={Keshav},
  year={2025},
  note={Netaji Subhas University of Technology}
}
```

---

## Acknowledgments

- **Dataset:** [AIM 2025 Low-Light RAW Video Denoising Challenge](https://www.codabench.org/competitions/8729/) — Yakovenko et al. (ICCVW 2025, [arXiv:2508.16830](https://arxiv.org/abs/2508.16830))
- Sensor forensics foundations: Lukas et al. (2006), Fridrich (2009)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Author:** Keshav · [keshav.poha@gmail.com](mailto:keshav.poha@gmail.com)
**Affiliation:** Netaji Subhas University of Technology, New Delhi

</div>
