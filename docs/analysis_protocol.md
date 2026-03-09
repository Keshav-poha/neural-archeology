# Analysis Protocol: Sensor Bias Study (AIM 2025 RAW Data)

## Research Goal

Identify and characterise **scene-independent, sensor-specific bias** in RAW low-light data.

Bias is defined as:
> A persistent, statistically significant pattern that remains stable across scenes for the same sensor.

---

## Scope

This study is strictly limited to **measurement, comparison, and visualisation** of sensor data.

Causality must not be inferred, and no learning-based methods may be applied, unless explicitly documented as a separate experimental step.

---

## Methodology Restrictions

The following operations are **out of scope** for the core bias analysis:

- Applying ISP operations of any kind
- Denoising prior to measurement
- Normalising or rescaling raw pixel values for analysis purposes
- Debayering (unless explicitly noted as a separate step)
- Converting to RGB colour space (unless explicitly noted)
- Training or evaluating statistical or machine-learning models
- Inferring scene content or object classes
- Speculating on physical causes without supporting evidence

All measurements must operate on the **native, unmodified RAW sensor values**.

---

## Dataset

- **Source:** AIM 2025 RAW Video Denoising Challenge
- **Data type:** RAW sensor output (pre-ISP)
- **Sensors:** 14
- **Scenes:** 5
- **Lighting:** ~1 lux
- **Frame rate:** 120 fps
- **Exposure:** 1/120 s
- **Frames per scene:**
  - 10 noisy frames (primary)
  - Additional extra-noisy frames (stability validation)

---

## Sensor Inventory

The 14 sensors originate from five smartphones.
Sensor numbering (`sensor_1` – `sensor_14`) is inferred from the column order of
the cross-sensor performance table (Table 4) in the challenge paper
(Yakovenko et al., 2025; arXiv:2508.16830). The dataset does not publish an
official numeric-to-model mapping.

| # | Device | Module | Max ISO | Resolution (W×H) |
|---|--------|--------|---------|-------------------|
| 1 | Samsung Galaxy Z Fold4 | Wide (W) | 1600 | 4080 × 3060 |
| 2 | Samsung Galaxy Z Fold4 | Ultrawide (UW) | 10000 | 4080 × 3060 |
| 3 | Samsung Galaxy Z Fold4 | Telephoto (T) | 10000 | 3648 × 2736 |
| 4 | Google Pixel 5a | Wide (W) | 7109 | 4032 × 3024 |
| 5 | Google Pixel 5a | Ultrawide (UW) | 9208 | 4032 × 3022 |
| 6 | Google Pixel 5a | Front (F) | 10000 | 3280 × 2464 |
| 7 | Google Pixel 7 Pro | Wide (W) | 10000 | 4080 × 3072 |
| 8 | Google Pixel 7 Pro | Ultrawide (UW) | 10000 | 4080 × 3072 |
| 9 | Google Pixel 7 Pro | Telephoto (T) | 1143 | 4080 × 3072 |
| 10 | Google Pixel 7 Pro | Front (F) | 3918 | 3440 × 2448 |
| 11 | Samsung Galaxy S20 | Wide (W) | 9993 | 4000 × 3000 |
| 12 | Samsung Galaxy S20 | Ultrawide (UW) | 10000 | 4000 × 3000 |
| 13 | POCO X3 Pro | Wide (W) | 10000 | 4000 × 3000 |
| 14 | POCO X3 Pro | Front (F) | 10000 | 2592 × 1944 |

**Note:** Sensor 3 (Galaxy Z Fold4 Telephoto, 3648 × 2736) matches the array
shape `(2736, 3648)` of the `.npz` files observed in this dataset subset.

### Reference

Yakovenko, A. et al. *AIM 2025 Low-light RAW Video Denoising Challenge: Dataset, Methods and Results.*
ICCVW 2025. arXiv:2508.16830. Available at:
<https://www.codabench.org/competitions/8729/>