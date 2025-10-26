# Bayesian Updating for 1‑D Consolidation (Grid Posterior, Lognormal Priors)

This repository provides a **comprehensive and transparent example** of Bayesian updating for the one‑dimensional consolidation problem in geotechnical engineering.  
It demonstrates how to sequentially update soil parameters using settlement observations, showing **how information from monitoring data progressively refines model uncertainty**.

The implementation uses a **grid‑based Bayesian approach**—completely reproducible and visualized step‑by‑step—ideal for teaching, research, and verification of more advanced stochastic or MCMC‑based frameworks.

---

## 🔍 Overview

The procedure estimates the two key parameters controlling settlement in soft clays:

- **Coefficient of volume compressibility** \(m_v\) \([\text{kPa}^{-1}]\)
- **Coefficient of consolidation** \(c_v\) \([\text{m}^2/\text{day}]\)

These parameters are treated as **random variables** with independent **lognormal priors**. As field settlement data \(y_i\) become available at times \(t_i\), Bayesian inference updates their probability distributions.

### Governing equations

**1‑D Consolidation settlement:**

\[
s(t; m_v, c_v) = 110{,}000\, m_v \, U(T_v), \quad 
U(T_v) = 1 - \frac{8}{\pi^2}\,\exp\!\Big(-\frac{\pi^2}{4}\,T_v\Big), \quad
T_v = \frac{c_v t}{(H/2)^2}
\]

Where:  
- \(H = 5\,\text{m}\) → \(H/2 = 2.5\,\text{m}\)  
- \(\Delta \sigma' = 22\,\text{kPa}\)  
- \(\sigma_e = 3\,\text{mm}\) (measurement noise)

**Bayesian updating equation:**

\[
p(\theta \mid \mathbf{y}) \propto
\Bigg[ \prod_i \phi\!\left(\frac{y_i - s(t_i;\theta)}{\sigma_e}\right) \Bigg] p(\theta),
\quad \theta = (m_v, c_v)
\]

where \(\phi(\cdot)\) is the standard normal probability density function.

---

## 🧩 Implementation Summary

| Step | Description |
|------|--------------|
| **1. Define grid** | Construct a 161×161 log‑spaced grid of \((m_v, c_v)\). |
| **2. Compute priors** | Evaluate independent lognormal PDFs for \(m_v\) and \(c_v\). |
| **3. Compute likelihood** | For each grid point, compute settlement \(s(t_i;\theta)\) and likelihood \(\phi((y_i - s)/\sigma_e)\). |
| **4. Form posterior** | Multiply prior and likelihood; normalize with log‑sum‑exp for numerical stability. |
| **5. Posterior statistics** | Compute posterior mean, MAP (mode), marginals, and 95% credible intervals. |
| **6. Sequential updating** | Repeat for \(k=1\), \(2\), \(3\), \(4\) observations to illustrate uncertainty reduction. |
| **7. Visualization** | Produce 2‑D and 3‑D posteriors, marginal plots, and parameter evolution tables. |

---

## 🖥️ Running the Example

### Requirements

- Python ≥ 3.9  
- `numpy`, `pandas`, `matplotlib`

No LaTeX installation is required — all equations are rendered via Matplotlib’s internal mathtext.

### How to Run

1. Clone this repository or copy the script:
   ```bash
   git clone https://github.com/YourUsername/Bayesian_Consolidation.git
   cd Bayesian_Consolidation
   ```
2. Run the main script:
   ```bash
   python bayes_consolidation_sequential_viz.py
   ```
3. Figures will **automatically display** in the correct order and be saved to:
   ```
   ./outputs_bayes_consolidation/
   ```

---

## 📊 Key Outputs (in final display order)

| Order | File | Description |
|-------|------|-------------|
| 1 | `equation_obs4_worked_example.png` | Final 4‑observation worked example |
| 2 | `fig_joint_posterior_heatmap.png` | Posterior joint heatmap (all data) |
| 3 | `k1_post_surface.png` | 3‑D posterior (k=1) — broad, skewed surface |
| 4 | `k4_post_surface.png` | 3‑D posterior (k=4) — concentrated peak |
| 5 | `fig_mv_prior_posterior.png` | Prior vs posterior marginal (m_v) |
| 6 | `fig_cv_prior_posterior.png` | Prior vs posterior marginal (c_v) |
| 7 | `equation_summary.png` | Bayesian updating formula summary |
| 8 | `param_summary.png` | Final table of prior and posterior parameters |

Each intermediate step (`step1_first_obs.png`, `equation_obs1.png`, etc.) shows how new measurements refine uncertainty.

---

## 🔬 Interpretation Guide

### Sequential updating

- **k=1**: posterior is broad; mean and MAP differ due to skewness.  
- **k=2–3**: posterior tightens as data reduce uncertainty.  
- **k=4**: posterior sharply peaks around true parameters, producing a narrow 95% credible interval.

### Mean vs MAP distinction

- **MAP (mode)**: highest probability point (peak of the surface).  
- **Mean (expected value)**: “center of mass” of the surface — sensitive to skewness.  
- Their difference is visualized explicitly on all 3‑D surfaces.

### Marginal plots

- The marginal PDFs of \(m_v\) and \(c_v\) show how uncertainty collapses relative to the prior.  
- 95% credible intervals (dotted lines) quantify the posterior range of plausible values.

---

## ⚙️ Numerical Considerations

- Posterior normalization uses the **log‑sum‑exp trick** to avoid numerical underflow.  
- The **grid posterior** approach is exact for low‑dimensional problems and provides interpretable visualizations without stochastic noise.  
- Priors can be modified to reflect project‑specific knowledge or alternative soil models.

---

## 📘 Building a Consolidated PDF Report

After running the main script, you can compile all generated figures (in order) into a single PDF:

```bash
python build_summary_pdf.py
```
This will create:

```
outputs_bayes_consolidation/Consolidation_Bayes_Summary.pdf
```

The PDF serves as a **visual report** of the entire Bayesian updating process — ideal for teaching, presentations, or supplementary journal material.

---

## 🧱 Citation & Applications

If you use this repository in academic or professional work, please cite it as:

> *Bayesian Updating for 1‑D Consolidation (2025). Demonstration code for sequential parameter updating using grid‑based posterior inference in geotechnical engineering.*

Applications include:
- Settlement prediction and back‑analysis of test embankments.  
- Bayesian calibration of soil models.  
- Comparison benchmark for MCMC or surrogate‑model Bayesian methods.

---

## 👨‍💻 Repository Structure

```
.
├── bayes_consolidation_sequential_viz.py    # Main sequential Bayesian update script
├── build_summary_pdf.py                     # Combines figures into a single PDF report
├── README.md                                # This file
└── outputs_bayes_consolidation/             # Folder containing generated figures
```

---

## 💡 Tips for Extending

- Replace the first‑term approximation with a multi‑term or large‑strain consolidation model.  
- Introduce pore pressure data and jointly update mechanical and hydraulic parameters.  
- Wrap this solver inside an MCMC sampler (e.g., DREAM‑ZS or DE‑Metropolis) for higher‑dimensional inference.

---

**Author:** Merrick Jones — 2025  
**Affiliation:** University of Newcastle / BECA Consulting  
**License:** MIT

