# Bayesian Updating for One-Dimensional Consolidation (Grid Posterior, Lognormal Priors)

This repository provides a **comprehensive and transparent example** of Bayesian updating for the one-dimensional consolidation problem in geotechnical engineering.  
It demonstrates how field monitoring data can be used to **progressively update soil parameters** controlling consolidation behaviour, showing how Bayesian inference systematically reduces uncertainty.

The implementation uses a **grid-based Bayesian approach**—a deterministic alternative to MCMC—making the inference process fully reproducible and easy to visualise.

---

## 🧩 Overview

The program performs sequential Bayesian updating for the parameters governing primary settlement of soft clay soils:

$$
\begin{aligned}
m_v &= \text{coefficient of volume compressibility }(1/\mathrm{kPa}) \\
c_v &= \text{coefficient of vertical consolidation }(\mathrm{m^2/day})
\end{aligned}
$$

Both parameters are treated as random variables with **independent lognormal priors**.  
As settlement measurements \(y_i\) are obtained at times \(t_i\), their joint posterior distribution is updated according to Bayes’ theorem.

---

## ⚙️ Forward Consolidation Model

The settlement of a clay layer of thickness \(H = 5\,\mathrm{m}\) under an effective stress increment \(\Delta\sigma' = 22\,\mathrm{kPa}\) is computed using the first-term approximation of Terzaghi’s 1-D consolidation theory:

$$
\begin{aligned}
s(t; m_v, c_v) &= 110{,}000\,m_v\,U(T_v) \\
U(T_v) &= 1 - \frac{8}{\pi^2}\exp\\left(-\frac{\pi^2}{4}\,T_v\right) \\
T_v &= \frac{c_v\,t}{(H/2)^2}
\end{aligned}
$$

The model assumes **double drainage** (\(H_d = H/2 = 2.5\,\mathrm{m}\)) and expresses settlement in millimetres.  
For small-strain primary consolidation, this approximation is accurate and computationally efficient.

---

## 🧠 Bayesian Formulation

### Likelihood Function

Each measured settlement \(y_i\) is assumed to be normally distributed about the model prediction \(s(t_i; m_v, c_v)\):

$$
\mathcal{L}(\mathbf{y}\,|\,m_v, c_v)
= \prod_i \frac{1}{\sqrt{2\pi}\sigma_e}
\exp\!\left[-\frac{(y_i - s(t_i;m_v,c_v))^2}{2\sigma_e^2}\right]
$$

where \(\sigma_e = 3\,\mathrm{mm}\) represents measurement error.

### Posterior Distribution

The posterior is obtained by combining the likelihood and prior probabilities:

$$
p(m_v, c_v \mid \mathbf{y}) \propto
\mathcal{L}(\mathbf{y}\,|\,m_v, c_v)\;
p(m_v)\,p(c_v)
$$

Since both priors are lognormal, positivity is enforced automatically.

---

## 🔢 Discrete Grid Implementation

Instead of Monte Carlo sampling, the posterior is computed on a **161 × 161 log-spaced grid** in `(m_v, c_v)`.

---

### 1. Grid definition

The parameter domains are defined as:

$$
m_v \in [3\times10^{-4},\,3\times10^{-3}]\, (1/\mathrm{kPa}), \quad
c_v \in [0.01,\,0.10]\, (\mathrm{m^2/day})
$$

---

### 2. Prior evaluation

Independent lognormal probability density functions are evaluated as:

$$
p(m_v, c_v) = p(m_v)\,p(c_v)
$$

---

### 3. Likelihood evaluation

For each grid point, compute the forward model

$$
\( s(t_i; m_v, c_v) \)  
$$

and evaluate the Gaussian likelihood term:

$$
L_{ij} = \prod_i \phi \Big( \frac{y_i - s(t_i; m_v, c_v)}{\sigma_e} \Big)
$$

---

### 4. Posterior normalization (log-sum-exp)

The posterior is formed by multiplying the prior and likelihood, then normalizing:

$$
p_{ij} =
\exp \Big( [\ln p^{prior}_{ij} + \ln L_{ij}] - \ln Z \Big)
$$

where the normalization constant \( Z \) is computed as:

$$
\ln Z =
\ln \Big( \sum_{i,j} \exp [ \ln p^{prior}_{ij} + \ln L_{ij} ] \Big)
$$

---

### 5. Posterior statistics

Posterior mean estimates are obtained from the normalized posterior as:

$$
\hat{m}_v = \sum_{i,j} p_{ij}\, m_{v,i}, \qquad
\hat{c}_v = \sum_{i,j} p_{ij}\, c_{v,j}
$$

Marginal posteriors (summing over rows/columns) provide 95 % credible intervals through discrete quantiles.

---

## 🔁 Sequential Updating

The updating is performed iteratively for subsets of observations:

| Iteration | Observations Used | Description |
|------------|------------------|--------------|
| **k = 1** | first measurement (10 days) | Initial, broad posterior — large uncertainty |
| **k = 2** | first two (10 + 20 days) | Posterior contracts; early-time rate better constrained |
| **k = 3** | first three (10 + 20 + 40 days) | Progressive narrowing of credible intervals |
| **k = 4** | all observations | Final posterior sharply peaked — confident parameter estimates |

---

## 📊 Key Equations

**Bayesian Updating Equation**

$$
p(\theta|\mathbf{y})
\propto
\Bigg[\prod_i
\phi\!\left(\frac{y_i - s(t_i;\theta)}{\sigma_e}\right)
\Bigg]p(\theta),
\qquad
\theta = (m_v,c_v)
$$

**Posterior Mean Estimates**

$$
\hat{m}_v = \sum p_{ij} m_{v,i},
\qquad
\hat{c}_v = \sum p_{ij} c_{v,j}
$$

---

## 🧮 Numerical Features

- **Deterministic grid-posterior** (no random seeds, fully reproducible)  
- **Log-sum-exp** normalization for numerical stability  
- **Lognormal priors** ensuring positive soil parameters  
- **2-D visualizations** of prior, likelihood, unnormalized, and normalized posteriors  
- **3-D posterior surfaces** with **mean (×)** and **MAP (▲)** markers  

---

## 🖥️ How to Run

### Prerequisites

- Python ≥ 3.9  
- `numpy`, `pandas`, `matplotlib`

### Execution

```bash
python bayes_consolidation_sequential_viz.py
```

Figures will pop up sequentially and be saved to:

```
./outputs_bayes_consolidation/
```

---

## 📁 Output Summary (final order)

| # | File | Description |
|---|------|-------------|
| 1 | `equation_obs4_worked_example.png` | Final 4-observation worked example |
| 2 | `fig_joint_posterior_heatmap.png` | Posterior joint heatmap (all data) |
| 3 | `k1_post_surface.png` | 3-D posterior (k = 1) — broad, skewed surface |
| 4 | `k4_post_surface.png` | 3-D posterior (k = 4) — concentrated posterior |
| 5 | `fig_mv_prior_posterior.png` | Prior vs posterior marginal for `m_v` |
| 6 | `fig_cv_prior_posterior.png` | Prior vs posterior marginal for `c_v` |
| 7 | `equation_summary.png` | Bayesian updating formula summary |
| 8 | `param_summary.png` | Table summarizing prior and posterior statistics |

Intermediate files (`step1_first_obs.png`, etc.) illustrate how uncertainty is reduced with each dataset.

---

## 🔬 Interpretation Guide

- **At k = 1:** Posterior is wide and skewed; mean ≠ MAP due to non-symmetry.  
- **At k = 2–3:** Posterior becomes narrower as new information is assimilated.  
- **At k = 4:** Posterior peak is sharp; credible intervals minimal — parameters well identified.

### Mean vs MAP

- **MAP (mode):** grid point of maximum posterior probability.  
- **Mean:** expected value (center of mass) of the posterior surface.  
  Their difference quantifies asymmetry in the posterior distribution.

---

## ⚙️ Numerical Notes

- Posterior computed entirely in **log-space** (prevents underflow).  
- Grid-posterior inference provides a **benchmark** for verifying stochastic samplers.  
- Priors can be modified for case-specific or site-specific soil conditions.

---

## 📘 Build a Consolidated PDF Report

After running the main script, compile all figures into a single report:

```bash
python build_summary_pdf.py
```

The report will be generated as:

```
outputs_bayes_consolidation/Consolidation_Bayes_Summary.pdf
```

---

## 🧱 Repository Structure

```
.
├── bayes_consolidation_sequential_viz.py    # Main sequential Bayesian update script
├── build_summary_pdf.py                     # Combines figures into a single PDF
├── README.md                                # This file
└── outputs_bayes_consolidation/             # Folder containing generated figures
```

---

## 💡 Extensions

- Replace the first-term approximation with a multi-term or large-strain consolidation model.  
- Add pore pressure data for coupled hydraulic–mechanical updating.  
- Embed this grid solver in an MCMC sampler (e.g. DREAM-ZS) for high-dimensional inference.

---

## 📚 Reference

Jones, M. (2025).  
*Bayesian Updating for One-Dimensional Consolidation: Sequential parameter estimation using a grid-based posterior.*  
University of Newcastle / BECA Consulting.

---

## 👨‍💻 Author & License

**Author:** Merrick Jones (2025)  
**Affiliation:** University of Newcastle  
**License:** MIT
