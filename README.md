# Bayesian Updating for 1-D Consolidation (Grid Posterior, Lognormal Priors)

This repository demonstrates a **transparent** and **reproducible** Bayesian update of 1-D consolidation parameters using a **grid posterior** in \((m_v, c_v)\) space. It produces sequential figures (prior → \(k=1\) → \(k=2\) → … → all data), 3-D posterior surfaces, and a parameter summary.

## Features

- Forward model: Terzaghi 1-D settlement with first-term approximation
  \[
  s(t; m_v, c_v) = 110{,}000\, m_v \cdot U\!\Big(\frac{c_v t}{(H/2)^2}\Big),\quad
  U(T_v)\approx 1-\frac{8}{\pi^2}\exp\!\Big(-\frac{\pi^2}{4}T_v\Big).
  \]
- Priors: independent lognormal on \(m_v\) and \(c_v\).
- Likelihood: Gaussian; measurement noise \(\sigma_e = 3\) mm.
- Posterior: computed on a **161×161** log-grid with log-sum-exp normalization.
- Outputs: prior/posterior plots, per-step worked examples, 3-D posteriors, and a summary.

## Requirements

- Python 3.9+  
- NumPy, Pandas, Matplotlib (no LaTeX or extra packages required)

## How to run

1. Place `bayes_consolidation_sequential_viz.py` in your working folder.  
2. Run:
   ```bash
   python bayes_consolidation_sequential_viz.py
   ```
3. Figures will **pop up** sequentially and also be saved to `./outputs_bayes_consolidation/`.

## Key outputs (file order at the end)

- `equation_obs4_worked_example.png`
- `fig_joint_posterior_heatmap.png`
- `k1_post_surface.png`
- `k4_post_surface.png`
- `fig_mv_prior_posterior.png`
- `fig_cv_prior_posterior.png`
- `equation_summary.png`
- `param_summary.png`

## Interpreting results

- **Worked example pages (k=1..4)** show how priors, likelihoods, and posteriors combine on the grid.  
- **3-D posterior** (k=1, k=4) illustrates peak (MAP) vs center of mass (mean).  
- **Marginal plots** (all data) display posterior density with **95% credible intervals**.

## Troubleshooting

- If you ever see glyph warnings, they are harmless; the code uses mathtext (no LaTeX).
- If figures appear behind other windows, your OS window manager is controlling z-order; re-run or alt-tab.

## Build a single PDF report

After running the main script, build a **single PDF** (with figures in order) by running:
```bash
python build_summary_pdf.py
```
This will create `outputs_bayes_consolidation/Consolidation_Bayes_Summary.pdf`.
