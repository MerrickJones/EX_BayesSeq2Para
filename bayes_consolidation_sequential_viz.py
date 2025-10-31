# bayes_consolidation_sequential_viz.py  (updated)
# ===============================================================
# Sequential Bayesian updating for 1D consolidation settlement.
#
# Key changes to original version:
# 1) Heatmaps use pcolormesh on the true (log-spaced) grid (no imshow warping).
# 2) Posterior normalization uses stable log-sum-exp.
# 3) All posterior statistics & sampling are in physical space, but priors are
#    lognormal and grids are log-spaced (as in your correct version).
# 4) Consistent, high-quality figure styles and 400 dpi saving.
# 5) Minor layout polish (legends, labels, tables), without changing the look/flow.
# ===============================================================
from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D)

# ---------------- Matplotlib config (polished, LaTeX-free) ----------------
mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["figure.dpi"] = 110          # comfortable on-screen
mpl.rcParams["savefig.dpi"] = 400         # high-resolution output
mpl.rcParams["savefig.facecolor"] = "white"
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["font.size"] = 11

# ---------------- Run metadata ----------------
CURRENT_TIME = "updated run"

# ---------------- Configuration ----------------
SAVE_DIR = "outputs_bayes_consolidation"
GRID_SIZE = 161
SAMPLE_COUNT = 4000
RNG_SEED = 321

# Physical constants
LAYER_THICKNESS = 5.0          # H (m)
DRAINAGE_PATH = LAYER_THICKNESS / 2.0  # H/2 (m)
LOAD_INCREMENT = 22.0          # Δσ' (kPa)
MEASUREMENT_NOISE = 3.0        # σe (mm)

# Observations
OBSERVATION_TIMES = np.array([10.0, 20.0, 40.0, 80.0])      # days
OBSERVED_SETTLEMENTS = np.array([49.5, 61.1, 86.4, 116.7])  # mm

# Priors: independent lognormal for m_v and c_v (medians + log-σ)
M_V_MEDIAN = 1.0e-3
C_V_MEDIAN = 0.040
LOG_SD_M = 0.6
LOG_SD_C = 0.6
MU_LOG_M = math.log(M_V_MEDIAN)
MU_LOG_C = math.log(C_V_MEDIAN)

# Fine time grid for predictive curves (unchanged)
PREDICTION_TIMES = np.linspace(1.0, 120.0, 180)

# ---------------- Utilities ----------------
def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def logsumexp(log_values: np.ndarray) -> float:
    m = np.max(log_values)
    return float(m + np.log(np.sum(np.exp(log_values - m))))

def lognormal_pdf(x: float, mu_log: float, sigma_log: float) -> float:
    if x <= 0:
        return 0.0
    return (1.0 / (x * sigma_log * math.sqrt(2.0 * math.pi))) * \
           math.exp(-0.5 * ((math.log(x) - mu_log) / sigma_log) ** 2)

def discrete_quantile(grid: np.ndarray, pmf: np.ndarray, q: float) -> float:
    cdf = np.cumsum(pmf)
    idx = np.searchsorted(cdf, q, side="left")
    idx = min(max(idx, 0), len(grid) - 1)
    return float(grid[idx])

def pmf_to_density(grid: np.ndarray, pmf: np.ndarray) -> np.ndarray:
    """Convert PMF on a nonuniform (log) grid to approx. continuous density."""
    dx = np.gradient(grid)                # true cell widths on the log grid
    dens = pmf / dx
    area = np.trapz(dens, grid)
    if area > 0:
        dens /= area
    return dens

def show_and_save(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved figure: {os.path.abspath(path)}")
    plt.show()

def format_small(x: float) -> str:
    return f"{x:.3e}"

def format_std(x: float) -> str:
    return f"{x:.3f}"

# ---------------- Parameter grids (log-spaced) ----------------
RNG = np.random.default_rng(RNG_SEED)
M_V_GRID = np.exp(np.linspace(np.log(3e-4), np.log(3e-3), GRID_SIZE))  # rows (i)
C_V_GRID = np.exp(np.linspace(np.log(0.01), np.log(0.10), GRID_SIZE))  # cols (j)
M_GRID, C_GRID = np.meshgrid(M_V_GRID, C_V_GRID, indexing="ij")        # (N,N) for pcolormesh

print(f"Grid ranges: m_v [{M_V_GRID.min():.3e}, {M_V_GRID.max():.3e}] 1/kPa")
print(f"              c_v [{C_V_GRID.min():.3f}, {C_V_GRID.max():.3f}] m²/day")
print(f"Grid size: {GRID_SIZE} x {GRID_SIZE} = {GRID_SIZE**2} points")

# ---------------- Forward model (unchanged physics) ----------------
def u_first_term(Tv: np.ndarray | float) -> np.ndarray | float:
    """U(Tv) ≈ 1 - (8/π²) exp(-π² Tv / 4)."""
    return 1.0 - (8.0 / (math.pi ** 2)) * np.exp(-(math.pi ** 2) * np.asarray(Tv) / 4.0)

def settlement_mm(times: np.ndarray, m_v: float, c_v: float) -> np.ndarray:
    """S(t) = m_v Δσ' H × 1000 × U( c_v t / (H/2)² )."""
    Tv = c_v * times / (DRAINAGE_PATH ** 2)
    S_inf = m_v * LOAD_INCREMENT * LAYER_THICKNESS * 1000.0
    return S_inf * u_first_term(Tv)

# ---------------- Bayesian pieces (log-space correct) ----------------
def log_prior(m_v: float, c_v: float) -> float:
    pm = lognormal_pdf(m_v, MU_LOG_M, LOG_SD_M)
    pc = lognormal_pdf(c_v, MU_LOG_C, LOG_SD_C)
    if pm <= 0.0 or pc <= 0.0:
        return -1e300
    return math.log(pm) + math.log(pc)

def log_likelihood_subset(m_v: float, c_v: float,
                          t_sub: np.ndarray, y_sub: np.ndarray) -> float:
    pred = settlement_mm(t_sub, m_v, c_v)
    resid = y_sub - pred
    n = len(y_sub)
    return -0.5 * np.sum((resid / MEASUREMENT_NOISE) ** 2) \
           - n * math.log(math.sqrt(2.0 * math.pi) * MEASUREMENT_NOISE)

def posterior_on_grid(t_sub: np.ndarray, y_sub: np.ndarray) -> np.ndarray:
    """Return normalized posterior PMF on the (m_v, c_v) log grids (N×N)."""
    logp = np.empty((GRID_SIZE, GRID_SIZE))
    for i, mv in enumerate(M_V_GRID):
        for j, cv in enumerate(C_V_GRID):
            lp = log_prior(mv, cv) + log_likelihood_subset(mv, cv, t_sub, y_sub)
            logp[i, j] = lp if np.isfinite(lp) else -1e300
    logZ = logsumexp(logp.ravel())
    return np.exp(logp - logZ)

def sample_from_discrete_2d(posterior: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw samples (m_v, c_v) ~ posterior PMF on the physical grid."""
    p = posterior.ravel()
    p /= p.sum()
    idx = RNG.choice(p.size, size=n, p=p)
    ii, jj = np.unravel_index(idx, posterior.shape)
    return M_V_GRID[ii], C_V_GRID[jj]

def predictive_bands_from_samples(t_vec: np.ndarray,
                                  m_samps: np.ndarray, c_samps: np.ndarray,
                                  q_lo: float = 0.025, q_hi: float = 0.975
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = np.array([settlement_mm(t_vec, mv, cv) for mv, cv in zip(m_samps, c_samps)])
    med = np.median(preds, axis=0)
    lo = np.quantile(preds, q_lo, axis=0)
    hi = np.quantile(preds, q_hi, axis=0)
    return med, lo, hi

# ---------------- Visualization helpers (kept aesthetic) ----------------
def draw_equation(k: int, file_name: str, mv_mean: float, cv_mean: float, pred_all: np.ndarray) -> None:
    plt.figure(figsize=(12.5, 7.2))
    ax = plt.gca(); ax.axis("off")
    plt.title(f"Bayesian Update Equation ({k} Observation{'s' if k > 1 else ''})", fontsize=14, pad=12)

    main_eq = (r"$p(\theta \mid y_{1:%d}) \propto "
               r"\left[ \prod_{i=1}^{%d} \phi \left( \frac{y_i - s(t_i; \theta)}{\sigma_e} \right) \right] p(\theta)$" % (k, k))
    ax.text(0.5, 0.86, main_eq, fontsize=22, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", ec="black", fc="none", lw=1.6))

    y_vals = OBSERVED_SETTLEMENTS[:k]
    t_vals = OBSERVATION_TIMES[:k]
    y_pos = 0.70
    for i in range(k):
        phi_term = (r"$\phi \left( \frac{%0.1f \, \mathrm{mm} - s(%d; \theta)}{3.0 \, \mathrm{mm}} \right)$"
                    % (y_vals[i], int(t_vals[i])))
        ax.text(0.5, y_pos, phi_term, fontsize=20, ha="center", va="center")
        y_pos -= 0.08

    s_eq = (r"$s(t; \theta) = m_v \, \Delta \sigma' \, H \, U \left( \frac{c_v \, t}{(H/2)^2} \right), "
            r"U(T_v) \approx 1 - \frac{8}{\pi^2} \exp \left( -\frac{\pi^2}{4} T_v \right)$")
    ax.text(0.5, 0.38, s_eq, fontsize=18, ha="center", va="center")

    params_text = (r"$\Delta \sigma' = 22 \, \mathrm{kPa}, \, H = 5 \, \mathrm{m}, \, H/2 = 2.5 \, \mathrm{m}, \, "
                   r"\sigma_e = 3.0 \, \mathrm{mm}$")
    ax.text(0.5, 0.28, params_text, fontsize=16, ha="center", va="center")

    means_text = (rf"$\hat{{m_v}} = {format_small(mv_mean)} \, \mathrm{{1/kPa}}, \, "
                  rf"\hat{{c_v}} = {format_std(cv_mean)} \, \mathrm{{m^2/day}}$")
    ax.text(0.5, 0.18, means_text, fontsize=18, ha="center", va="center")

    t_labels = ", ".join([f"{int(t)}" for t in OBSERVATION_TIMES])
    s_labels = ", ".join([f"{s:.1f}" for s in pred_all])
    pred_text = rf"Updated $s(t)$ (mm) at $t = [{t_labels}]: [{s_labels}]$"
    ax.text(0.5, 0.08, pred_text, fontsize=14, ha="center", va="center")

    show_and_save(os.path.join(SAVE_DIR, file_name))

def worked_example_figure(k: int, posterior: np.ndarray, file_name: str) -> tuple[float, float, np.ndarray]:
    """Prior (scaled), Likelihood, Unnormalized Post, Normalized Post (+ markers), Text panel"""
    t_sub, y_sub = OBSERVATION_TIMES[:k], OBSERVED_SETTLEMENTS[:k]

    # Prior grid (scaled)
    log_prior_grid = np.empty_like(posterior)
    for i, mv in enumerate(M_V_GRID):
        for j, cv in enumerate(C_V_GRID):
            log_prior_grid[i, j] = log_prior(mv, cv)
    prior_scaled = np.exp(log_prior_grid - np.max(log_prior_grid))

    # Likelihood grid
    log_lik_grid = np.empty_like(posterior)
    for i, mv in enumerate(M_V_GRID):
        for j, cv in enumerate(C_V_GRID):
            log_lik_grid[i, j] = log_likelihood_subset(mv, cv, t_sub, y_sub)
    lik_pos = np.exp(log_lik_grid)

    # Unnormalized posterior (display only)
    unnorm_post = prior_scaled * lik_pos

    # Posterior stats (means and MAP) in physical space
    mv_mean = float(np.sum(posterior * M_V_GRID[:, None]))
    cv_mean = float(np.sum(posterior * C_V_GRID[None, :]))
    i_map, j_map = np.unravel_index(np.argmax(posterior), posterior.shape)
    mv_map, cv_map = M_V_GRID[i_map], C_V_GRID[j_map]
    pred_all = settlement_mm(OBSERVATION_TIMES, mv_mean, cv_mean)

    # --- Figure ---
    fig = plt.figure(figsize=(12.8, 8.4))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.12])

    # Prior (scaled) on true log grid
    ax1 = fig.add_subplot(gs[0, 0])
    pcm1 = ax1.pcolormesh(M_GRID, C_GRID, prior_scaled, shading="auto", cmap="viridis")
    ax1.set_title("Prior Density (Scaled)")
    ax1.set_xlabel("m_v (1/kPa)"); ax1.set_ylabel("c_v (m²/day)")
    fig.colorbar(pcm1, ax=ax1, shrink=0.8)

    # Likelihood (k obs)
    ax2 = fig.add_subplot(gs[0, 1])
    pcm2 = ax2.pcolormesh(M_GRID, C_GRID, lik_pos, shading="auto", cmap="viridis")
    ax2.set_title(f"Likelihood (k={k})")
    ax2.set_xlabel("m_v (1/kPa)"); ax2.set_ylabel("c_v (m²/day)")
    fig.colorbar(pcm2, ax=ax2, shrink=0.8)

    # Unnormalized posterior
    ax3 = fig.add_subplot(gs[1, 0])
    pcm3 = ax3.pcolormesh(M_GRID, C_GRID, unnorm_post, shading="auto", cmap="viridis")
    ax3.set_title("Unnormalized Posterior ∝ Prior × Likelihood")
    ax3.set_xlabel("m_v (1/kPa)"); ax3.set_ylabel("c_v (m²/day)")
    fig.colorbar(pcm3, ax=ax3, shrink=0.8)

    # Normalized posterior + contours + markers
    ax4 = fig.add_subplot(gs[1, 1])
    pcm4 = ax4.pcolormesh(M_GRID, C_GRID, posterior, shading="auto", cmap="viridis")
    cs = ax4.contour(M_V_GRID, C_V_GRID, posterior.T, levels=12, linewidths=0.8, colors="k")
    ax4.clabel(cs, inline=True, fontsize=7)
    ax4.scatter([mv_map], [cv_map], marker="^", s=70, label="MAP (Mode)")
    ax4.scatter([mv_mean], [cv_mean], marker="x", s=70, label="Mean (×)")
    ax4.set_title(f"Normalized Posterior (k={k})")
    ax4.set_xlabel("m_v (1/kPa)"); ax4.set_ylabel("c_v (m²/day)")
    ax4.legend(fontsize=8, loc="upper left")
    fig.colorbar(pcm4, ax=ax4, shrink=0.8)

    # Text Panel
    ax5 = fig.add_subplot(gs[:, 2]); ax5.axis("off")
    lines = [
        rf"$\hat m_v \approx {format_small(mv_mean)} \, \mathrm{{1/kPa}},\quad "
        rf"\hat c_v \approx {format_std(cv_mean)} \, \mathrm{{m^2/day}}$",
        r"Updated $s(t)$ (mm) at $t=\{10,20,40,80\}$:",
        rf"$[{pred_all[0]:.1f}, {pred_all[1]:.1f}, {pred_all[2]:.1f}, {pred_all[3]:.1f}]$",
    ]
    ax5.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=11)

    fig.suptitle(f"Worked Example: Bayesian Update with k={k}", fontsize=14, y=0.98)
    show_and_save(os.path.join(SAVE_DIR, file_name))
    return mv_mean, cv_mean, pred_all

def draw_posterior_3d(k: int, t_sub: np.ndarray, y_sub: np.ndarray, file_name: str) -> None:
    posterior = posterior_on_grid(t_sub, y_sub)
    mv_mean = float(np.sum(posterior * M_V_GRID[:, None]))
    cv_mean = float(np.sum(posterior * C_V_GRID[None, :]))
    i_map, j_map = np.unravel_index(np.argmax(posterior), posterior.shape)
    mv_map, cv_map = M_V_GRID[i_map], C_V_GRID[j_map]

    M, C = np.meshgrid(M_V_GRID, C_V_GRID, indexing="ij")
    Z = posterior
    vmin, vmax = np.min(Z), np.max(Z)

    fig = plt.figure(figsize=(12.5, 7.5))
    ax3d = fig.add_subplot(121, projection="3d")
    surf = ax3d.plot_surface(M, C, Z, rstride=3, cstride=3, linewidth=0.0,
                             antialiased=True, alpha=0.95, cmap="viridis", vmin=vmin, vmax=vmax)
    ax3d.contour(M, C, Z, zdir='z', offset=0, cmap="viridis", levels=8)
    ax3d.set_title(f"k={k} Normalized Posterior — 3D Surface")
    ax3d.set_xlabel("m_v (1/kPa)"); ax3d.set_ylabel("c_v (m²/day)"); ax3d.set_zlabel("Posterior")
    fig.colorbar(surf, ax=ax3d, shrink=0.65, pad=0.08)
    ax3d.scatter(mv_map, cv_map, Z[i_map, j_map], marker="^", s=60, label="MAP (▲)")
    ax3d.scatter(mv_mean, cv_mean, Z[np.argmin(abs(M_V_GRID - mv_mean)),
                                     np.argmin(abs(C_V_GRID - cv_mean))],
                 marker="x", s=60, label="Mean (×)")
    ax3d.legend(loc="upper left", fontsize=8)

    ax2d = fig.add_subplot(122)
    pcm = ax2d.pcolormesh(M, C, Z, shading="auto", cmap="viridis")
    cs = ax2d.contour(M_V_GRID, C_V_GRID, Z.T, levels=8, linewidths=0.8, colors="k")
    ax2d.clabel(cs, inline=True, fontsize=7)
    ax2d.scatter([mv_map], [cv_map], marker="^", s=70, label="MAP")
    ax2d.scatter([mv_mean], [cv_mean], marker="x", s=70, label="Mean")
    ax2d.set_title(f"Top-down with Colormap + Contours (k={k})")
    ax2d.set_xlabel("m_v (1/kPa)"); ax2d.set_ylabel("c_v (m²/day)")
    fig.colorbar(pcm, ax=ax2d, shrink=0.75, pad=0.06)
    ax2d.legend(loc="upper left", fontsize=8)

    show_and_save(os.path.join(SAVE_DIR, file_name))

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("=" * 60)
    print("Starting Sequential Bayesian Consolidation Analysis")
    print(f"Date and Time: {CURRENT_TIME}")
    print("=" * 60)

    ensure_directory(SAVE_DIR)
    print(f"Output directory: {os.path.abspath(SAVE_DIR)}")

    # Prior predictive curve (use medians of lognormals)
    print("\nGenerating prior predictive curve...")
    PRIOR_CURVE = settlement_mm(PREDICTION_TIMES, M_V_MEDIAN, C_V_MEDIAN)

    plt.figure(figsize=(8, 6))
    plt.plot(PREDICTION_TIMES, PRIOR_CURVE, label="Prior Prediction", color='blue', linestyle='--', lw=2)
    plt.xlabel("Time (days)"); plt.ylabel("Settlement (mm)")
    plt.title("Prior Predictive Settlement")
    plt.legend(); plt.grid(True, alpha=0.3)
    show_and_save(os.path.join(SAVE_DIR, "step0_prior_only.png"))

    # Sequential scenarios (unchanged naming)
    SCENARIOS = [
        {"name": "step1_first_obs",    "k": 1},
        {"name": "step2_first_two",    "k": 2},
        {"name": "step3_first_three",  "k": 3},
        {"name": "step4_all_four",     "k": 4},
    ]

    # Parameter summary table
    PARAM_SUMMARY = []

    # Prior 95% bands from lognormal
    prior_mv_lo = math.exp(MU_LOG_M - 1.96 * LOG_SD_M)
    prior_mv_hi = math.exp(MU_LOG_M + 1.96 * LOG_SD_M)
    prior_cv_lo = math.exp(MU_LOG_C - 1.96 * LOG_SD_C)
    prior_cv_hi = math.exp(MU_LOG_C + 1.96 * LOG_SD_C)
    PARAM_SUMMARY.append({
        "Stage": "Prior",
        "m_v_mean": M_V_MEDIAN,
        "m_v_95_lo": prior_mv_lo,
        "m_v_95_hi": prior_mv_hi,
        "c_v_mean": C_V_MEDIAN,
        "c_v_95_lo": prior_cv_lo,
        "c_v_95_hi": prior_cv_hi
    })

    TABLES = []
    print("\nPerforming sequential Bayesian updates...")
    for scenario in SCENARIOS:
        name, k = scenario["name"], scenario["k"]
        print(f"\n--- Updating with first {k} observation{'s' if k > 1 else ''} ---")
        t_sub = OBSERVATION_TIMES[:k]
        y_sub = OBSERVED_SETTLEMENTS[:k]

        POSTERIOR = posterior_on_grid(t_sub, y_sub)

        # Draw posterior-predictive band from posterior samples (physical space)
        M_SAMPLES, C_SAMPLES = sample_from_discrete_2d(POSTERIOR, SAMPLE_COUNT)
        POST_MED, POST_LO, POST_HI = predictive_bands_from_samples(PREDICTION_TIMES, M_SAMPLES, C_SAMPLES)

        # Predictive plot (kept aesthetic, corrected stats)
        plt.figure(figsize=(8, 6))
        plt.plot(PREDICTION_TIMES, PRIOR_CURVE, label="Prior Prediction", color='blue', linestyle='--', lw=2)
        plt.scatter(t_sub, y_sub, color='red', label=f"Observed (n={k})", zorder=5)
        plt.fill_between(PREDICTION_TIMES, POST_LO, POST_HI, alpha=0.25, color='green', label="95% Credible Band")
        plt.plot(PREDICTION_TIMES, POST_MED, label="Posterior Prediction", color='green', lw=2)
        plt.xlabel("Time (days)"); plt.ylabel("Settlement (mm)")
        plt.title(f"Update with First {k} Observation{'s' if k > 1 else ''}")
        plt.legend(); plt.grid(True, alpha=0.3)
        show_and_save(os.path.join(SAVE_DIR, f"{name}.png"))

        # Posterior statistics & marginals (physical space)
        MV_MEAN = float(np.sum(POSTERIOR * M_V_GRID[:, None]))
        CV_MEAN = float(np.sum(POSTERIOR * C_V_GRID[None, :]))
        MARG_M = np.sum(POSTERIOR, axis=1)
        MARG_C = np.sum(POSTERIOR, axis=0)
        MV_LO = discrete_quantile(M_V_GRID, MARG_M, 0.025)
        MV_HI = discrete_quantile(M_V_GRID, MARG_M, 0.975)
        CV_LO = discrete_quantile(C_V_GRID, MARG_C, 0.025)
        CV_HI = discrete_quantile(C_V_GRID, MARG_C, 0.975)

        PARAM_SUMMARY.append({
            "Stage": f"Posterior ({k} obs)",
            "m_v_mean": MV_MEAN, "m_v_95_lo": MV_LO, "m_v_95_hi": MV_HI,
            "c_v_mean": CV_MEAN, "c_v_95_lo": CV_LO, "c_v_95_hi": CV_HI
        })

        print(f"  Posterior mean: m_v = {format_small(MV_MEAN)} 1/kPa, c_v = {format_std(CV_MEAN)} m²/day")
        print(f"  95% Credible Interval: m_v [{format_small(MV_LO)}, {format_small(MV_HI)}], "
              f"c_v [{format_std(CV_LO)}, {format_std(CV_HI)}]")

        # Prediction table for report
        prior_preds = np.round(settlement_mm(t_sub, M_V_MEDIAN, C_V_MEDIAN), 1)
        post_preds  = np.round(settlement_mm(t_sub, MV_MEAN, CV_MEAN), 1)
        df_step = pd.DataFrame({
            "Scenario": name,
            "Time (days)": t_sub,
            "Observed (mm)": y_sub,
            "Prior Prediction (mm)": prior_preds,
            "Posterior Prediction (mm)": post_preds
        })
        TABLES.append(df_step)

        # Equations + worked example (now using pcolormesh grids)
        PRED_ALL = settlement_mm(OBSERVATION_TIMES, MV_MEAN, CV_MEAN)
        draw_equation(k, f"equation_obs{k}.png", MV_MEAN, CV_MEAN, PRED_ALL)
        worked_example_figure(k, POSTERIOR, f"equation_obs{k}_worked_example.png")

    # Save combined prediction table
    df_predictions = pd.concat(TABLES, ignore_index=True)
    csv_path = os.path.join(SAVE_DIR, "sequential_predictions.csv")
    df_predictions.to_csv(csv_path, index=False)
    print(f"\nSaved prediction table: {csv_path}")

    # ---- Additional figures (unchanged sequence, corrected internals) ----
    print("\nDisplaying remaining figures in requested order...")

    # 1) Joint posterior (all data) — pcolormesh + contours
    POST_ALL = posterior_on_grid(OBSERVATION_TIMES, OBSERVED_SETTLEMENTS)
    plt.figure()
    pcm = plt.pcolormesh(M_GRID, C_GRID, POST_ALL, shading="auto", cmap="viridis")
    cs = plt.contour(M_V_GRID, C_V_GRID, POST_ALL.T, levels=8, linewidths=0.8, colors="k")
    plt.clabel(cs, inline=True, fontsize=7)
    plt.colorbar(pcm, shrink=0.8, pad=0.06, label="Posterior Probability (Discrete)")
    plt.xlabel("m_v (1/kPa)"); plt.ylabel("c_v (m²/day)")
    plt.title("Posterior Joint (m_v, c_v | All Data)")
    show_and_save(os.path.join(SAVE_DIR, "fig_joint_posterior_heatmap.png"))

    # 2) k=1 3D View + Explanation
    draw_posterior_3d(1, OBSERVATION_TIMES[:1], OBSERVED_SETTLEMENTS[:1], "k1_post_surface.png")

    # 3) k=4 3D View + Explanation
    draw_posterior_3d(4, OBSERVATION_TIMES, OBSERVED_SETTLEMENTS, "k4_post_surface.png")

    # 4) m_v: Prior vs Posterior (all data)
    MARG_M_ALL = np.sum(POST_ALL, axis=1)
    prior_mv_pdf = np.array([lognormal_pdf(x, MU_LOG_M, LOG_SD_M) for x in M_V_GRID])
    prior_mv_pdf /= np.trapz(prior_mv_pdf, M_V_GRID)
    post_mv_dens = pmf_to_density(M_V_GRID, MARG_M_ALL)
    plt.figure()
    plt.plot(M_V_GRID, prior_mv_pdf, label="Prior Density", lw=2)
    plt.plot(M_V_GRID, post_mv_dens, label="Posterior Density (All Data)", lw=2)
    mv_lo_all = discrete_quantile(M_V_GRID, MARG_M_ALL, 0.025)
    mv_hi_all = discrete_quantile(M_V_GRID, MARG_M_ALL, 0.975)
    plt.axvline(mv_lo_all, linestyle=":", label="Post 95% Lower")
    plt.axvline(mv_hi_all, linestyle=":", label="Post 95% Upper")
    plt.xlabel("m_v (1/kPa)"); plt.ylabel("Density")
    plt.title("m_v: Prior vs Posterior (All Data)")
    plt.legend(fontsize=8, loc="upper right")
    show_and_save(os.path.join(SAVE_DIR, "fig_mv_prior_posterior.png"))

    # 5) c_v: Prior vs Posterior (all data)
    MARG_C_ALL = np.sum(POST_ALL, axis=0)
    prior_cv_pdf = np.array([lognormal_pdf(x, MU_LOG_C, LOG_SD_C) for x in C_V_GRID])
    prior_cv_pdf /= np.trapz(prior_cv_pdf, C_V_GRID)
    post_cv_dens = pmf_to_density(C_V_GRID, MARG_C_ALL)
    plt.figure()
    plt.plot(C_V_GRID, prior_cv_pdf, label="Prior Density", lw=2)
    plt.plot(C_V_GRID, post_cv_dens, label="Posterior Density (All Data)", lw=2)
    cv_lo_all = discrete_quantile(C_V_GRID, MARG_C_ALL, 0.025)
    cv_hi_all = discrete_quantile(C_V_GRID, MARG_C_ALL, 0.975)
    plt.axvline(cv_lo_all, linestyle=":", label="Post 95% Lower")
    plt.axvline(cv_hi_all, linestyle=":", label="Post 95% Upper")
    plt.xlabel("c_v (m²/day)"); plt.ylabel("Density")
    plt.title("c_v: Prior vs Posterior (All Data)")
    plt.legend(fontsize=8, loc="upper right")
    show_and_save(os.path.join(SAVE_DIR, "fig_cv_prior_posterior.png"))

    # 6) Summary equation (kept aesthetic)
    plt.figure(figsize=(11, 5))
    ax = plt.gca(); ax.axis("off")
    main_eq = (r"$p(\theta \mid \mathbf{y}) \propto "
               r"\left[ \prod_i \phi \left( \frac{y_i - s(t_i; \theta)}{\sigma_e} \right) \right] p(\theta), "
               r"\theta = (m_v, c_v)$")
    ax.text(0.5, 0.65, main_eq, fontsize=24, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", ec="black", fc="none", lw=1.6))
    s_eq = (r"$s(t_i; \theta) = m_v \, \Delta \sigma' \, H \, U \left( \frac{c_v \, t_i}{(H/2)^2} \right), "
            r"U(T_v) \approx 1 - \frac{8}{\pi^2} \exp \left( -\frac{\pi^2}{4} T_v \right)$")
    ax.text(0.5, 0.30, s_eq, fontsize=21, ha="center", va="center")
    ax.set_title("Bayesian Updating Equation (Summary)", fontsize=14, pad=12)
    show_and_save(os.path.join(SAVE_DIR, "equation_summary.png"))

    # 7) Parameter summary table
    summary_df = pd.DataFrame(PARAM_SUMMARY)
    plt.figure(figsize=(11.5, 6.0))
    plt.axis("off")
    plt.title("Parameter Summary (Prior and Sequential Posteriors)", fontsize=14, pad=12)

    df_display = summary_df.copy()
    for col in df_display.columns:
        if col == "Stage":
            continue
        df_display[col] = df_display[col].apply(lambda x: format_small(x) if col.startswith("m_v") else format_std(x))

    table = plt.table(cellText=df_display.values,
                      colLabels=df_display.columns,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.05, 1.25)
    show_and_save(os.path.join(SAVE_DIR, "param_summary.png"))

    # Save predictions CSV again at the end (harmless, explicit)
    df_predictions = pd.concat(TABLES, ignore_index=True)
    csv_path = os.path.join(SAVE_DIR, "sequential_predictions.csv")
    df_predictions.to_csv(csv_path, index=False)
    print(f"\nSaved prediction table: {csv_path}")

    print(f"\nAnalysis completed. Check {os.path.abspath(SAVE_DIR)} for results!")
