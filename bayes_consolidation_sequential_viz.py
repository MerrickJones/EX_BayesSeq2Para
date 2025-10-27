# bayes_consolidation_sequential_viz.py
# ---------------------------------------------------------------
# Sequential Bayesian updating for 1D consolidation (m_v, c_v).
# Pop-ups are standard size; saved PNGs are high-resolution (400 dpi).
# Heatmaps use pcolormesh on the true (log-spaced) grid so contours align.
# ---------------------------------------------------------------

from __future__ import annotations
import os
import math
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------- Matplotlib defaults ----------
mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["figure.dpi"] = 110          # comfortable on-screen size
mpl.rcParams["savefig.dpi"] = 400         # high-resolution saves
mpl.rcParams["savefig.facecolor"] = "white"
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["font.size"] = 11

# ---------- Config ----------
SAVE_DIR = "outputs_bayes_consolidation"
N_GRID = 161
RNG = np.random.default_rng(321)

# ---------- Utilities ----------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def logsumexp(v: np.ndarray) -> float:
    m = np.max(v)
    return float(m + np.log(np.sum(np.exp(v - m))))

def lognormal_pdf(x: float, mu_log: float, sigma_log: float) -> float:
    """Lognormal PDF with log-mean mu_log and log-std sigma_log."""
    if x <= 0:
        return 0.0
    return (1.0 / (x * sigma_log * math.sqrt(2.0 * math.pi))) * math.exp(
        -((math.log(x) - mu_log) ** 2) / (2.0 * sigma_log ** 2)
    )

def discrete_quantile(grid: np.ndarray, pmf: np.ndarray, q: float) -> float:
    cdf = np.cumsum(pmf)
    idx = np.searchsorted(cdf, q, side="left")
    idx = max(0, min(idx, len(grid) - 1))
    return float(grid[idx])

def pmf_to_density(grid: np.ndarray, pmf: np.ndarray) -> np.ndarray:
    """Convert discrete pmf over a nonuniform grid to a piecewise-constant density."""
    dx = np.gradient(grid)
    dens = pmf / dx
    area = np.trapezoid(dens, grid)
    if area > 0:
        dens = dens / area
    return dens

def show_and_save(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=400, bbox_inches="tight")
    plt.show()

def fmt_small(x: float) -> str:
    return f"{x:.3e}"

def fmt_std(x: float) -> str:
    return f"{x:.3f}"

# ---------- Problem setup ----------
H = 5.0                 # m
Hd = H / 2.0            # 2.5 m
delta_sigma = 22.0      # kPa
sigma_e = 3.0           # mm

t_all = np.array([10.0, 20.0, 40.0, 80.0])
y_all = np.array([49.5, 61.1, 86.4, 116.7])  # mm

# Priors (lognormal via medians + log-std)
m_v_median = 1.0e-3
c_v_median = 0.040
sigma_log_m = 0.6
sigma_log_c = 0.6
mu_log_m = math.log(m_v_median)
mu_log_c = math.log(c_v_median)

# Parameter grids (N × N, log-spaced) and mesh for pcolormesh
N = N_GRID
m_v_grid = np.exp(np.linspace(np.log(3e-4), np.log(3e-3), N))   # rows (i)
c_v_grid = np.exp(np.linspace(np.log(0.01), np.log(0.10), N))   # cols (j)
M_GRID, C_GRID = np.meshgrid(m_v_grid, c_v_grid, indexing="ij") # shape (N,N)

# ---------- Forward model ----------
def U_first_term(Tv):
    return 1.0 - (8.0 / (math.pi ** 2)) * np.exp(-(math.pi ** 2) * np.asarray(Tv) / 4.0)

def settlement_mm(t: np.ndarray, m_v: float, c_v: float) -> np.ndarray:
    Tv = c_v * t / (Hd ** 2)     # Hd^2 = 6.25
    S_inf_mm = m_v * 110000.0    # 22 kPa × 5 m × 1000 mm/m
    return S_inf_mm * U_first_term(Tv)

# ---------- Bayesian pieces ----------
def log_prior(m_v: float, c_v: float) -> float:
    pm = lognormal_pdf(m_v, mu_log_m, sigma_log_m)
    pc = lognormal_pdf(c_v, mu_log_c, sigma_log_c)
    if pm <= 0 or pc <= 0:
        return -1e300
    return math.log(pm) + math.log(pc)

def log_likelihood_subset(m_v: float, c_v: float, t_sub: np.ndarray, y_sub: np.ndarray) -> float:
    """Gaussian log-likelihood for subset with σ = sigma_e."""
    pred = settlement_mm(t_sub, m_v, c_v)
    resid = y_sub - pred
    return -0.5 * np.sum((resid / sigma_e) ** 2) - len(y_sub) * math.log(math.sqrt(2.0 * math.pi) * sigma_e)

def posterior_on_grid(t_sub: np.ndarray, y_sub: np.ndarray) -> np.ndarray:
    logp = np.empty((N, N))
    for i, mv in enumerate(m_v_grid):
        for j, cv in enumerate(c_v_grid):
            lp = log_prior(mv, cv) + log_likelihood_subset(mv, cv, t_sub, y_sub)
            logp[i, j] = lp if np.isfinite(lp) else -1e300
    Zlog = logsumexp(logp.ravel())
    return np.exp(logp - Zlog)

# ---------- Executive Summary ----------
def draw_executive_summary(mv_mean: float, mv_lo: float, mv_hi: float,
                           cv_mean: float, cv_lo: float, cv_hi: float):
    """Concise first-page summary for reviewers (first popup)."""
    fig = plt.figure(figsize=(9.6, 6.0))
    ax = plt.gca(); ax.axis("off")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "Executive Summary — Bayesian Consolidation Analysis",
        "",
        "• Objective: Estimate mᵥ (1/kPa) and cᵥ (m²/day) for a 5 m clay layer under Δσ' = 22 kPa using "
        "settlement at t = [10, 20, 40, 80] days.",
        "• Method: Bayesian updating on a 161×161 (mᵥ, cᵥ) grid with lognormal priors "
        f"(medians mᵥ={fmt_small(m_v_median)}, cᵥ={fmt_std(c_v_median)}; log-σ=0.6), "
        "Gaussian measurement noise (σ=3 mm), and Terzaghi 1D consolidation (double drainage).",
        "• Outputs: Prior/predictive plots, per-step worked examples, joint posterior heatmaps (pcolormesh + contours), "
        "3-D normalized posteriors (k=1, k=4), marginal posteriors with 95% CrI, and a parameter summary table.",
        "",
        "Final (k=4) Posterior:",
        f"  – mᵥ mean = {fmt_small(mv_mean)}  (95% CrI: {fmt_small(mv_lo)} – {fmt_small(mv_hi)})",
        f"  – cᵥ mean = {fmt_std(cv_mean)}  (95% CrI: {fmt_std(cv_lo)} – {fmt_std(cv_hi)})",
        "",
        f"Run completed: {now_str} — Figures saved under “{SAVE_DIR}/”."
    ]
    ax.text(0.03, 0.98, "\n".join(lines), ha="left", va="top", fontsize=11, wrap=True)
    show_and_save(os.path.join(SAVE_DIR, "executive_summary.png"))

# ---------- Numbers (equation) page per k ----------
def draw_numbered_equation(k: int, mv_mean: float, cv_mean: float,
                           pred_all_mm: np.ndarray, out_path: str):
    fig = plt.figure(figsize=(9.0, 6.0))
    ax = plt.gca(); ax.axis("off")
    ax.set_title(f"Bayesian Updating Equation ({k} observation(s))", fontsize=14, pad=10)

    eq_main = (r"$p(\theta\mid y_{1:{%d}})\ \propto\ "
               r"\left[\prod_{i=1}^{%d}\ \phi\!\left(\frac{y_i - s(t_i;\theta)}{\sigma_e}\right)\right]\ p(\theta)$" % (k, k))
    ax.text(0.5, 0.86, eq_main, fontsize=20, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", ec="black", fc="none", lw=1.4))

    y_vals = [49.5, 61.1, 86.4, 116.7][:k]
    t_vals = [10, 20, 40, 80][:k]
    y_base = 0.70
    for i in range(k):
        ax.text(0.5, y_base - 0.07 * i,
                r"$\phi\!\left(\dfrac{%0.1f\ \mathrm{mm} - s(%d;\,\theta)}{3.0\ \mathrm{mm}}\right)$" % (y_vals[i], t_vals[i]),
                fontsize=18, ha="center", va="center")

    ax.text(0.5, 0.35,
            (r"$s(t;\theta)=m_v\,\Delta\sigma'\,H\,U\!\left(\frac{c_v\,t}{(H/2)^2}\right),\quad"
             r"U(T_v)\approx 1-\dfrac{8}{\pi^2}\exp\!\left(-\dfrac{\pi^2}{4}\,T_v\right)$"),
            fontsize=16, ha="center", va="center")
    ax.text(0.5, 0.27,
            (r"$\Delta\sigma'=22\ \mathrm{kPa},\ H=5\ \mathrm{m},\ H/2=2.5\ \mathrm{m},\ "
             r"\sigma_e=3.0\ \mathrm{mm}.$"),
            fontsize=14, ha="center", va="center")

    ax.text(0.5, 0.18,
            rf"$\hat m_v={fmt_small(mv_mean)}\ \mathrm{{1/kPa}},\quad "
            rf"\hat c_v={fmt_std(cv_mean)}\ \mathrm{{m^2/day}}$",
            fontsize=16, ha="center", va="center")

    t_labels = ", ".join([f"{int(ti)}" for ti in [10, 20, 40, 80]])
    s_labels = ", ".join([f"{s:.1f}" for s in pred_all_mm])
    ax.text(0.5, 0.08,
            rf"Updated settlements (mm) at t = [{t_labels}]:  [{s_labels}]",
            fontsize=12, ha="center", va="center")

    show_and_save(out_path)

# ---------- Worked example (4-panel) per k ----------
def worked_example_figure(k: int, post_grid: np.ndarray, out_path: str) -> tuple[float, float, np.ndarray]:
    t_sub, y_sub = t_all[:k], y_all[:k]

    # Prior grid (scaled)
    logprior = np.empty((N, N))
    for i, mv in enumerate(m_v_grid):
        for j, cv in enumerate(c_v_grid):
            lp = log_prior(mv, cv)
            logprior[i, j] = lp if np.isfinite(lp) else -1e300
    prior_scaled = np.exp(logprior - np.max(logprior))

    # Likelihood grid (k obs)
    loglik = np.empty((N, N))
    for i, mv in enumerate(m_v_grid):
        for j, cv in enumerate(c_v_grid):
            loglik[i, j] = log_likelihood_subset(mv, cv, t_sub, y_sub)
    lik_pos = np.exp(loglik)

    # Unnormalized posterior (display only, positive)
    unnorm_pos = prior_scaled * lik_pos

    # Posterior means & MAP
    mv_mean_k = float(np.sum(post_grid * m_v_grid[:, None]))
    cv_mean_k = float(np.sum(post_grid * c_v_grid[None, :]))
    i_map, j_map = np.unravel_index(np.argmax(post_grid), post_grid.shape)
    mv_map_k, cv_map_k = float(m_v_grid[i_map]), float(c_v_grid[j_map])
    pred_all_k = settlement_mm(t_all, mv_mean_k, cv_mean_k)

    fig = plt.figure(figsize=(9.6, 6.2))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.12])

    # Prior (scaled)
    ax1 = fig.add_subplot(gs[0, 0])
    pcm1 = ax1.pcolormesh(M_GRID, C_GRID, prior_scaled, shading="auto", cmap="viridis")
    ax1.set_title("Prior density (scaled)")
    ax1.set_xlabel("m_v (1/kPa)"); ax1.set_ylabel("c_v (m^2/day)")
    fig.colorbar(pcm1, ax=ax1, shrink=0.8)

    # Likelihood
    ax2 = fig.add_subplot(gs[0, 1])
    pcm2 = ax2.pcolormesh(M_GRID, C_GRID, lik_pos, shading="auto", cmap="viridis")
    ax2.set_title(f"Likelihood (k={k})")
    ax2.set_xlabel("m_v (1/kPa)"); ax2.set_ylabel("c_v (m^2/day)")
    fig.colorbar(pcm2, ax=ax2, shrink=0.8)

    # Unnormalized posterior
    ax3 = fig.add_subplot(gs[1, 0])
    pcm3 = ax3.pcolormesh(M_GRID, C_GRID, unnorm_pos, shading="auto", cmap="viridis")
    ax3.set_title("Unnormalized posterior = prior × likelihood")
    ax3.set_xlabel("m_v (1/kPa)"); ax3.set_ylabel("c_v (m^2/day)")
    fig.colorbar(pcm3, ax=ax3, shrink=0.8)

    # Normalized posterior + markers
    ax4 = fig.add_subplot(gs[1, 1])
    pcm4 = ax4.pcolormesh(M_GRID, C_GRID, post_grid, shading="auto", cmap="viridis")
    cs = ax4.contour(m_v_grid, c_v_grid, post_grid.T, levels=6, linewidths=0.8, colors="k")
    ax4.clabel(cs, inline=True, fontsize=7)
    ax4.scatter([mv_map_k], [cv_map_k], marker="^", s=70, label="MAP (mode)")
    ax4.scatter([mv_mean_k], [cv_mean_k], marker="x", s=70, label="Mean (×)")
    ax4.set_title(f"Normalized posterior (k={k})")
    ax4.set_xlabel("m_v (1/kPa)"); ax4.set_ylabel("c_v (m^2/day)")
    ax4.legend(fontsize=8, loc="upper left")
    fig.colorbar(pcm4, ax=ax4, shrink=0.8)

    # Text panel
    ax5 = fig.add_subplot(gs[:, 2]); ax5.axis("off")
    lines = [
        fr"$\hat m_v\approx {fmt_small(mv_mean_k)}\ \mathrm{{1/kPa}},\quad "
        fr"\hat c_v\approx {fmt_std(cv_mean_k)}\ \mathrm{{m^2/day}}$",
        r"$\text{Updated } s(t)\ \text{(mm) at } t=\{10,20,40,80\}:$",
        fr"$[{pred_all_k[0]:.1f},\ {pred_all_k[1]:.1f},\ {pred_all_k[2]:.1f},\ {pred_all_k[3]:.1f}]$"
    ]
    ax5.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=11)

    fig.suptitle(f"Worked Example: Bayesian Updating with k={k}", fontsize=14, y=0.98)
    show_and_save(out_path)
    return mv_mean_k, cv_mean_k, pred_all_k

# ---------------- Run sequence ----------------
ensure_dir(SAVE_DIR)

# Compute full posterior first (k=4) for executive summary
post_all = posterior_on_grid(t_all, y_all)
marg_m_all = np.sum(post_all, axis=1)
marg_c_all = np.sum(post_all, axis=0)
mv_mean_all = float(np.sum(post_all * m_v_grid[:, None]))
cv_mean_all = float(np.sum(post_all * c_v_grid[None, :]))
mv_lo_all = discrete_quantile(m_v_grid, marg_m_all, 0.025)
mv_hi_all = discrete_quantile(m_v_grid, marg_m_all, 0.975)
cv_lo_all = discrete_quantile(c_v_grid, marg_c_all, 0.025)
cv_hi_all = discrete_quantile(c_v_grid, marg_c_all, 0.975)

# 0) Executive summary (first popup)
draw_executive_summary(mv_mean_all, mv_lo_all, mv_hi_all, cv_mean_all, cv_lo_all, cv_hi_all)

# 1) Prior-only predictive
plt.figure()
plt.plot(t_all, settlement_mm(t_all, m_v_median, c_v_median), "C0-o", label="Prior predictive (median)")
plt.scatter(t_all, y_all, color="k", s=30, zorder=3, label="Observed")
plt.xlabel("Time t (days)"); plt.ylabel("Settlement (mm)")
plt.title("Prior only")
plt.legend()
show_and_save(os.path.join(SAVE_DIR, "step0_prior_only.png"))

# Summary rows
rows_summary = []
rows_summary.append({
    "Stage": "Prior",
    "m_v (mean/median)": m_v_median,
    "m_v 95% CrI lo": math.exp(mu_log_m - 1.96 * sigma_log_m),
    "m_v 95% CrI hi": math.exp(mu_log_m + 1.96 * sigma_log_m),
    "c_v (mean/median)": c_v_median,
    "c_v 95% CrI lo": math.exp(mu_log_c - 1.96 * sigma_log_c),
    "c_v 95% CrI hi": math.exp(mu_log_c + 1.96 * sigma_log_c),
})

def draw_predictive_page(k: int, post_grid: np.ndarray, out_path: str):
    """Per-k predictive plot."""
    t_sub, y_sub = t_all[:k], y_all[:k]
    mv_mean_k = float(np.sum(post_grid * m_v_grid[:, None]))
    cv_mean_k = float(np.sum(post_grid * c_v_grid[None, :]))
    s_prior = settlement_mm(t_all, m_v_median, c_v_median)
    s_post = settlement_mm(t_all, mv_mean_k, cv_mean_k)

    plt.figure()
    plt.plot(t_all, s_prior, "C0-o", label="Prior predictive (median)")
    plt.plot(t_all, s_post, "C1-o", label=f"Posterior predictive (k={k}, mean)")
    plt.scatter(t_sub, y_sub, color="k", s=30, zorder=3, label=f"Observed (k={k})")
    plt.xlabel("Time t (days)"); plt.ylabel("Settlement (mm)")
    plt.title(f"Sequential update using first {k} observation(s)")
    plt.legend()
    show_and_save(out_path)

# k = 1..4 loop with per-step pages
post_k_cache = {}
for k in [1, 2, 3, 4]:
    t_sub, y_sub = t_all[:k], y_all[:k]
    post_k = posterior_on_grid(t_sub, y_sub)
    post_k_cache[k] = post_k

    draw_predictive_page(k, post_k, os.path.join(SAVE_DIR, f"step{k}_first_obs.png"))

    mv_mean_k = float(np.sum(post_k * m_v_grid[:, None]))
    cv_mean_k = float(np.sum(post_k * c_v_grid[None, :]))
    pred_all_k = settlement_mm(t_all, mv_mean_k, cv_mean_k)
    draw_numbered_equation(k, mv_mean_k, cv_mean_k, pred_all_k,
                           os.path.join(SAVE_DIR, f"equation_obs{k}.png"))

    _mv_k, _cv_k, _ = worked_example_figure(k, post_k,
                           os.path.join(SAVE_DIR, f"equation_obs{k}_worked_example.png"))

    marg_m_k = np.sum(post_k, axis=1)
    marg_c_k = np.sum(post_k, axis=0)
    rows_summary.append({
        "Stage": f"Posterior ({k} obs)",
        "m_v (mean/median)": mv_mean_k,
        "m_v 95% CrI lo": discrete_quantile(m_v_grid, marg_m_k, 0.025),
        "m_v 95% CrI hi": discrete_quantile(m_v_grid, marg_m_k, 0.975),
        "c_v (mean/median)": cv_mean_k,
        "c_v 95% CrI lo": discrete_quantile(c_v_grid, marg_c_k, 0.025),
        "c_v 95% CrI hi": discrete_quantile(c_v_grid, marg_c_k, 0.975),
    })

# ===== Final sequence after equation_obs4_worked_example =====

# A) Posterior joint heatmap (all data) — pcolormesh + contours
plt.figure()
pcm_joint = plt.pcolormesh(M_GRID, C_GRID, post_all, shading="auto", cmap="viridis")
cs_joint = plt.contour(m_v_grid, c_v_grid, post_all.T, levels=8, linewidths=0.8, colors="k")
plt.clabel(cs_joint, inline=True, fontsize=7)
plt.colorbar(pcm_joint, shrink=0.8, pad=0.06, label="Posterior probability (discrete)")
plt.xlabel("m_v (1/kPa)"); plt.ylabel("c_v (m^2/day)")
plt.title("Posterior joint (m_v, c_v | y) — all data")
show_and_save(os.path.join(SAVE_DIR, "fig_joint_posterior_heatmap.png"))

# B) k=1 3D + top-down
post_k1 = post_k_cache[1]
M1, C1 = np.meshgrid(m_v_grid, c_v_grid, indexing="ij")
Z1 = post_k1
mv_mean_k1 = float(np.sum(Z1 * m_v_grid[:, None]))
cv_mean_k1 = float(np.sum(Z1 * c_v_grid[None, :]))
i_map1, j_map1 = np.unravel_index(np.argmax(Z1), Z1.shape)
mv_map_k1, cv_map_k1 = float(m_v_grid[i_map1]), float(c_v_grid[j_map1])

fig1 = plt.figure(figsize=(10.5, 6.5))
ax3d_1 = fig1.add_subplot(121, projection="3d")
surf1 = ax3d_1.plot_surface(M1, C1, Z1, rstride=3, cstride=3, linewidth=0.0, antialiased=True, alpha=0.95, cmap="viridis")
ax3d_1.contour(M1, C1, Z1, zdir='z', offset=0, cmap="viridis", levels=8)
ax3d_1.set_title("k=1 Normalized Posterior — 3D Surface")
ax3d_1.set_xlabel("m_v (1/kPa)"); ax3d_1.set_ylabel("c_v (m^2/day)"); ax3d_1.set_zlabel("p(m_v,c_v | y₁)")
fig1.colorbar(surf1, ax=ax3d_1, shrink=0.65, pad=0.08)
ax3d_1.scatter(mv_map_k1, cv_map_k1, Z1[i_map1, j_map1], marker="^", s=60, label="MAP (▲)")
ax3d_1.scatter(mv_mean_k1, cv_mean_k1, Z1[np.argmin(abs(m_v_grid - mv_mean_k1)),
                                         np.argmin(abs(c_v_grid - cv_mean_k1))],
               marker="x", s=60, label="Mean (×)")
ax3d_1.legend(loc="upper left", fontsize=8)

ax2d_1 = fig1.add_subplot(122)
pcm1 = ax2d_1.pcolormesh(M1, C1, Z1, shading="auto", cmap="viridis")
cs1 = ax2d_1.contour(m_v_grid, c_v_grid, Z1.T, levels=8, linewidths=0.8, colors="k")
ax2d_1.clabel(cs1, inline=True, fontsize=7)
ax2d_1.scatter([mv_map_k1], [cv_map_k1], marker="^", s=70, label="MAP")
ax2d_1.scatter([mv_mean_k1], [cv_mean_k1], marker="x", s=70, label="Mean")
ax2d_1.set_title("Top-down with colormap + contours (k=1)")
ax2d_1.set_xlabel("m_v (1/kPa)")
ax2d_1.set_ylabel("c_v (m^2/day)")
fig1.colorbar(pcm1, ax=ax2d_1, shrink=0.75, pad=0.06)
ax2d_1.legend(loc="upper left", fontsize=8)
show_and_save(os.path.join(SAVE_DIR, "k1_post_surface.png"))

# C) k=4 3D + top-down
Z4 = post_all
mv_mean_k4 = float(np.sum(Z4 * m_v_grid[:, None]))
cv_mean_k4 = float(np.sum(Z4 * c_v_grid[None, :]))
i_map4, j_map4 = np.unravel_index(np.argmax(Z4), Z4.shape)
mv_map_k4, cv_map_k4 = float(m_v_grid[i_map4]), float(c_v_grid[j_map4])
M4, C4 = np.meshgrid(m_v_grid, c_v_grid, indexing="ij")

fig2 = plt.figure(figsize=(10.5, 6.5))
ax3d_4 = fig2.add_subplot(121, projection="3d")
surf4 = ax3d_4.plot_surface(M4, C4, Z4, rstride=3, cstride=3, linewidth=0.0, antialiased=True, alpha=0.95, cmap="viridis")
ax3d_4.contour(M4, C4, Z4, zdir='z', offset=0, cmap="viridis", levels=8)
ax3d_4.set_title("k=4 Normalized Posterior — 3D Surface")
ax3d_4.set_xlabel("m_v (1/kPa)"); ax3d_4.set_ylabel("c_v (m^2/day)"); ax3d_4.set_zlabel("p(m_v,c_v | y)")
fig2.colorbar(surf4, ax=ax3d_4, shrink=0.65, pad=0.08)
ax3d_4.scatter(mv_map_k4, cv_map_k4, Z4[i_map4, j_map4], marker="^", s=60, label="MAP (▲)")
ax3d_4.scatter(mv_mean_k4, cv_mean_k4, Z4[np.argmin(abs(m_v_grid - mv_mean_k4)),
                                         np.argmin(abs(c_v_grid - cv_mean_k4))],
               marker="x", s=60, label="Mean (×)")
ax3d_4.legend(loc="upper left", fontsize=8)

ax2d_4 = fig2.add_subplot(122)
pcm4 = ax2d_4.pcolormesh(M4, C4, Z4, shading="auto", cmap="viridis")
cs4 = ax2d_4.contour(m_v_grid, c_v_grid, Z4.T, levels=8, linewidths=0.8, colors="k")
ax2d_4.clabel(cs4, inline=True, fontsize=7)
ax2d_4.scatter([mv_map_k4], [cv_map_k4], marker="^", s=70, label="MAP")
ax2d_4.scatter([mv_mean_k4], [cv_mean_k4], marker="x", s=70, label="Mean")
ax2d_4.set_title("Top-down with colormap + contours (k=4)")
ax2d_4.set_xlabel("m_v (1/kPa)")
ax2d_4.set_ylabel("c_v (m^2/day)")
fig2.colorbar(pcm4, ax=ax2d_4, shrink=0.75, pad=0.06)
ax2d_4.legend(loc="upper left", fontsize=8)
show_and_save(os.path.join(SAVE_DIR, "k4_post_surface.png"))

# D) m_v prior vs posterior (all data) — posterior CrI only
plt.figure()
prior_mv_pdf = np.array([lognormal_pdf(x, mu_log_m, sigma_log_m) for x in m_v_grid])
prior_mv_pdf = prior_mv_pdf / np.trapezoid(prior_mv_pdf, m_v_grid)
post_mv_dens = pmf_to_density(m_v_grid, marg_m_all)
plt.plot(m_v_grid, prior_mv_pdf, label="Prior density")
plt.plot(m_v_grid, post_mv_dens, label="Posterior density (all data)")
plt.axvline(discrete_quantile(m_v_grid, marg_m_all, 0.025), linestyle=":", label="Post 95% lo (all)")
plt.axvline(discrete_quantile(m_v_grid, marg_m_all, 0.975), linestyle=":", label="Post 95% hi (all)")
plt.xlabel("m_v (1/kPa)"); plt.ylabel("Density"); plt.title("m_v: Prior vs Posterior (all data)")
plt.legend(fontsize=8, loc="upper right")
show_and_save(os.path.join(SAVE_DIR, "fig_mv_prior_posterior.png"))

# E) c_v prior vs posterior (all data) — posterior CrI only
plt.figure()
prior_cv_pdf = np.array([lognormal_pdf(x, mu_log_c, sigma_log_c) for x in c_v_grid])
prior_cv_pdf = prior_cv_pdf / np.trapezoid(prior_cv_pdf, c_v_grid)
post_cv_dens = pmf_to_density(c_v_grid, marg_c_all)
plt.plot(c_v_grid, prior_cv_pdf, label="Prior density")
plt.plot(c_v_grid, post_cv_dens, label="Posterior density (all data)")
plt.axvline(discrete_quantile(c_v_grid, marg_c_all, 0.025), linestyle=":", label="Post 95% lo (all)")
plt.axvline(discrete_quantile(c_v_grid, marg_c_all, 0.975), linestyle=":", label="Post 95% hi (all)")
plt.xlabel("c_v (m^2/day)"); plt.ylabel("Density"); plt.title("c_v: Prior vs Posterior (all data)")
plt.legend(fontsize=8, loc="upper right")
show_and_save(os.path.join(SAVE_DIR, "fig_cv_prior_posterior.png"))

# F) Summary equation page
plt.figure(figsize=(9.2, 5.2))
ax = plt.gca(); ax.axis("off")
main_eq = (r"$p(\theta\mid\mathbf{y}) \propto "
           r"\left[\prod_i \phi\!\left(\frac{y_i - s(t_i;\theta)}{\sigma_e}\right)\right] p(\theta),\ "
           r"\theta=(m_v,c_v)$")
ax.text(0.5, 0.65, main_eq, fontsize=18, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.35", ec="black", fc="none", lw=1.6))
with_line = (
    r"$s(t;\theta)=m_v\,\Delta\sigma'\,H\,U\!\left(\frac{c_v\,t}{(H/2)^2}\right),\ "
    r"U(T_v)\approx 1-\frac{8}{\pi^2}\exp\!\left(-\frac{\pi^2}{4}T_v\right).$"
)
ax.text(0.5, 0.30, with_line, fontsize=14, ha="center", va="center")
ax.set_title("Bayesian Updating Equation (Summary)", fontsize=14, pad=8)
show_and_save(os.path.join(SAVE_DIR, "equation_summary.png"))

# G) Parameter summary table
summary_df = pd.DataFrame(rows_summary)
plt.figure(figsize=(11.0, 6.0))
plt.axis("off")
plt.title("Parameter Summary (Prior and Sequential Posteriors)", fontsize=14, pad=12)

df_show = summary_df.copy()
def _fmt(col, name):
    if name.startswith("m_v"):
        return col.map(fmt_small)
    if name.startswith("c_v"):
        return col.map(fmt_std)
    return col
for col in df_show.columns:
    if col == "Stage":
        continue
    df_show[col] = _fmt(df_show[col], col)

table = plt.table(cellText=df_show.values,
                  colLabels=df_show.columns,
                  loc="center",
                  cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.05, 1.25)
show_and_save(os.path.join(SAVE_DIR, "param_summary.png"))

print("\nSaved to:", os.path.abspath(SAVE_DIR))
