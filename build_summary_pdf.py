# High-Quality PDF Report Builder (A4, crisp images, optional smart upsampling)
# - Page 1: "Geotechnical Summary Report"
# - Embeds each PNG at true 300–450 dpi on A4; avoids blur
# - Uses pcolormesh/contours etc. as rendered in your PNGs (no extra smoothing)
# - Preserves your figure order; skips any missing files

import os
import math
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Try Pillow for high-quality LANCZOS upsampling when needed
try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
OUTDIR = "outputs_bayes_consolidation"
PDF_PATH = os.path.join(OUTDIR, "Consolidation_Bayes_Summary.pdf")

# Your strict figure order
ORDER = [
    "step0_prior_only.png",
    "step1_first_obs.png",
    "equation_obs1.png",
    "equation_obs1_worked_example.png",
    "step2_first_two.png",
    "equation_obs2.png",
    "equation_obs2_worked_example.png",
    "step3_first_three.png",
    "equation_obs3.png",
    "equation_obs3_worked_example.png",
    "step4_all_four.png",
    "equation_obs4.png",
    "equation_obs4_worked_example.png",
    "fig_joint_posterior_heatmap.png",
    "k1_post_surface.png",
    "k4_post_surface.png",
    "fig_mv_prior_posterior.png",
    "fig_cv_prior_posterior.png",
    "equation_summary.png",
    "param_summary.png",
]

# A4 landscape page size in inches
A4_W_IN, A4_H_IN = 11.69, 8.27
MARGIN_IN = 0.35                  # page margin
USABLE_W_IN = A4_W_IN - 2 * MARGIN_IN
USABLE_H_IN = A4_H_IN - 2 * MARGIN_IN

# Embed quality targets
MIN_EMBED_DPI = 300               # minimum effective DPI you want in the PDF
MAX_EMBED_DPI = 450               # cap so we don't make pages tiny

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _safe_imread(path: str) -> np.ndarray:
    """Read PNG with matplotlib; returns float [0,1] or uint8."""
    img = plt.imread(path)
    if img.dtype in (np.float32, np.float64):
        img = np.clip(img, 0.0, 1.0)
    return img

def _optional_upsample(img: np.ndarray, scale: float) -> np.ndarray:
    """Upsample using PIL LANCZOS when scale>1 and Pillow available; else nearest."""
    if scale <= 1.0:
        return img
    if PIL_OK:
        arr = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.dtype != np.uint8 else img
        im = Image.fromarray(arr)
        w, h = im.size
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        im2 = im.resize(new_size, resample=Image.LANCZOS)
        return np.asarray(im2)
    # numpy fallback (nearest)
    rep = int(math.ceil(scale))
    if img.ndim == 3:
        return np.repeat(np.repeat(img, rep, axis=0), rep, axis=1)
    else:
        return np.repeat(np.repeat(img, rep, axis=0), rep, axis=1)

def _summary_page(pdf):
    """Insert a concise first-page geotechnical summary."""
    fig = plt.figure(figsize=(A4_W_IN, A4_H_IN))
    ax = fig.add_subplot(111); ax.axis("off")

    title = "Geotechnical Summary Report — Bayesian Updating for 1-D Consolidation"
    sub = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    assumptions = [
        "Soil model: Terzaghi 1-D, first-term approximation U(T_v)",
        "Layer thickness H = 5 m (double drainage ⇒ H/2 = 2.5 m)",
        "Surcharge Δσ' = 22 kPa  →  S_inf(mm) = 110000 · m_v",
        "Measurement noise σ_e = 3 mm (Gaussian, independent)",
        "Priors: independent lognormal on m_v and c_v",
        "Posterior: computed on a 161×161 log-spaced grid in (m_v, c_v)",
    ]
    workflow = [
        "1) Define (m_v, c_v) log grids; evaluate prior p(m_v)p(c_v).",
        "2) Compute s(t_i; θ) and likelihood φ((y_i − s)/σ_e) on the grid.",
        "3) Form unnormalized posterior = prior × likelihood; normalize via log-sum-exp.",
        "4) Summaries: posterior mean (m̂_v, ĉ_v), marginals, 95% credible intervals.",
        "5) Sequential updates for k = 1..4; 3-D and joint posterior maps generated.",
    ]

    bullets = "\n".join(f"• {x}" for x in assumptions)
    steps = "\n".join(workflow)

    text = (
        f"{title}\n\n{sub}\n\n"
        "Key Assumptions:\n"
        f"{bullets}\n\n"
        "Analysis Workflow:\n"
        f"{steps}\n\n"
        "Deliverables:\n"
        "• Sequential predictive figures (prior and k=1..4)\n"
        "• Worked-example panels (prior, likelihood, posterior) per k\n"
        "• Posterior joint heatmap (all data) and 3-D surfaces (k=1, k=4)\n"
        "• Marginal prior vs posterior plots (m_v, c_v) with 95% CrI\n"
        "• Summary equation page; parameter summary table\n"
    )
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=11, wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def _embed_image_page(pdf, img: np.ndarray, title: str):
    """
    Embed one image at high quality on A4 landscape:
      - Prefer not to resample; display size is computed to achieve MIN_EMBED_DPI.
      - If the image is too small to reach MIN_EMBED_DPI at full usable width/height,
        optionally upsample with LANCZOS to the minimum required size.
    """
    # Original pixels
    if img.ndim == 2:  # grayscale -> 3ch for consistent imshow
        img = np.stack([img, img, img], axis=-1)
    h_px, w_px = img.shape[0], img.shape[1]

    # Compute desired display size to meet DPI target, but cap DPI
    # At MIN_EMBED_DPI, the physical width (in) would be: w_px / MIN_EMBED_DPI
    disp_w_in = w_px / MIN_EMBED_DPI
    disp_h_in = h_px / MIN_EMBED_DPI

    # If too large for page, scale down to fit (DPI increases => even crisper)
    scale_fit = min(USABLE_W_IN / disp_w_in, USABLE_H_IN / disp_h_in, 1.0)
    disp_w_in *= scale_fit
    disp_h_in *= scale_fit

    # If too small on page (i.e., effective DPI would drop below MIN_EMBED_DPI because we need to fill more area),
    # we keep the smaller display size (higher DPI). If you want bigger visuals, we can upsample:
    # Determine effective DPI at the chosen display size:
    eff_dpi_w = w_px / disp_w_in if disp_w_in > 0 else MIN_EMBED_DPI
    eff_dpi_h = h_px / disp_h_in if disp_h_in > 0 else MIN_EMBED_DPI
    eff_dpi = min(eff_dpi_w, eff_dpi_h)

    # Optional smart upsampling when the image is genuinely tiny (rare for your figures)
    # Only upsample if effective DPI would exceed MAX_EMBED_DPI by >20% (i.e., the figure is *very* small on page)
    # and you'd prefer a larger on-page footprint without losing crispness.
    upscale = 1.0
    if eff_dpi > MAX_EMBED_DPI * 1.2:
        upscale = min(eff_dpi / MAX_EMBED_DPI, 4.0)  # cap at 4×
        img = _optional_upsample(img, upscale)
        # Recompute sizes with upsampled pixels
        h_px, w_px = img.shape[0], img.shape[1]
        disp_w_in = min(USABLE_W_IN, w_px / MAX_EMBED_DPI)
        disp_h_in = min(USABLE_H_IN, h_px / MAX_EMBED_DPI)

    # Create a page figure sized exactly to A4 landscape
    fig = plt.figure(figsize=(A4_W_IN, A4_H_IN))
    ax = fig.add_axes([
        (A4_W_IN - disp_w_in) / (2 * A4_W_IN),     # left (normalized)
        (A4_H_IN - disp_h_in) / (2 * A4_H_IN),     # bottom
        disp_w_in / A4_W_IN,                       # width
        disp_h_in / A4_H_IN                        # height
    ])
    ax.imshow(img, interpolation="none")  # no extra smoothing
    ax.axis("off")
    fig.suptitle(title, fontsize=10, y=0.995)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    with PdfPages(PDF_PATH) as pdf:
        # 1) Summary page
        _summary_page(pdf)

        # 2) Embed images in strict order with high effective DPI
        for name in ORDER:
            path = os.path.join(OUTDIR, name)
            if not os.path.isfile(path):
                print(f"[skip] {name} not found.")
                continue
            img = _safe_imread(path)
            _embed_image_page(pdf, img, title=name)

    print("Saved:", os.path.abspath(PDF_PATH))

if __name__ == "__main__":
    main()
