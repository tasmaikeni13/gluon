"""visualize_split.py

Visualizes the core mathematical mechanism of GLUON:
1. The Split: H = S + R
   - S : Structural Core (PSD, Fisher, diag >= 0)
   - R : Residual (linear in loss gradient, ||R|| ~ ||grad loss||)
2. Trajectory Evolution: As training progresses, ||grad loss|| -> 0.
   - Hutchinson variance on H: PERSISTS FOREVER (bounded by S).
   - GLUON variance on R: COLLAPSES TO ZERO (proportional to ||grad loss||^2).
3. The Trust Region: Coordinates transition from clipped momentum (+/-1)
   to confident Newton braking (mu / gamma*h).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

def main():
    np.random.seed(42)
    n_steps = 24
    temp_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(temp_dir, exist_ok=True)

    # Simulated training progression: gradient norm shrinks along trajectory
    steps = np.arange(1, n_steps + 1)
    grad_norms = np.exp(-0.12 * steps) + 0.02

    # Fixed structural core diagonal (Fisher sensitivity)
    n_params = 12
    s_diag = np.array([2.8, 3.5, 1.2, 4.1, 0.8, 2.2, 3.0, 1.9, 2.5, 3.8, 1.5, 2.0])
    # S has strong off-diagonal energy (why probing H directly is noisy)
    s_offdiag_var = 1.8

    # Residual ground truth diagonal (scales with grad_norm)
    r_diag_base = np.array([-0.8, 1.2, -0.5, 0.9, -1.1, 0.4, -0.6, 0.7, -0.3, 0.8, -0.5, 0.2])

    frames = []
    viz_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(viz_dir, exist_ok=True)
    print("Generating frames for split_variance_collapse.gif...")

    hutch_errors = []
    gluon_errors = []

    for idx, (step, g_norm) in enumerate(zip(steps, grad_norms)):
        # True residual at this step
        r_diag_cur = r_diag_base * g_norm
        h_diag_cur = s_diag + r_diag_cur

        # Estimated values:
        # Hutchinson on full H has variance driven by off-diagonal of H (mostly S)
        hutch_noise_std = np.sqrt(s_offdiag_var + 0.5 * (g_norm ** 2))
        hutch_est = h_diag_cur + np.random.randn(n_params) * hutch_noise_std
        hutch_err = np.mean(np.abs(hutch_est - h_diag_cur))
        hutch_errors.append(hutch_err)

        # GLUON: S is known exactly (from Fisher sampling), R probed with variance ~ ||grad||^2
        gluon_noise_std = 0.35 * g_norm  # Shrinks to zero!
        gluon_est = s_diag + (r_diag_cur + np.random.randn(n_params) * gluon_noise_std)
        gluon_err = np.mean(np.abs(gluon_est - h_diag_cur))
        gluon_errors.append(gluon_err)

        fig = plt.figure(figsize=(16, 9), dpi=100)
        fig.suptitle(
            f"GLUON Split-Curvature Dynamics: Training Step {step} / {n_steps}\n"
            r"Curvature Split: $H = S + R$ | Core $S$ (exact) + Residual $R$ (variance $\propto \|\nabla\ell\|^2 \to 0$)",
            fontsize=14, fontweight="bold", y=0.965
        )

        gs = fig.add_gridspec(2, 3, top=0.81, bottom=0.08, left=0.06, right=0.96, hspace=0.38, wspace=0.25, height_ratios=[1.1, 1])

        # Panel 1: Loss Gradient Norm Shrinking
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(steps[:idx + 1], grad_norms[:idx + 1], 'o-', color="#1f77b4", lw=2.5, ms=5)
        ax1.set_title(r"Loss Gradient $\|\nabla \ell\|$ (Converging)", fontsize=10.5)
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel(r"Gradient Magnitude $\|\nabla \ell\|$")
        ax1.set_xlim(1, n_steps)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Panel 2: The Curvature Split (S vs. R)
        ax2 = fig.add_subplot(gs[0, 1])
        x_coords = np.arange(n_params)
        width = 0.35
        ax2.bar(x_coords - width / 2, s_diag, width=width, color="#2ca02c", alpha=0.85, label="Core $S_{ii}$ (PSD Fisher)")
        ax2.bar(x_coords + width / 2, r_diag_cur, width=width, color="#ff7f0e", alpha=0.85, label="Residual $R_{ii}$ (Shrinking)")
        ax2.axhline(0, color="black", lw=1)
        ax2.set_title("Curvature Decomposition: $S$ vs. $R$", fontsize=10.5)
        ax2.set_xlabel("Coordinate Index")
        ax2.set_ylabel("Curvature Amplitude")
        ax2.set_ylim(-1.5, 4.8)
        ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax2.legend(loc="upper right", fontsize=8.5)

        # Panel 3: Preconditioner Trust Region (Clipping vs. Newton)
        ax3 = fig.add_subplot(gs[0, 2])
        # Simulate momentum mu / (gamma * h)
        mu = np.random.randn(n_params) * 0.05
        gamma = 0.02
        raw_ratio = mu / np.maximum(gamma * gluon_est, 1e-4)
        clipped_ratio = np.clip(raw_ratio, -1.0, 1.0)
        is_clipped = np.abs(raw_ratio) >= 1.0

        bar_colors = ["#d62728" if c else "#1f77b4" for c in is_clipped]
        ax3.bar(x_coords, clipped_ratio, color=bar_colors, alpha=0.85)
        ax3.axhline(1.0, color="black", linestyle="--", alpha=0.6)
        ax3.axhline(-1.0, color="black", linestyle="--", alpha=0.6)
        ax3.set_title(r"Trust Region Step: $\mathrm{clamp}(\mu / (\gamma \hat{h}), -1, 1)$" + "\n"
                      f"({np.sum(~is_clipped)}/{n_params} in True Newton Braking)", fontsize=10.5)
        ax3.set_xlabel("Coordinate Index")
        ax3.set_ylabel("Step Direction / Magnitude")
        ax3.set_ylim(-1.25, 1.25)
        ax3.grid(True, linestyle="--", alpha=0.5, axis="y")

        # Panel 4: Diagonal Estimation Comparison at Current Step
        ax4 = fig.add_subplot(gs[1, :2])
        w = 0.28
        ax4.bar(x_coords - w, h_diag_cur, width=w, color="#7f7f7f", alpha=0.8, label="True Curvature $H_{ii} = S_{ii} + R_{ii}$")
        ax4.bar(x_coords, hutch_est, width=w, color="#d62728", alpha=0.7, label="Direct Hessian Probe (Hutchinson)")
        ax4.bar(x_coords + w, gluon_est, width=w, color="#2ca02c", alpha=0.85, label="GLUON Split Probe ($S$ + $R$ probe)")
        ax4.set_title(f"Estimated Curvature Diagonal at Step {step}", fontsize=10.5)
        ax4.set_xlabel("Parameter Coordinate")
        ax4.set_ylabel("Curvature Value")
        ax4.set_ylim(-1.5, 7.0)
        ax4.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax4.legend(loc="upper right", fontsize=8.5)

        # Panel 5: Variance / Error Contraction Curve
        ax5 = fig.add_subplot(gs[1, 2])
        cur_s = np.arange(1, idx + 2)
        ax5.plot(cur_s, hutch_errors, 's--', color="#d62728", lw=2, label="Hutchinson Error (Plateaus)")
        ax5.plot(cur_s, gluon_errors, 'o-', color="#2ca02c", lw=2.5, label=r"GLUON Error ($\propto \|\nabla \ell\|^2 \to 0$)")
        ax5.set_title("Curvature Estimation Error over Time", fontsize=10.5)
        ax5.set_xlabel("Training Steps")
        ax5.set_ylabel("Mean Absolute Error")
        ax5.set_xlim(1, n_steps)
        ax5.set_ylim(0, 2.5)
        ax5.grid(True, linestyle="--", alpha=0.5)
        ax5.legend(loc="upper right", fontsize=8.5)

        # Render frame to image without tight_layout
        frame_path = os.path.join(viz_dir, f"_frame_split_{step:02d}.png")
        fig.savefig(frame_path, dpi=100)
        plt.close(fig)
        frames.append(Image.open(frame_path))

    gif_path = os.path.join(viz_dir, "split_variance_collapse.gif")
    durations = [350] * (n_steps - 1) + [2500]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0
    )
    print(f"Saved: {gif_path}")

    for s in range(1, n_steps + 1):
        os.remove(os.path.join(viz_dir, f"_frame_split_{s:02d}.png"))

if __name__ == "__main__":
    main()
