"""visualize_tomography.py

Animates the '1D Tomographic Shadow' concept and shows:
1. True Curvature Matrix M (ground truth).
2. The 1D Probe Vector v_k (the 1D directional angle / shadow).
3. The 1D Projection (the shadow response p_k = M * v_k).
4. Cumulative Diagonal Reconstruction:
   - Standard Stochastic Hutchinson (random +/-1): noisy cross-talk persists.
   - GLUON Coded Tomography (Hadamard + Latin codes): 2D cross-talk cancels to 0.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

def generate_hadamard(m):
    H = np.array([[1.0]])
    while H.shape[0] < m:
        H = np.block([[H, H], [H, -H]])
    return H

def main():
    np.random.seed(42)
    m = 16  # Cycle length (16 probes)
    n = 16  # Matrix dimension (16x16 coordinates)

    # Construct a realistic residual matrix R (symmetric, non-diagonal)
    # Strong diagonal + structured off-diagonal couplings
    diag_true = np.array([2.5, 1.8, 3.2, 0.9, 2.1, 4.0, 1.5, 2.8,
                          3.5, 1.2, 2.0, 3.8, 1.7, 2.9, 3.1, 1.4])
    A = np.random.randn(n, n) * 0.7
    off_diag = 0.5 * (A + A.T)
    np.fill_diagonal(off_diag, 0.0)
    R = np.diag(diag_true) + off_diag

    # Hadamard codes & random flip
    H_had = generate_hadamard(m)  # (16, 16)
    # Assign distinct codes to coordinates: code(i) = i % m
    codes = np.arange(n) % m
    global_flip = np.random.choice([-1.0, 1.0], size=n)

    # Storage for cumulative estimates
    gluon_acc = np.zeros(n)
    hutch_acc = np.zeros(n)
    
    # Store random hutchinson vectors
    hutch_probes = np.random.choice([-1.0, 1.0], size=(m, n))

    frames = []
    viz_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(viz_dir, exist_ok=True)

    print("Generating frames for tomography_shadows.gif...")

    for k in range(m):
        # 1. GLUON probe vector: d * Hadamard[k, code]
        v_gluon = global_flip * H_had[k, codes]
        # 1D shadow response: p = R * v
        p_gluon = R @ v_gluon
        # Tomographic diagonal readout: v * p
        gluon_acc += v_gluon * p_gluon

        # 2. Hutchinson probe: random +/- 1
        v_hutch = hutch_probes[k]
        p_hutch = R @ v_hutch
        hutch_acc += v_hutch * p_hutch

        # Current means
        gluon_est = gluon_acc / (k + 1)
        hutch_est = hutch_acc / (k + 1)

        # Plot setup
        fig = plt.figure(figsize=(16, 9), dpi=100)
        fig.suptitle(
            f"GLUON 1D Tomographic Shadows: Step {k+1} / {m}\n"
            r"Extracting 2D Curvature from 1D Directional Responses ($p_k = M v_k$)",
            fontsize=14, fontweight="bold", y=0.965
        )

        gs = fig.add_gridspec(2, 3, top=0.81, bottom=0.08, left=0.06, right=0.96, hspace=0.38, wspace=0.25, height_ratios=[1.1, 1])

        # Panel 1: The 2D Curvature Matrix M
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(R, cmap="coolwarm", vmin=-2, vmax=4)
        ax1.set_title("True 2D Curvature Matrix ($R_{ij}$)\n(Diagonal to recover + Off-diagonal cross-talk)", fontsize=10.5)
        ax1.set_xlabel("Coordinate $j$")
        ax1.set_ylabel("Coordinate $i$")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Panel 2: The 1D Probe Angle v_k and Shadow Response p_k
        ax2 = fig.add_subplot(gs[0, 1])
        x_coords = np.arange(n)
        ax2.step(x_coords, v_gluon, where="mid", color="#1f77b4", lw=2, label="Probe $v_k$ (±1 Code)")
        ax2.plot(x_coords, p_gluon, 'o-', color="#ff7f0e", lw=1.5, ms=5, label="1D Shadow $p_k = R v_k$")
        ax2.set_title(f"1D Shadow at Step {k+1}\n(Directional Projection along $v_k$)", fontsize=10.5)
        ax2.set_xlabel("Parameter Coordinate")
        ax2.set_ylabel("Amplitude")
        ax2.set_ylim(-11, 11)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right", fontsize=8.5)

        # Panel 3: Interference Cancellation in 1D Shadow
        ax3 = fig.add_subplot(gs[0, 2])
        # Show how much off-diagonal interference is in current cumulative estimate
        err_gluon = np.abs(gluon_est - diag_true)
        err_hutch = np.abs(hutch_est - diag_true)
        ax3.plot(x_coords, err_hutch, 's--', color="#d62728", lw=2, label="Hutchinson Error (Random)")
        ax3.plot(x_coords, err_gluon, 'o-', color="#2ca02c", lw=2.5, label="GLUON Error (Hadamard)")
        ax3.set_title(f"Reconstruction Error at Step {k+1}\n($|\\hat{{h}}_i - H_{{ii}}|$ per coordinate)", fontsize=10.5)
        ax3.set_xlabel("Parameter Coordinate")
        ax3.set_ylabel("Absolute Error")
        ax3.set_ylim(0, 3.8)
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.legend(loc="upper right", fontsize=8.5)

        # Panel 4: Reconstructed Diagonal Comparison
        ax4 = fig.add_subplot(gs[1, :2])
        width = 0.28
        ax4.bar(x_coords - width, diag_true, width=width, color="#7f7f7f", alpha=0.8, label="Ground Truth $H_{ii}$")
        ax4.bar(x_coords, hutch_est, width=width, color="#d62728", alpha=0.7, label=f"Hutchinson Mean (k={k+1})")
        ax4.bar(x_coords + width, gluon_est, width=width, color="#2ca02c", alpha=0.85, label=f"GLUON Coded Mean (k={k+1})")
        ax4.set_title("Tomographic Diagonal Recovery vs. Ground Truth", fontsize=10.5)
        ax4.set_xlabel("Parameter Coordinate")
        ax4.set_ylabel("Curvature Value")
        ax4.set_ylim(-1, 6.8)
        ax4.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax4.legend(loc="upper right", fontsize=8.5)

        # Panel 5: Mean Squared Error Tracking (k = 1..k)
        ax5 = fig.add_subplot(gs[1, 2])
        # Compute trajectory of errors up to current step
        k_steps = np.arange(1, k + 2)
        hutch_mses = []
        gluon_mses = []
        h_tmp = np.zeros(n)
        g_tmp = np.zeros(n)
        for s in range(k + 1):
            g_tmp += (global_flip * H_had[s, codes]) * (R @ (global_flip * H_had[s, codes]))
            h_tmp += hutch_probes[s] * (R @ hutch_probes[s])
            gluon_mses.append(np.mean((g_tmp / (s + 1) - diag_true) ** 2))
            hutch_mses.append(np.mean((h_tmp / (s + 1) - diag_true) ** 2))

        ax5.plot(k_steps, hutch_mses, 's--', color="#d62728", lw=2, label="Hutchinson MSE")
        ax5.plot(k_steps, gluon_mses, 'o-', color="#2ca02c", lw=2.5, label="GLUON Coded MSE")
        ax5.set_title("MSE vs Probe Count (1 to 16)", fontsize=10.5)
        ax5.set_xlabel("Number of 1D Shadows ($k$)")
        ax5.set_ylabel("Mean Squared Error")
        ax5.set_xlim(1, 16)
        ax5.set_ylim(-0.1, 4.0)
        ax5.grid(True, linestyle="--", alpha=0.5)
        ax5.legend(loc="upper right", fontsize=8.5)

        # Render frame to image without tight_layout
        frame_path = os.path.join(viz_dir, f"_frame_tomo_{k:02d}.png")
        fig.savefig(frame_path, dpi=100)
        plt.close(fig)

        frames.append(Image.open(frame_path))

    # Save as animated GIF (repeat = 0 means infinite loop, duration in ms per frame)
    gif_path = os.path.join(viz_dir, "tomography_shadows.gif")
    # Hold the last frame a bit longer so user can absorb the final result
    durations = [350] * (m - 1) + [2000]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0
    )
    print(f"Saved: {gif_path}")

    # Clean up temp frames
    for k in range(m):
        os.remove(os.path.join(viz_dir, f"_frame_tomo_{k:02d}.png"))

if __name__ == "__main__":
    main()
