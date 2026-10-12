"""visualize_latin_coloring.py

Visualizes how Latin Square Coloring on 2D weight matrices causes
the dominant 2D off-diagonal Hessian cross-talk (same row / same column)
to destructively cancel out to EXACTLY ZERO over a cycle.

Demonstrates:
1. The Latin Coloring Grid: code(row, col) = (row + col) mod m
2. Probe patterns cycling through Hadamard basis.
3. Cumulative cross-talk between target weight W[2, 3] and its neighbors.
4. GLUON (Orthogonal cancellation to 0) vs. Hutchinson (Stochastic noise).
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
    np.random.seed(1337)
    m = 8  # Cycle of 8 probes
    rows, cols = 8, 8

    # 1. Generate Latin Coloring: code(a, b) = (a + b) % m
    a_grid = np.arange(rows)[:, None]
    b_grid = np.arange(cols)[None, :]
    latin_grid = (a_grid + b_grid) % m

    # 2. Hadamard matrix of order m
    H_had = generate_hadamard(m)  # (8, 8)

    # Deterministic global sign flip per coordinate (re-derived per cycle)
    flip = np.random.choice([-1.0, 1.0], size=(rows, cols))

    # Random Rademacher probes for Hutchinson comparison
    hutch_probes = np.random.choice([-1.0, 1.0], size=(m, rows, cols))

    target_r, target_c = 2, 3  # Target coordinate to inspect

    viz_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(viz_dir, exist_ok=True)
    frames = []

    print("Generating frames for latin_cancellation.gif...")

    # Track cumulative coherence (cross-talk) with same-row neighbors
    gluon_row_coherence = np.zeros(cols)
    hutch_row_coherence = np.zeros(cols)

    for step in range(m):
        # Current Hadamard probe pattern
        had_row = H_had[step]  # (m,)
        v_gluon_2d = flip * had_row[latin_grid]  # (8, 8)
        v_hutch_2d = hutch_probes[step]

        # Accumulate cross-talk with all coordinates in the same row: target_r
        # Coherence: v[target_r, target_c] * v[target_r, c]
        for c in range(cols):
            gluon_row_coherence[c] += v_gluon_2d[target_r, target_c] * v_gluon_2d[target_r, c]
            hutch_row_coherence[c] += v_hutch_2d[target_r, target_c] * v_hutch_2d[target_r, c]

        fig = plt.figure(figsize=(16, 9), dpi=100)
        fig.suptitle(
            f"GLUON 2D Latin-Square Coloring: Cycle Step {step+1} / {m}\n"
            r"Destructive Interference: Off-Diagonal 2D Cross-Talk $\sum_r v_i^r v_j^r \to 0$",
            fontsize=14, fontweight="bold", y=0.965
        )

        gs = fig.add_gridspec(2, 3, top=0.81, bottom=0.08, left=0.06, right=0.96, hspace=0.45, wspace=0.25)

        # Panel 1: The Latin Square Code Assignment
        ax1 = fig.add_subplot(gs[0, 0])
        cmap_codes = plt.get_cmap("tab10", m)
        im1 = ax1.imshow(latin_grid, cmap=cmap_codes, vmin=0, vmax=m - 1)
        ax1.set_title("Latin Code Grid: code(a, b) = (a + b) % 8", fontsize=10.5)
        ax1.set_xlabel("Column Index")
        ax1.set_ylabel("Row Index")
        # Annotate codes
        for r in range(rows):
            for c in range(cols):
                is_target = (r == target_r and c == target_c)
                color = "white" if is_target else "black"
                weight = "bold" if is_target else "normal"
                ax1.text(c, r, str(latin_grid[r, c]), ha="center", va="center",
                         color=color, fontsize=10, fontweight=weight)
        # Highlight target coordinate
        rect = plt.Rectangle((target_c - 0.45, target_r - 0.45), 0.9, 0.9,
                             fill=False, edgecolor="red", lw=2.5)
        ax1.add_patch(rect)
        fig.colorbar(im1, ax=ax1, ticks=range(m), fraction=0.046, pad=0.04)

        # Panel 2: Current Probe Pattern v^r (±1)
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(v_gluon_2d, cmap="bwr", vmin=-1, vmax=1)
        ax2.set_title(f"Probe Pattern at Step {step+1} (Hadamard Row {step})", fontsize=10.5)
        ax2.set_xlabel("Column Index")
        ax2.set_ylabel("Row Index")
        for r in range(rows):
            for c in range(cols):
                sym = "+" if v_gluon_2d[r, c] > 0 else "−"
                ax2.text(c, r, sym, ha="center", va="center", color="black", fontsize=11)
        fig.colorbar(im2, ax=ax2, ticks=[-1, 1], fraction=0.046, pad=0.04)

        # Panel 3: Orthogonality in the Target Row
        ax3 = fig.add_subplot(gs[0, 2])
        row_codes = latin_grid[target_r, :]
        bar_colors = [cmap_codes(cd) for cd in row_codes]
        bars = ax3.bar(range(cols), row_codes, color=bar_colors, edgecolor="black")
        bars[target_c].set_edgecolor("red")
        bars[target_c].set_linewidth(2.5)
        ax3.set_title(f"Codes Assigned in Row {target_r} (Target W[{target_r},{target_c}])\n(All 8 Neighbors Have Distinct Codes!)", fontsize=10.5)
        ax3.set_xlabel(f"Column in Row {target_r}")
        ax3.set_ylabel("Hadamard Code Index")
        ax3.set_ylim(0, m)
        ax3.grid(True, linestyle="--", alpha=0.5, axis="y")

        # Panel 4: Cumulative Off-Diagonal Cross-Talk (Same-Row Neighbors)
        ax4 = fig.add_subplot(gs[1, :2])
        col_indices = np.arange(cols)
        width = 0.35
        # Exclude self-target from cancellation (self-product should be step+1)
        off_diag_mask = col_indices != target_c

        ax4.bar(col_indices[off_diag_mask] - width / 2, hutch_row_coherence[off_diag_mask],
                width=width, color="#d62728", alpha=0.75, label="Hutchinson (Random Probes)")
        ax4.bar(col_indices[off_diag_mask] + width / 2, gluon_row_coherence[off_diag_mask],
                width=width, color="#2ca02c", alpha=0.85, label="GLUON (Latin Coded Probes)")

        ax4.set_title(f"Cumulative 2D Cross-Talk Interference with Row Neighbors at Step {step+1}\n"
                      r"$\sum_t v_{(2,3)}^t \cdot v_{(2,j)}^t$ ($j \neq 3$): Exact Destructive Cancellation to 0", fontsize=10.5)
        ax4.set_xlabel(f"Neighbor Column in Row {target_r}")
        ax4.set_ylabel("Cumulative Interference")
        ax4.set_ylim(-7.5, 7.5)
        ax4.axhline(0, color="black", lw=1.2, linestyle="-")
        ax4.grid(True, linestyle="--", alpha=0.5)
        ax4.legend(loc="upper right", fontsize=8.5)

        # Panel 5: Total Off-Diagonal Noise Energy over Time
        ax5 = fig.add_subplot(gs[1, 2])
        steps_arr = np.arange(1, step + 2)
        # Recompute trajectory of total off-diagonal noise
        g_noise_hist = []
        h_noise_hist = []
        g_acc_step = np.zeros(cols)
        h_acc_step = np.zeros(cols)
        for s in range(step + 1):
            vr_g = flip * H_had[s][latin_grid]
            vr_h = hutch_probes[s]
            for c in range(cols):
                g_acc_step[c] += vr_g[target_r, target_c] * vr_g[target_r, c]
                h_acc_step[c] += vr_h[target_r, target_c] * vr_h[target_r, c]
            g_noise_hist.append(np.sum(g_acc_step[off_diag_mask] ** 2))
            h_noise_hist.append(np.sum(h_acc_step[off_diag_mask] ** 2))

        ax5.plot(steps_arr, h_noise_hist, 's--', color="#d62728", lw=2, label="Hutchinson Noise")
        ax5.plot(steps_arr, g_noise_hist, 'o-', color="#2ca02c", lw=2.5, label="GLUON Noise")
        ax5.set_title("Total Row Cross-Talk Noise Energy", fontsize=10.5)
        ax5.set_xlabel("Cycle Steps ($1 \dots 8$)")
        ax5.set_ylabel("Total Off-Diagonal Cross-Talk Squared")
        ax5.set_xlim(1, 8)
        ax5.set_ylim(-2, 80)
        ax5.grid(True, linestyle="--", alpha=0.5)
        ax5.legend(loc="upper right", fontsize=8.5)

        # Render frame to image without tight_layout
        frame_path = os.path.join(viz_dir, f"_frame_latin_{step:02d}.png")
        fig.savefig(frame_path, dpi=100)
        plt.close(fig)
        frames.append(Image.open(frame_path))

    gif_path = os.path.join(viz_dir, "latin_cancellation.gif")
    durations = [450] * (m - 1) + [2500]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0
    )
    print(f"Saved: {gif_path}")

    for s in range(m):
        os.remove(os.path.join(viz_dir, f"_frame_latin_{s:02d}.png"))

if __name__ == "__main__":
    main()
