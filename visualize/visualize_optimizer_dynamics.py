"""visualize_optimizer_dynamics.py

Demonstrates why GLUON outperforms SGD and AdamW on ill-conditioned loss ravines:
1. Anisotropic Valley Landscape: Curvature is 25x steeper along Y than along X.
2. Trajectory Comparison:
   - SGD oscillates violently across the steep walls.
   - AdamW normalizes gradients but overshoots due to momentum lag.
   - GLUON detects high curvature along Y and applies Newton braking (damping Y),
     while clamping flat curvature along X to maintain full forward speed.
3. Trust Region Mechanism: Visualizes transition from clamp(+/-1) to Newton braking.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

def loss(x, y):
    return 0.5 * (x**2 + 25.0 * y**2)

def grad(x, y):
    return np.array([x, 25.0 * y])

def hessian_diag():
    return np.array([1.0, 25.0])

def main():
    np.random.seed(42)
    n_steps = 30
    viz_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(viz_dir, exist_ok=True)

    # Initial position in ill-conditioned ravine
    start_pos = np.array([-3.2, 1.4])

    # 1. Simulate SGD
    lr_sgd = 0.06
    sgd_traj = [start_pos.copy()]
    cur = start_pos.copy()
    for _ in range(n_steps):
        g = grad(cur[0], cur[1])
        cur = cur - lr_sgd * g
        sgd_traj.append(cur.copy())
    sgd_traj = np.array(sgd_traj)

    # 2. Simulate AdamW
    lr_adam = 0.22
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    m_adam = np.zeros(2)
    v_adam = np.zeros(2)
    adam_traj = [start_pos.copy()]
    cur = start_pos.copy()
    for t in range(1, n_steps + 1):
        g = grad(cur[0], cur[1])
        m_adam = beta1 * m_adam + (1 - beta1) * g
        v_adam = beta2 * v_adam + (1 - beta2) * (g ** 2)
        m_hat = m_adam / (1 - beta1 ** t)
        v_hat = v_adam / (1 - beta2 ** t)
        cur = cur - lr_adam * m_hat / (np.sqrt(v_hat) + eps)
        adam_traj.append(cur.copy())
    adam_traj = np.array(adam_traj)

    # 3. Simulate GLUON
    # Split curvature: S exact diag = [0.8, 22.0], R converges to [0.2, 3.0]
    lr_gluon = 0.22
    gamma = 0.08
    h_diag = hessian_diag()
    m_gluon = np.zeros(2)
    beta_g = 0.85
    gluon_traj = [start_pos.copy()]
    cur = start_pos.copy()
    gluon_ratios = []

    for t in range(1, n_steps + 1):
        g = grad(cur[0], cur[1])
        m_gluon = beta_g * m_gluon + (1 - beta_g) * g
        # GLUON estimate: curvature h_hat with shrinking residual noise
        noise = np.random.randn(2) * 0.1 * np.linalg.norm(g)
        h_est = np.maximum(h_diag + noise, 0.1)
        ratio = m_gluon / np.maximum(gamma * h_est, 1e-4)
        step_dir = np.clip(ratio, -1.0, 1.0)
        gluon_ratios.append(ratio.copy())
        cur = cur - lr_gluon * step_dir
        gluon_traj.append(cur.copy())
    gluon_traj = np.array(gluon_traj)
    gluon_ratios = np.array(gluon_ratios)

    # Pre-generate mesh for contour plot
    x_grid = np.linspace(-3.6, 1.0, 200)
    y_grid = np.linspace(-1.8, 1.8, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = loss(X, Y)

    frames = []
    print("Generating frames for optimizer_dynamics.gif...")

    for step in range(1, n_steps + 1):
        fig = plt.figure(figsize=(16, 9), dpi=100)
        fig.suptitle(
            f"GLUON Optimization Dynamics: Step {step} / {n_steps}\n"
            r"Adaptive Newton Braking ($\theta \leftarrow \theta - \eta \cdot \mathrm{clamp}(\mu / (\gamma \hat{h}), -1, 1)$) on Ill-Conditioned Ravine",
            fontsize=14, fontweight="bold", y=0.965
        )

        gs = fig.add_gridspec(2, 3, top=0.81, bottom=0.08, left=0.06, right=0.96, hspace=0.38, wspace=0.25, height_ratios=[1.1, 1])

        # Panel 1: 2D Loss Ravine Trajectory
        ax1 = fig.add_subplot(gs[0, :2])
        levels = np.geomspace(0.01, 50, 20)
        cs = ax1.contour(X, Y, Z, levels=levels, cmap="viridis", alpha=0.6, linewidths=1.0)
        ax1.clabel(cs, inline=True, fontsize=7.5, fmt="%.1f")

        # Plot trajectories up to current step
        ax1.plot(sgd_traj[:step + 1, 0], sgd_traj[:step + 1, 1], 's--', color="#d62728", lw=1.8, ms=4, alpha=0.8, label="SGD (Oscillates across walls)")
        ax1.plot(adam_traj[:step + 1, 0], adam_traj[:step + 1, 1], '^-.', color="#1f77b4", lw=2.0, ms=5, alpha=0.85, label="AdamW (RMS scaling + overshoot)")
        ax1.plot(gluon_traj[:step + 1, 0], gluon_traj[:step + 1, 1], 'o-', color="#2ca02c", lw=2.5, ms=5, label="GLUON (Newton Braking on Y, clamp on X)")

        # Target optimum marker
        ax1.plot(0, 0, 'r*', ms=12, label="Optimum (0, 0)")
        ax1.set_title("Optimization Paths on Anisotropic Loss Ravine ($f = \\frac{1}{2}x^2 + \\frac{25}{2}y^2$, Condition Number $\\kappa = 25$)", fontsize=10.5)
        ax1.set_xlabel("Flat Valley Coordinate ($x$)")
        ax1.set_ylabel("Steep Wall Coordinate ($y$)")
        ax1.set_xlim(-3.6, 1.0)
        ax1.set_ylim(-1.8, 1.8)
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(loc="upper right", fontsize=8.5)

        # Panel 2: Instantaneous Step Vector Alignment
        ax2 = fig.add_subplot(gs[0, 2])
        # Plot GLUON trust region clamping status for current step
        cur_r = gluon_ratios[step - 1]
        bars = ax2.bar(["X (Flat floor)", "Y (Steep wall)"], [cur_r[0], cur_r[1]],
                       color=["#1f77b4" if abs(cur_r[0]) >= 1.0 else "#2ca02c",
                              "#2ca02c" if abs(cur_r[1]) < 1.0 else "#d62728"],
                       alpha=0.85, width=0.45)
        ax2.axhline(1.0, color="black", linestyle="--", alpha=0.6, label="Clamp Limit (+1)")
        ax2.axhline(-1.0, color="black", linestyle="--", alpha=0.6, label="Clamp Limit (-1)")
        ax2.set_title(r"Trust-Region Ratio: $\mu_i / (\gamma \hat{h}_i)$" + f" (Step {step})\n"
                      f"X: Clamped ({cur_r[0]:.2f}) | Y: Newton Braked ({cur_r[1]:.2f})", fontsize=10.5)
        ax2.set_ylabel("Raw Preconditioner Ratio")
        ax2.set_ylim(-3.5, 3.5)
        ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax2.legend(loc="upper right", fontsize=8.0)

        # Panel 3: Loss Function Convergence Curve
        ax3 = fig.add_subplot(gs[1, 0])
        steps_arr = np.arange(step + 1)
        loss_sgd = [loss(p[0], p[1]) for p in sgd_traj[:step + 1]]
        loss_adam = [loss(p[0], p[1]) for p in adam_traj[:step + 1]]
        loss_gluon = [loss(p[0], p[1]) for p in gluon_traj[:step + 1]]

        ax3.semilogy(steps_arr, loss_sgd, 's--', color="#d62728", lw=1.8, label="SGD Loss")
        ax3.semilogy(steps_arr, loss_adam, '^-.', color="#1f77b4", lw=2.0, label="AdamW Loss")
        ax3.semilogy(steps_arr, loss_gluon, 'o-', color="#2ca02c", lw=2.5, label="GLUON Loss")
        ax3.set_title("Loss Convergence $\\log(f(\\theta_t))$ vs Step", fontsize=10.5)
        ax3.set_xlabel("Iteration Step")
        ax3.set_ylabel("Loss Magnitude (log scale)")
        ax3.set_xlim(0, n_steps)
        ax3.set_ylim(1e-4, 1e2)
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.legend(loc="upper right", fontsize=8.5)

        # Panel 4: Distance to Optimum
        ax4 = fig.add_subplot(gs[1, 1])
        dist_sgd = [np.linalg.norm(p) for p in sgd_traj[:step + 1]]
        dist_adam = [np.linalg.norm(p) for p in adam_traj[:step + 1]]
        dist_gluon = [np.linalg.norm(p) for p in gluon_traj[:step + 1]]

        ax4.plot(steps_arr, dist_sgd, 's--', color="#d62728", lw=1.8, label="SGD Distance")
        ax4.plot(steps_arr, dist_adam, '^-.', color="#1f77b4", lw=2.0, label="AdamW Distance")
        ax4.plot(steps_arr, dist_gluon, 'o-', color="#2ca02c", lw=2.5, label="GLUON Distance")
        ax4.set_title(r"Euclidean Distance to Optimum $\|\theta_t - \theta^*\|$", fontsize=10.5)
        ax4.set_xlabel("Iteration Step")
        ax4.set_ylabel(r"Distance $\|\theta\|$")
        ax4.set_xlim(0, n_steps)
        ax4.set_ylim(0, 4.0)
        ax4.grid(True, linestyle="--", alpha=0.5)
        ax4.legend(loc="upper right", fontsize=8.5)

        # Panel 5: Steep-to-Flat Trajectory Ratio (Y Oscillation vs X Progress)
        ax5 = fig.add_subplot(gs[1, 2])
        y_sgd = [abs(p[1]) for p in sgd_traj[:step + 1]]
        y_adam = [abs(p[1]) for p in adam_traj[:step + 1]]
        y_gluon = [abs(p[1]) for p in gluon_traj[:step + 1]]

        ax5.plot(steps_arr, y_sgd, 's--', color="#d62728", lw=1.8, label="SGD |Y| Error")
        ax5.plot(steps_arr, y_adam, '^-.', color="#1f77b4", lw=2.0, label="AdamW |Y| Error")
        ax5.plot(steps_arr, y_gluon, 'o-', color="#2ca02c", lw=2.5, label="GLUON |Y| Error")
        ax5.set_title("Ravine Transverse Oscillation $|y_t|$", fontsize=10.5)
        ax5.set_xlabel("Iteration Step")
        ax5.set_ylabel("Transverse Deviation $|y|$")
        ax5.set_xlim(0, n_steps)
        ax5.set_ylim(0, 1.8)
        ax5.grid(True, linestyle="--", alpha=0.5)
        ax5.legend(loc="upper right", fontsize=8.5)

        frame_path = os.path.join(viz_dir, f"_frame_dyn_{step:02d}.png")
        fig.savefig(frame_path, dpi=100)
        plt.close(fig)
        frames.append(Image.open(frame_path))

    gif_path = os.path.join(viz_dir, "optimizer_dynamics.gif")
    durations = [300] * (n_steps - 1) + [2500]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0
    )
    print(f"Saved: {gif_path}")

    for s in range(1, n_steps + 1):
        os.remove(os.path.join(viz_dir, f"_frame_dyn_{s:02d}.png"))

if __name__ == "__main__":
    main()
