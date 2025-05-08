from krylov_pe.ocean import (
    build_depth_grid,
    exact_parabolic_equation,
    krylov_parabolic_equation,
    munk_profile,
    plot_pe_result,
    plot_pe_tl_at_z,
    simple_parabolic_equation,
)

import matplotlib.pyplot as plt


def main():
    """Plot the PE results for Munk profile using Crank Nicholson"""
    freq = 20  # Frequency (Hz)
    z_src = 1000  # Source depth (m)
    clims = (-110, -60)

    z = build_depth_grid(zmax=5000, dz=10)
    ssp = munk_profile(z)
    # ssp = 1460 + 0.016 * z  # Sound speed profile (m/s)

    press_bwd_euler, r = simple_parabolic_equation(
        freq, z, z_src, 1e5, 10, ssp, "backward_euler"
    )
    press_crank_nicholson, _ = simple_parabolic_equation(
        freq, z, z_src, 1e5, 10, ssp, "crank_nicholson"
    )
    press_krylov, _ = krylov_parabolic_equation(freq, z, z_src, 1e5, 10, ssp, 30)
    press_exact, _ = exact_parabolic_equation(freq, z, z_src, 1e5, 10, ssp)

    # Plot PE TL contours results
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    plot_pe_result(
        ax1,
        r,
        z,
        press_bwd_euler,
        title="Backward Euler",
        clims=clims,
    )
    plot_pe_result(
        ax2,
        r,
        z,
        press_crank_nicholson,
        title="Crank-Nicholson",
        clims=clims,
    )
    plot_pe_result(
        ax3,
        r,
        z,
        press_krylov,
        title="Krylov",
        clims=clims,
    )
    plot_pe_result(
        ax4,
        r,
        z,
        press_exact,
        title="Exact",
        clims=clims,
    )
    # ax1.set_xlim(0, 10)
    # ax2.set_xlim(0, 10)
    # ax3.set_xlim(0, 10)
    plt.tight_layout()

    fig2 = plt.figure(figsize=(10, 6))
    ax = fig2.add_subplot(1, 1, 1)
    plot_pe_tl_at_z(
        ax, r, z, press_bwd_euler, z_src, label="Backward Euler", dr_plot=100
    )
    plot_pe_tl_at_z(
        ax, r, z, press_crank_nicholson, z_src, label="Crank-Nicholson", dr_plot=100
    )
    plot_pe_tl_at_z(ax, r, z, press_krylov, z_src, label="Krylov", dr_plot=100)
    plot_pe_tl_at_z(ax, r, z, press_exact, z_src, label="Exact", dr_plot=100)
    ax.set_ylim(clims[0], clims[1])
    plt.show()
    print("Done!")


if __name__ == "__main__":
    main()
