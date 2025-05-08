from krylov_pe.ocean import (
    build_depth_grid,
    exact_parabolic_equation,
    krylov_parabolic_equation,
    plot_pe_result,
    plot_pe_tl_at_z,
    simple_parabolic_equation,
)

import numpy as np
import matplotlib.pyplot as plt


def main():
    """Plot the PE results for Munk profile using Crank Nicholson"""
    freq = 20  # Frequency (Hz)
    z_src = 36  # Source depth (m)
    clims = (-100, -40)
    dr_exact = 5.0  # Base range in m
    dr_model = 15.0  # Model range step in m

    z = build_depth_grid(zmax=100, dz=1)
    ssp = 1500 * np.ones_like(z)  # Constant sound speed profile (m/s)

    press_bwd_euler, r_euler = simple_parabolic_equation(
        freq, z, z_src, 3e3, dr_model, ssp, "backward_euler"
    )
    press_crank_nicholson, r_cn = simple_parabolic_equation(
        freq, z, z_src, 3e3, dr_model, ssp, "crank_nicholson"
    )
    press_krylov, r_krylov = krylov_parabolic_equation(
        freq, z, z_src, 3e3, dr_model, ssp, 40
    )
    press_exact, r_exact = exact_parabolic_equation(freq, z, z_src, 3e3, dr_exact, ssp)

    # Plot PE TL contours results
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    plot_pe_result(
        ax1,
        r_euler,
        z,
        press_bwd_euler,
        title=r"Backward Euler ($\Delta r = 15$ m)",
        clims=clims,
    )
    plot_pe_result(
        ax2,
        r_cn,
        z,
        press_crank_nicholson,
        title=r"Crank-Nicholson ($\Delta r = 15$ m)",
        clims=clims,
    )
    plot_pe_result(
        ax3,
        r_krylov,
        z,
        press_krylov,
        title=r"Krylov ($\Delta r = 15$ m, $m=40$)",
        clims=clims,
    )
    plot_pe_result(
        ax4,
        r_exact,
        z,
        press_exact,
        title=r"Exact ($\Delta r = 5$ m)",
        clims=clims,
    )
    # ax1.set_xlim(0, 10)
    # ax2.set_xlim(0, 10)
    # ax3.set_xlim(0, 10)
    plt.tight_layout()

    fig2 = plt.figure(figsize=(10, 6))
    ax = fig2.add_subplot(1, 1, 1)
    plot_pe_tl_at_z(
        ax, r_euler, z, press_bwd_euler, z_src, label="Backward Euler", dr_plot=5
    )
    plot_pe_tl_at_z(
        ax, r_cn, z, press_crank_nicholson, z_src, label="Crank-Nicholson", dr_plot=5
    )
    plot_pe_tl_at_z(ax, r_krylov, z, press_krylov, z_src, label="Krylov", dr_plot=10)
    plot_pe_tl_at_z(ax, r_exact, z, press_exact, z_src, label="Exact", dr_plot=5)
    ax.set_ylim(clims[0], clims[1])
    plt.show()
    print("Done!")


if __name__ == "__main__":
    main()
