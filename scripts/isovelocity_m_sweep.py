from krylov_pe.ocean import (
    build_depth_grid,
    exact_parabolic_equation,
    krylov_parabolic_equation,
    simple_parabolic_equation,
)

import numpy as np
import matplotlib.pyplot as plt


def tl_and_phase(press):
    """Calculate the magnitude and phase of the pressure field."""
    tl = 20 * np.log10(np.abs(press))
    phase = np.angle(press)
    return tl, phase


def main():
    """Plot the PE results for Munk profile using Crank Nicholson"""
    freq = 20  # Frequency (Hz)
    z_src = 36  # Source depth (m)
    z_plt = z_src  # Depth to plot (m)

    # m_range for krylov method
    m_range = np.arange(10, 101, 10)

    # freq = 20  # Frequency (Hz)
    # z_src = 1000  # Source depth (m)
    # clims = (-110, -60)

    dr_model = 30  # Model range in m
    ref_wavelength = 1500 / freq  # Reference wavelength (m)
    dr_exact = 37.5  # Base range step in m
    dr_model = 37.5  # Model range step in m
    r_max = 3e3  # Maximum range in m

    z = build_depth_grid(zmax=100, dz=1)
    ssp = 1500 * np.ones_like(z)  # Constant sound speed profile (m/s)
    # z = build_depth_grid(zmax=5000, dz=10)
    # ssp = munk_profile(z)

    press_exact, r_exact = exact_parabolic_equation(
        freq, z, z_src, r_max, dr_exact, ssp
    )
    press_bwd_euler, r_euler = simple_parabolic_equation(
        freq, z, z_src, r_max, 37.5, ssp, "backward_euler"
    )
    press_crank_nicholson, r_cn = simple_parabolic_equation(
        freq, z, z_src, r_max, 37.5, ssp, "crank_nicholson"
    )

    # plot the phase and magnitude error at z_plt compared to the exact solution

    # get the variables at z_plt
    zind = np.argmin(z - z_plt)
    tl_exact, phase_exact = tl_and_phase(press_exact[zind, :])
    tl_bwd_euler, phase_bwd_euler = tl_and_phase(press_bwd_euler[zind, :])
    tl_crank_nicholson, phase_crank_nicholson = tl_and_phase(
        press_crank_nicholson[zind, :]
    )

    # interpolate the exact solution to the model range
    tl_exact_interp = np.interp(r_euler, r_exact, tl_exact)
    phase_exact_interp = np.interp(r_euler, r_exact, phase_exact)

    # calculate the RMSE for each method
    tl_rmse_bwd_euler = np.sqrt(np.mean((tl_exact_interp - tl_bwd_euler) ** 2))
    tl_rmse_crank_nicholson = np.sqrt(
        np.mean((tl_exact_interp - tl_crank_nicholson) ** 2)
    )

    phase_rmse_bwd_euler = np.sqrt(np.mean((phase_exact_interp - phase_bwd_euler) ** 2))
    phase_rmse_crank_nicholson = np.sqrt(
        np.mean((phase_exact_interp - phase_crank_nicholson) ** 2)
    )

    tl_rmse_krylov = np.zeros_like(m_range, dtype=float)
    phase_rmse_krylov = np.zeros_like(m_range, dtype=float)
    for i, m in enumerate(m_range):
        press_krylov, r_krylov = krylov_parabolic_equation(
            freq,
            z,
            z_src,
            r_max,
            dr_model,
            ssp,
            m=int(m),
        )
        tl_exact_interp_kry = np.interp(r_krylov, r_exact, tl_exact)
        phase_exact_interp_kry = np.interp(r_krylov, r_exact, phase_exact)
        tl_krylov, phase_krylov = tl_and_phase(press_krylov[zind, :])
        tl_rmse_krylov[i] = np.sqrt(np.mean((tl_exact_interp_kry - tl_krylov) ** 2))
        phase_rmse_krylov[i] = np.sqrt(
            np.mean((phase_exact_interp_kry - phase_krylov) ** 2)
        )

    fig = plt.figure(figsize=(10, 6))
    # plot magnitude RMSE
    ax1 = fig.add_subplot(1, 1, 1)
    # ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(
        [m_range[0], m_range[-1]],
        [tl_rmse_bwd_euler, tl_rmse_bwd_euler],
        "r--",
        label="bwd Euler",
    )
    ax1.plot(
        [m_range[0], m_range[-1]],
        [tl_rmse_crank_nicholson, tl_rmse_crank_nicholson],
        "b--",
        label="Crank-Nicholson",
    )
    ax1.plot(m_range, tl_rmse_krylov, "ko-", label="Krylov")
    ax1.set_xlabel("Krylov subspace dimension (m)")
    ax1.set_ylabel("TL RMSE (dB)")
    ax1.set_title(
        f"TL RMSE at z = {z_plt} m ($\Delta r / \lambda_0$ = {dr_model / ref_wavelength:0.3f})"
    )

    # plot the phase RMSE
    # ax2 = fig.add_subplot(2, 1, 2)
    # ax2.plot(
    #     [m_range[0], m_range[-1]],
    #     [phase_rmse_bwd_euler, phase_rmse_bwd_euler],
    #     "r--",
    #     label="bwd Euler",
    # )
    # ax2.plot(
    #     [m_range[0], m_range[-1]],
    #     [phase_rmse_crank_nicholson, phase_rmse_crank_nicholson],
    #     "b--",
    #     label="Crank-Nicholson",
    # )
    # ax2.plot(m_range, phase_rmse_krylov, "ko-", label="Krylov")
    # ax2.set_xlabel("Krylov subspace dimension (m)")
    # ax2.set_ylabel("Phase RMSE (radians)")
    # ax2.set_title("Phase RMSE at z = %.1f m" % z_plt)

    ax1.legend(loc="best")
    ax1.grid()
    # ax2.grid()
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    main()
