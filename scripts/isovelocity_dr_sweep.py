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
    tl = 20 * np.log10(np.abs(press))  # transmission loss in dB
    phase = np.angle(press)
    return tl, phase


def main():
    """Plot the PE results for Munk profile using Crank Nicholson"""
    freq = 20  # Frequency (Hz)
    z_src = 36  # Source depth (m)
    z_plt = z_src  # Depth to plot (m)

    m = 70  # Krylov subspace dimension - from the m_sweep script
    dr_exact = 5.0  # Base range in m
    r_max = 3e3  # Maximum range in m

    z = build_depth_grid(zmax=100, dz=1)
    ssp = 1500 * np.ones_like(z)  # Constant sound speed profile (m/s)

    # Loop to check the effects of dr on the different methods
    ref_wavelength = 1500 / freq  # Reference wavelength (m)
    max_wavelength_multiplier = 10
    dr_range = np.linspace(
        1, max_wavelength_multiplier * dr_exact, 21, dtype=np.float64
    )

    # exact solution should not be affected by dr
    press_exact, r_exact = exact_parabolic_equation(
        freq, z, z_src, r_max, dr_exact, ssp
    )

    # initialize RMSE arrays
    tl_rmse_bwd_euler = np.zeros_like(dr_range, dtype=float)
    phase_rmse_bwd_euler = np.zeros_like(dr_range, dtype=float)
    tl_rmse_crank_nicholson = np.zeros_like(dr_range, dtype=float)
    phase_rmse_crank_nicholson = np.zeros_like(dr_range, dtype=float)
    tl_rmse_krylov = np.zeros_like(dr_range, dtype=float)
    phase_rmse_krylov = np.zeros_like(dr_range, dtype=float)

    for i, dr_model in enumerate(dr_range):
        press_bwd_euler, r_euler = simple_parabolic_equation(
            freq, z, z_src, r_max, dr_model, ssp, "backward_euler"
        )
        press_crank_nicholson, r_cn = simple_parabolic_equation(
            freq, z, z_src, r_max, dr_model, ssp, "crank_nicholson"
        )
        press_krylov, r_krylov = krylov_parabolic_equation(
            freq,
            z,
            z_src,
            r_max,
            dr_model,
            ssp,
            m=int(m),
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
        tl_rmse_bwd_euler[i] = np.sqrt(np.mean((tl_exact_interp - tl_bwd_euler) ** 2))
        tl_rmse_crank_nicholson[i] = np.sqrt(
            np.mean((tl_exact_interp - tl_crank_nicholson) ** 2)
        )

        phase_rmse_bwd_euler[i] = np.sqrt(
            np.mean((phase_exact_interp - phase_bwd_euler) ** 2)
        )
        phase_rmse_crank_nicholson[i] = np.sqrt(
            np.mean((phase_exact_interp - phase_crank_nicholson) ** 2)
        )

        tl_krylov, phase_krylov = tl_and_phase(press_krylov[zind, :])
        tl_rmse_krylov[i] = np.sqrt(np.mean((tl_exact_interp - tl_krylov) ** 2))
        phase_rmse_krylov[i] = np.sqrt(
            np.mean((phase_exact_interp - phase_krylov) ** 2)
        )

    fig = plt.figure(figsize=(10, 6))
    # plot magnitude RMSE
    ax1 = fig.add_subplot(1, 1, 1)
    # ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(
        dr_range / dr_exact,
        tl_rmse_bwd_euler,
        "ro-",
        label="Backward Euler",
    )
    ax1.plot(
        dr_range / dr_exact,
        tl_rmse_crank_nicholson,
        "bo-",
        label="Crank-Nicholson",
    )
    ax1.plot(dr_range / dr_exact, tl_rmse_krylov, "ko-", label="Krylov")
    ax1.set_xlabel(r"$\Delta r / \Delta r_exact$")
    ax1.set_ylabel("TL RMSE (dB)")
    ax1.set_title("TL RMSE at z = %.1f m" % z_plt)

    # plot the phase RMSE
    # ax2 = fig.add_subplot(2, 1, 2)
    # ax2.plot(
    #     dr_range / ref_wavelength,
    #     phase_rmse_bwd_euler,
    #     "ro-",
    #     label="bwd Euler",
    # )
    # ax2.plot(
    #     dr_range / ref_wavelength,
    #     phase_rmse_crank_nicholson,
    #     "bo-",
    #     label="Crank-Nicholson",
    # )
    # ax2.plot(dr_range / ref_wavelength, phase_rmse_krylov, "ko-", label="Krylov")
    # ax2.set_xlabel(r"$\Delta r / \lambda_0$")
    # ax2.set_ylabel("Phase RMSE (radians)")
    # ax2.set_title("Phase RMSE at z = %.1f m" % z_plt)

    ax1.legend(loc="best")
    ax1.grid()
    # ax2.grid()
    plt.rc("text", usetex=True)
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    main()
