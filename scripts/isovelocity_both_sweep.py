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
    freq = 100  # Frequency (Hz)
    z_src = 36  # Source depth (m)
    z_plt = z_src  # Depth to plot (m)

    # m_range for krylov method
    m_range = np.arange(20, 101, 5)

    # Loop to check the effects of dr on the different methods
    ref_wavelength = 1500 / freq  # Reference wavelength (m)
    max_wavelength_multiplier = 4
    dr_range = np.linspace(
        1, max_wavelength_multiplier * ref_wavelength, 21, dtype=np.float64
    )

    # freq = 20  # Frequency (Hz)
    # z_src = 1000  # Source depth (m)
    # clims = (-110, -60)

    # dr_model = 5  # Model range in m
    dr_exact = 5.0  # Base range in m
    r_max = 3e3  # Maximum range in m

    z = build_depth_grid(zmax=100, dz=1)
    ssp = 1500 * np.ones_like(z)  # Constant sound speed profile (m/s)
    # z = build_depth_grid(zmax=5000, dz=10)
    # ssp = munk_profile(z)

    press_exact, r_exact = exact_parabolic_equation(
        freq, z, z_src, r_max, dr_exact, ssp
    )

    # get the variables at z_plt
    zind = np.argmin(z - z_plt)
    tl_exact, phase_exact = tl_and_phase(press_exact[zind, :])

    # initialize RMSE arrays
    tl_rmse_fwd_euler = np.zeros_like(dr_range, dtype=float)
    phase_rmse_fwd_euler = np.zeros_like(dr_range, dtype=float)
    tl_rmse_crank_nicholson = np.zeros_like(dr_range, dtype=float)
    phase_rmse_crank_nicholson = np.zeros_like(dr_range, dtype=float)

    tl_rmse_krylov = np.zeros((len(m_range), len(dr_range)), dtype=float)
    phase_rmse_krylov = np.zeros((len(m_range), len(dr_range)), dtype=float)
    for i, m in enumerate(m_range):
        for j, dr_model in enumerate(dr_range):
            # plot the phase and magnitude error at z_plt compared to the exact solution
            press_fwd_euler, r_euler = simple_parabolic_equation(
                freq, z, z_src, r_max, dr_model, ssp, "backward_euler"
            )
            press_crank_nicholson, r_cn = simple_parabolic_equation(
                freq, z, z_src, r_max, dr_model, ssp, "crank_nicholson"
            )
            # get the variables at z_plt
            zind = np.argmin(z - z_plt)
            tl_exact, phase_exact = tl_and_phase(press_exact[zind, :])
            tl_fwd_euler, phase_fwd_euler = tl_and_phase(press_fwd_euler[zind, :])
            tl_crank_nicholson, phase_crank_nicholson = tl_and_phase(
                press_crank_nicholson[zind, :]
            )

            # interpolate the exact solution to the model range
            tl_exact_interp = np.interp(r_euler, r_exact, tl_exact)
            phase_exact_interp = np.interp(r_euler, r_exact, phase_exact)

            # calculate the RMSE for each method
            tl_rmse_fwd_euler[j] = np.sqrt(
                np.mean((tl_exact_interp - tl_fwd_euler) ** 2)
            )
            tl_rmse_crank_nicholson[j] = np.sqrt(
                np.mean((tl_exact_interp - tl_crank_nicholson) ** 2)
            )

            phase_rmse_fwd_euler[j] = np.sqrt(
                np.mean((phase_exact_interp - phase_fwd_euler) ** 2)
            )
            phase_rmse_crank_nicholson[j] = np.sqrt(
                np.mean((phase_exact_interp - phase_crank_nicholson) ** 2)
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
            tl_krylov, phase_krylov = tl_and_phase(press_krylov[zind, :])
            tl_rmse_krylov[i, j] = np.sqrt(np.mean((tl_exact_interp - tl_krylov) ** 2))
            phase_rmse_krylov[i, j] = np.sqrt(
                np.mean((phase_exact_interp - phase_krylov) ** 2)
            )

    fig = plt.figure(figsize=(10, 6))
    # plot magnitude RMSE
    ax1 = fig.add_subplot(1, 1, 1)
    cf = ax1.pcolormesh(
        dr_range / ref_wavelength,
        m_range,
        tl_rmse_krylov,
        shading="auto",
        cmap="jet",
    )
    plt.colorbar(cf, ax=ax1, label="dB")
    ax1.set_xlabel(r"$\Delta r/ \lambda_0 (m)$")
    ax1.set_ylabel("Krylov subspace dimension (m)")
    ax1.set_title(
        f"TL RMSE at z = {z_plt:.1f} m "
        + r"($\Delta r_{exact} / \lambda_0)="
        + f"{dr_exact / ref_wavelength:.3f}$"
    )

    # plot the phase RMSE
    # ax2 = fig.add_subplot(2, 1, 2)
    # ax2.plot(
    #     [m_range[0], m_range[-1]],
    #     [phase_rmse_fwd_euler, phase_rmse_fwd_euler],
    #     "r--",
    #     label="Fwd Euler",
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
    # ax2.grid()
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    main()
