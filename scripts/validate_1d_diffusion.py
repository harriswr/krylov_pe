"""Script to create validation against analytic solution for 1D diffusion equation."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm

from krylov_pe.krylov import krylov_expm_with_dirichlet
from krylov_pe.diffusion import (
    create_1d_diffusion_matrix,
    analytic_1d_diffusion_solution,
)


def main():
    """Validate the Krylov subspace method for 1D diffusion equation."""

    # Parameters
    n = 100  # Number of spatial points
    alpha = 0.01  # Diffusion coefficient
    t = 0.5  # Time step

    # create the grid and intitial conditions
    x = np.linspace(0, 1, n, dtype=np.float64)  # Spatial grid
    dx = x[1] - x[0]
    u0 = (
        np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x) + 0.2 * np.sin(5 * np.pi * x)
    )  # Initial condition

    # analytic solution
    u_analytic = analytic_1d_diffusion_solution(x, t, alpha)

    # Plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(3, 1, (1, 2))
    ax2 = fig.add_subplot(3, 1, 3)
    ax.plot(x, u_analytic, "ko", label="Analytic solution")

    for m in np.arange(1, 5, 1, dtype=int):  # Dimension of Krylov subspace
        # Create the diffusion matrix
        diff_mat, idx = create_1d_diffusion_matrix(n, dx, alpha)
        u_krylov = krylov_expm_with_dirichlet(diff_mat * t, idx, u0, m, n)

        ax.plot(x, u_krylov, label=f"Krylov solution (m={int(m)})")

        # error
        error = norm(u_krylov - u_analytic) / norm(u_analytic)
        ax2.plot(m, error, "ko")

    ax.set_xlabel("X")
    ax.set_ylabel("U")
    ax.legend()
    ax.grid()

    ax2.set_xlabel("Krylov subspace dimension")
    ax2.set_ylabel("Relative error")
    ax2.set_yscale("log")
    ax2.grid()
    fig.suptitle("1D Diffusion Equation Comparison")
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    main()
