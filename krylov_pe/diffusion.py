"""Module to use for 1D heat equation validation cases."""

import numpy as np
import numpy.typing as npt
from scipy.sparse import diags


def create_1d_diffusion_matrix(
    n: int, dx: float, alpha: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """Create a 1D diffusion matrix for the heat equation.

    Assumes that x is discretized on a uniform grid with spacing dx = 1/(n-1)
    with Dirichlet boundary conditions at x=0 and x=1.

    Parameters
    ----------
    n : int
        Number of spatial points.
    dx : float
        Spatial discretization step size.
    alpha : float
        Diffusion coefficient.

    Returns
    -------
    diff_mat: npt.NDArray
        The diffusion matrix.
    idx: npt.NDArray
        The indices of the interior points.
    """
    n_interior = n - 2  # Number of interior points
    diag = -2 * alpha / (dx**2) * np.ones(n_interior)
    off_diag = alpha / (dx**2) * np.ones(n_interior - 1)
    diff_mat = diags([off_diag, diag, off_diag], [-1, 0, 1]).toarray()  # type: ignore[no-untyped-call]

    # indices for the interior points
    idx = np.arange(1, n - 1)

    return diff_mat, idx


def analytic_1d_diffusion_solution(
    x: npt.NDArray,
    t: float,
    alpha: float,
) -> npt.NDArray:
    """Compute the analytic solution of the 1D diffusion equation.

    Analytic solution for 1D diffusion equation with initial condition
    u(x,0) = sin(pi*x) + 0.5*sin(3*pi*x) + 0.2*sin(5*pi*x).

    Assumes dirchilet boundary conditions at x=0 and x=1.
    Parameters
    ----------
    x : npt.NDArray
        Spatial points.
    t : float
        Time point.
    alpha : float
        Diffusion coefficient.
    initial_condition : npt.NDArray | None, optional
        Initial condition. Default is None, in which case a default intitial condition of u(x,0) = sin(pi*x) is used.

    Returns
    -------
    npt.NDArray
        The analytic solution at time t.
    """
    return (
        np.exp(-(np.pi**2) * alpha * t) * np.sin(np.pi * x)
        + 0.5 * np.exp(-((3 * np.pi) ** 2) * alpha * t) * np.sin(3 * np.pi * x)
        + 0.2 * np.exp(-((5 * np.pi) ** 2) * alpha * t) * np.sin(5 * np.pi * x)
    )
