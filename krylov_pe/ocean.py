"""Module for PE comparison."""

from matplotlib.axes import Axes
import numpy as np
import numpy.typing as npt
import scipy.linalg as linalg
from scipy.sparse import diags
from scipy.linalg import expm

from krylov_pe.krylov import krylov_expm


def munk_profile(z: np.ndarray) -> np.ndarray:
    """Munk profile for oceanographic applications.

    Parameters
    ----------
    z : np.ndarray
        Depth array.

    Returns
    -------
    np.ndarray
        Munk profile.

    Reference:
    W. H. Munk, "Sound channel in an exponentially stratified ocean with applications to SOFAR,"
    J. Acoust. Soc. Am. 55, 220--226 (1974).
    """
    eps = 0.00737
    z_bar = 2 * (z - 1300) / 1000
    return 1500 * (1 + eps * (z_bar - 1 + np.exp(-z_bar)))
    # eta = 2 * (z - 1000) / 1000  # Normalized depth
    # return 1500 * (1 + 0.0057 * (-1 + eta + np.exp(-eta)))  # Munk profile


def build_depth_grid(zmax: int = 5000, dz: int = 10) -> np.ndarray:
    """Build a depth grid for oceanographic applications.

    Returns
    -------
    np.ndarray
        Depth grid.
    """
    h = zmax
    return np.arange(0, h + dz, dz)  # Depth grid


def gaussian_starter(k0: float, z_src: float, z: npt.NDArray[np.float64]) -> np.ndarray:
    """Gaussian starter for oceanographic applications.

    Parameters
    ----------
    k0 : float
        Reference wavenumber (1/m).
    z_src : float
        Source depth (m).
    z : np.ndarray
        Depth array.

    Returns
    -------
    np.ndarray
        Gaussian starter.
    """
    return np.sqrt(k0) * np.exp(-(k0**2) / 2 * (z - z_src) ** 2)  # Gaussian starter


def make_finite_difference_matrix(
    z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Make finite difference matrix for 2nd derivative in z."""
    n = len(z)
    dz = z[1] - z[0]  # Depth step (m)

    diag = -2 * np.ones(n) / (dz**2)  # Diagonal
    off_diag = np.ones(n - 1) / (dz**2)  # Off-diagonal
    diff_mat = diags([off_diag, diag, off_diag], [-1, 0, 1]).toarray()  # type: ignore[no-untyped-call]
    diff_mat[-1, -1] = -1  # enforces the rigid boundary condition at the bottom
    return diff_mat  # Finite difference matrix


def propagator_backward_euler(
    reference_wavenumber: float,
    z: npt.NDArray,
    r: npt.NDArray,
    n_squared: npt.NDArray,
    diff_mat: npt.NDArray,
) -> np.ndarray:
    """Propagator for backward Euler method."""
    nz = len(z)
    dr = r[1] - r[0]

    propagator_matrix = np.eye(nz) - 1j / 2 * reference_wavenumber * dr * (
        diags(n_squared - 1) + diff_mat / reference_wavenumber**2
    )

    return linalg.inv(propagator_matrix)


def propagator_crank_nicholson(
    reference_wavenumber: float,
    z: npt.NDArray,
    r: npt.NDArray,
    n_squared: npt.NDArray,
    diff_mat: npt.NDArray,
) -> np.ndarray:
    """Propagator for Crank-Nicholson method."""
    nz = len(z)
    dr = r[1] - r[0]

    propagator_matrix_1 = np.eye(nz) - 1j / 4 * reference_wavenumber * dr * (
        diags(n_squared - 1) + diff_mat / reference_wavenumber**2
    )
    propagator_matrix_2 = np.eye(nz) + 1j / 4 * reference_wavenumber * dr * (
        diags(n_squared - 1) + diff_mat / reference_wavenumber**2
    )

    return linalg.inv(propagator_matrix_1) @ propagator_matrix_2


def simple_parabolic_equation(
    freq: float,
    z: npt.NDArray[np.float64],
    z_src: float,
    r_max: float,
    dr: float,
    ssp: npt.NDArray[np.float64],
    propagator: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple parabolic equation for oceanographic applications.

    Returns
    -------
    np.ndarray
        Simple parabolic equation.
    """
    # depth grid
    wavenumber = 2 * np.pi * freq / ssp

    # range grid
    r = np.arange(dr, r_max + dr, dr)  # Range grid

    # initialize the PE matrix
    psi = np.zeros((len(z), len(r)), dtype=np.complex128)  # PE matrix

    # reference values for PE step
    ref_ssp = 1500  # Reference sound speed (m/s)
    ref_wavenumber = 2 * np.pi * freq / ref_ssp  # Reference wavenumber (1/m)
    n_squared = (wavenumber / ref_wavenumber) ** 2

    # starter
    psi0 = gaussian_starter(ref_wavenumber, z_src, z)  # Gaussian starter

    # farfield bessel function
    d_sq = make_finite_difference_matrix(z)  # Finite difference matrix

    match propagator:
        case "backward_euler":
            pe_propagator = propagator_backward_euler(
                ref_wavenumber, z, r, n_squared, d_sq
            )
        case "crank_nicholson":
            pe_propagator = propagator_crank_nicholson(
                ref_wavenumber, z, r, n_squared, d_sq
            )
        case _:
            raise ValueError(
                "Invalid propagator type. Use 'backward_euler' or 'crank_nicholson'."
            )

    psi[:, 0] = pe_propagator @ psi0
    for i in np.arange(0, len(r) - 1, dtype=int):
        psi[:, i + 1] = pe_propagator @ psi[:, i]

    # apply bessel function for range decay
    hk = 1 / np.sqrt(r)
    press = psi * hk

    return press, r


def press_to_db(press: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
    """Convert pressure to dB"""
    return 20 * np.log10(np.abs(press))


def plot_pe_result(
    ax: Axes,
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    press: npt.NDArray[np.complex128],
    title: str,
    clims: tuple[float, float] = (-110, -60),
):
    """Plot the transmission loss field"""
    r_km = r / 1e3  # Convert range to km
    p_db = press_to_db(press)  # Convert pressure to dB

    ax.set_title(title)
    ax.set_xlabel("Range (km)")
    ax.set_ylabel("Depth (m)")
    ax.set_aspect("auto")
    ax.invert_yaxis()
    mesh = ax.pcolormesh(r_km, z, p_db, shading="auto", cmap="jet")
    mesh.set_clim(clims[0], clims[1])  # Set color limits for dB scale
    cbar = ax.get_figure().colorbar(mesh, ax=ax, pad=0.01, aspect=10)  # type: ignore
    cbar.set_label("Transmission Loss (dB)", rotation=270, labelpad=15)


def plot_pe_tl_at_z(
    ax: Axes,
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    press: npt.NDArray[np.complex128],
    z_plt: float,
    label: str,
    dr_plot: float = 100,
):
    """Plot line plot of transmission loss at a given depth."""
    dr = r[1] - r[0]  # Range step (m)
    r_skip_ind = int(dr_plot / dr)  # Skip every r_skip_ind points
    r_skip_ind = max(1, r_skip_ind)  # Ensure at least one point is plotted
    r = r[::r_skip_ind]  # Skip points for plotting

    r_km = r / 1e3  # Convert range to km
    p_db = press_to_db(press[:, ::r_skip_ind])

    zind = np.argmin(z - z_plt)  # Find the index of the source depth
    ax.plot(r_km, p_db[zind, :], label=f"{label}")
    ax.set_xlabel("Range (km)")
    ax.set_ylabel("Transmission Loss (dB)")
    ax.set_ylim(-110, -60)
    ax.legend()
    ax.grid()
    ax.set_title(f"Transmission Loss at {z_plt} m")


def tappert_sqrt(
    reference_wavenumber: float,
    diff_mat: npt.NDArray,
    n_squared: npt.NDArray,
) -> npt.NDArray:
    """Use tappert approximation for sqrt operator.


    Approximates sqrt(1+X) where X = 1/k_0^2 * d^2/dz^2 + n^2 - 1
    should go to 1 + 0.5 * X for small X
    """
    x = (1 / reference_wavenumber**2) * diff_mat + diags(n_squared - 1, 0)
    return np.array(0.5 * x)


def krylov_parabolic_equation(
    freq: float,
    z: npt.NDArray[np.float64],
    z_src: float,
    r_max: float,
    dr: float,
    ssp: npt.NDArray[np.float64],
    m: int,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
    """Compute the propagator using Krylov subspace method."""
    # depth grid
    wavenumber = 2 * np.pi * freq / ssp

    # range grid
    r = np.arange(dr, r_max + dr, dr)  # Range grid

    # initialize the PE matrix
    psi = np.zeros((len(z), len(r)), dtype=np.complex128)  # PE matrix

    # reference values for PE step
    ref_ssp = 1500  # Reference sound speed (m/s)
    ref_wavenumber = 2 * np.pi * freq / ref_ssp  # Reference wavenumber (1/m)
    n_squared = (wavenumber / ref_wavenumber) ** 2

    # starter
    psi0 = gaussian_starter(ref_wavenumber, z_src, z)  # Gaussian starter

    # farfield bessel function
    d_sq = make_finite_difference_matrix(z)  # Finite difference matrix

    q_approx = tappert_sqrt(ref_wavenumber, d_sq, n_squared)
    exponent = 1j * ref_wavenumber * dr * q_approx

    # ensure not writing as np.matrix
    if not isinstance(exponent, np.ndarray):
        exponent = np.asarray(exponent)

    psi[:, 0] = krylov_expm(exponent, psi0, m)
    for i in np.arange(0, len(r) - 1, dtype=int):
        psi[:, i + 1] = krylov_expm(exponent, psi[:, i], m)

    # apply bessel function for range decay
    hk = 1 / np.sqrt(r)
    press = psi * hk

    return press, r.astype(np.float64)  # Ensure r is float64 for consistency


def exact_parabolic_equation(
    freq: float,
    z: npt.NDArray[np.float64],
    z_src: float,
    r_max: float,
    dr: float,
    ssp: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
    """Compute the propagator using Scipy expm as exact method."""
    # depth grid
    wavenumber = 2 * np.pi * freq / ssp

    # range grid
    r = np.arange(dr, r_max + dr, dr)  # Range grid

    # initialize the PE matrix
    psi = np.zeros((len(z), len(r)), dtype=np.complex128)  # PE matrix

    # reference values for PE step
    ref_ssp = 1500  # Reference sound speed (m/s)
    ref_wavenumber = 2 * np.pi * freq / ref_ssp  # Reference wavenumber (1/m)
    n_squared = (wavenumber / ref_wavenumber) ** 2

    # starter
    psi0 = gaussian_starter(ref_wavenumber, z_src, z)  # Gaussian starter

    # farfield bessel function
    d_sq = make_finite_difference_matrix(z)  # Finite difference matrix

    q_approx = tappert_sqrt(ref_wavenumber, d_sq, n_squared)
    exponent = 1j * ref_wavenumber * dr * q_approx

    # ensure not writing as np.matrix
    if not isinstance(exponent, np.ndarray):
        exponent = np.asarray(exponent)

    psi[:, 0] = expm(exponent) @ psi0
    for i in np.arange(0, len(r) - 1, dtype=int):
        psi[:, i + 1] = expm(exponent) @ psi[:, i]

    # apply bessel function for range decay
    hk = 1 / np.sqrt(r)
    press = psi * hk

    return press, r.astype(np.float64)  # Ensure r is float64 for consistency
