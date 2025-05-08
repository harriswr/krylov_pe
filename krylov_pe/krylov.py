"""Module to house matrix exponential and Krylov subspace methods."""

from scipy.linalg import expm, norm
import numpy as np
import numpy.typing as npt


def arnoldi_iteration(
    input_mat: npt.NDArray,
    init_vector: npt.NDArray,
    num_iterations: int,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Perform m steps of Arnoldi iteration to construct the subspace.

    Parameters
    ----------
    input_mat : npt.ndarray
        The matrix for which we want to compute the Krylov subspace.
    init_vector : npt.ndarray
        The initial vector to start the Arnoldi iteration.
    num_iterations : int
        The number of steps of the Arnoldi iteration.

    Returns
    -------
    basis_mat : npt.ndarray
        The orthonormal basis of the Krylov subspace.
    hessen_mat : npt.ndarray
        The upper Hessenberg matrix that represents the action of A on the Krylov subspace.

    """
    if isinstance(input_mat, np.matrix):
        input_mat = np.asarray(input_mat)

    n = input_mat.shape[0]
    m = num_iterations
    basis_mat = np.zeros(
        (n, num_iterations), dtype=input_mat.dtype
    )  # Orthonormal basis of the Krylov subspace
    hessen_mat = np.zeros((m, m), dtype=input_mat.dtype)  # Upper Hessenberg matrix

    # normalize the initial vector
    v_norm = norm(init_vector)
    if v_norm < 1e-10:
        print("Warning: Initial vector norm is near zero in Arnoldi iteration")
    init_vector = init_vector / v_norm
    basis_mat[:, 0] = init_vector

    for j in range(m - 1):
        w = input_mat @ basis_mat[:, j]  # w = A * v_j

        # orthogonalize against the previous bases
        for i in range(j + 1):
            hessen_mat[i, j] = np.vdot(basis_mat[:, i], w)
            w -= hessen_mat[i, j] * basis_mat[:, i]

        # normalize and store
        hessen_mat[j + 1, j] = norm(w)
        if hessen_mat[j + 1, j] > 1e-10:
            basis_mat[:, j + 1] = w / hessen_mat[j + 1, j]
        else:
            return basis_mat, hessen_mat

    w = input_mat @ basis_mat[:, m - 1]  # w = A * v_m-1
    for i in range(m):
        hessen_mat[i, m - 1] = np.vdot(basis_mat[:, i], w)
        w -= hessen_mat[i, m - 1] * basis_mat[:, i]

    return basis_mat, hessen_mat


def krylov_expm(
    input_mat: npt.NDArray,
    init_vector: npt.NDArray,
    num_iterations: int,
) -> npt.NDArray:
    """Approximate the matrix exponential using the Krylov subspace.

    Parameters
    ----------
    input_mat : npt.ndarray
        The matrix for which we want to compute the Krylov subspace.
    init_vector : npt.ndarray
        The initial vector to start the Arnoldi iteration.
    num_iterations : int
        The number of steps of the Arnoldi iteration and also the dimension of the Krylov subspace.

    Returns
    -------
    new_vector : npt.ndarray
        The result of the matrix exponential applied to the initial vector.
        This is the approximation of expm(input_mat) @ init_vector.
    """

    # Perform Arnoldi iteration
    basis_mat, hessen_mat = arnoldi_iteration(input_mat, init_vector, num_iterations)

    # Compute the matrix exponential of the upper Hessenberg matrix
    beta = norm(init_vector)
    e1 = np.zeros(hessen_mat.shape[0])
    e1[0] = 1.0  # First basis vector
    expm_hessen = expm(hessen_mat)

    return beta * basis_mat @ (expm_hessen @ e1)


def krylov_expm_with_dirichlet(
    input_mat: npt.NDArray,
    idx: npt.NDArray,
    init_vector: npt.NDArray,
    num_iterations: int,
    total_number_of_points: int,
) -> npt.NDArray:
    # pull only the interior matrices
    next_mat = np.zeros(total_number_of_points)
    next_interior = krylov_expm(input_mat, init_vector[idx], num_iterations)
    next_mat[idx] = next_interior
    return next_mat
