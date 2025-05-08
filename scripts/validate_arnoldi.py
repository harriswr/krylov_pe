from krylov_pe.krylov import arnoldi_iteration

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals, eigh, eig, norm
import scipy.sparse.linalg as sp_la


def main():
    """Validate the Arnoldi iteration."""
    n = 50
    A = np.random.randn(n, n)
    A = A @ A.T  # Make symmetric for simplicity
    init_vector = np.random.randn(n)
    num_iterations = 40

    basis_mat, hessen_mat = arnoldi_iteration(A, init_vector, num_iterations)
    approx_eigenvalues = eigvals(hessen_mat)

    _, u = eigh(hessen_mat)

    exact_eigenvalues, exact_eigenvectors = eig(A)
    # exact_eigenvalues, exact_eigenvectors = sp_la.eigs(A, k=num_iterations, which="LM")

    # Sort eigenvalues and eigenvectors by magnitude
    approx_eigenvalues = np.real(approx_eigenvalues)  # Ensure real for symmetric matrix
    exact_eigenvalues = np.real(exact_eigenvalues)
    pairing = np.argmin(
        np.abs(approx_eigenvalues[:, None] - exact_eigenvalues[None, :]), axis=1
    )
    exact_eigenvalues = exact_eigenvalues[pairing]
    exact_eigenvectors = exact_eigenvectors[:, pairing]
    u = u[:, pairing]  # Reorder approximate eigenvectors
    approx_eigenvectors = basis_mat @ u

    # compute error
    eigenvalue_errors = np.abs(approx_eigenvalues - exact_eigenvalues)

    # Compare eigenvectors
    eigenvector_errors = []
    cosine_similarities = []
    residuals = []
    for i in range(num_iterations):
        # Normalize eigenvectors
        approx_vec = approx_eigenvectors[:, i] / norm(approx_eigenvectors[:, i])
        exact_vec = exact_eigenvectors[:, i] / norm(exact_eigenvectors[:, i])
        # Account for sign ambiguity
        error = min(norm(approx_vec - exact_vec), norm(approx_vec + exact_vec))
        eigenvector_errors.append(error)
        # Compute cosine similarity
        cosine_sim = np.abs(np.dot(approx_vec, exact_vec))
        cosine_similarities.append(cosine_sim)
        # Compute residual norm
        lambda_i = approx_eigenvalues[i]
        residual = norm(A @ approx_vec - lambda_i * approx_vec)
        residuals.append(residual)
        print(
            f"Eigenvector {i}: Cosine Similarity = {cosine_sim:.6f}, Residual = {residual:.6e}"
        )

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(approx_eigenvalues, "ko", markersize=5, label="Approx Eigenvalues")
    ax.plot(exact_eigenvalues, "ro", markersize=2, label="Exact Eigenvalues")
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")
    ax.legend()
    ax.grid()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(eigenvalue_errors, "ko")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Eigenvalue Error")
    ax2.set_yscale("log")
    ax2.grid()

    plt.tight_layout()
    plt.show(block=True)
    print("Eigenvalue Errors:", eigenvalue_errors)
    print("Eigenvector Errors:", eigenvector_errors)

    # Verify orthonormality of basis_mat
    orthonormality_error = norm(basis_mat.T @ basis_mat - np.eye(num_iterations))
    print("Orthonormality Error of basis_mat:", orthonormality_error)


if __name__ == "__main__":
    main()
