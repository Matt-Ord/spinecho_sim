from __future__ import annotations

from functools import reduce

import numpy as np
import scipy.sparse as sp  # type: ignore[import]

from spinecho_sim.util import (
    csr_add,
    csr_scale,
    csr_subtract,
    kronecker_n,
    sparse_matmul,
)

sigma_x = sp.csr_matrix([[0, 1], [1, 0]])
sigma_y = sp.csr_matrix([[0, -1j], [1j, 0]])
sigma_z = sp.csr_matrix([[1, 0], [0, -1]])
identity = sp.csr_matrix([[1, 0], [0, 1]])


def zeeman_hamiltonian_majorana(
    *,
    n_i: int,
    n_j: int,
    a: float,
    b: float,
    b_vec: tuple[float, float, float],
) -> sp.csr_matrix:
    """Construct the Zeeman Hamiltonian for two spin systems."""
    dim = 2 ** (n_i + n_j)
    hamiltonian = sp.csr_matrix((dim, dim), dtype=complex)
    b_mag = np.linalg.norm(b_vec)

    # Terms for I subsystem
    for i_index in range(n_i):
        operator_list = [identity] * (n_i + n_j)
        for component, operator_i in zip(
            b_vec, (sigma_x, sigma_y, sigma_z), strict=True
        ):
            operator_list[i_index] = operator_i
            hamiltonian = csr_add(
                hamiltonian,
                csr_scale(kronecker_n(operator_list), complex(-a * component / b_mag)),
            )
            operator_list[i_index] = identity

    # Terms for J subsystem
    for j_index in range(n_i, n_i + n_j):
        operator_list = [identity] * (n_i + n_j)
        for component, operator_j in zip(
            b_vec, (sigma_x, sigma_y, sigma_z), strict=True
        ):
            operator_list[j_index] = operator_j
            hamiltonian = csr_add(
                hamiltonian,
                csr_scale(kronecker_n(operator_list), complex(-b * component / b_mag)),
            )
            operator_list[j_index] = identity

    hamiltonian.eliminate_zeros()
    return hamiltonian


def spin_rotational_block_majorana(*, n_i: int, n_j: int, c: float) -> sp.csr_matrix:
    """Construct the spin rotational block for two spin systems."""
    hamiltonian = sp.csr_matrix((2 ** (n_i + n_j),) * 2, dtype=complex)
    operator_list = [identity] * (n_i + n_j)
    for i_index in range(n_i):
        for j_index in range(n_i, n_i + n_j):
            for operator_i, operator_j in zip(
                (sigma_x, sigma_y, sigma_z), (sigma_x, sigma_y, sigma_z), strict=True
            ):
                operator_list[i_index] = operator_i
                operator_list[j_index] = operator_j
                hamiltonian = csr_add(
                    hamiltonian, csr_scale(kronecker_n(operator_list), -c)
                )
                operator_list[i_index] = identity
                operator_list[j_index] = identity
    hamiltonian.eliminate_zeros()
    return hamiltonian


def collective_ij_majorana(
    n_i: int, n_j: int
) -> tuple[list[sp.csr_matrix], list[sp.csr_matrix]]:
    """Return the collective spin operators for two spin systems."""
    dim = 2 ** (n_i + n_j)

    def i_alpha_component(operator: sp.csr_matrix) -> sp.csr_matrix:
        operator_sum = sp.csr_matrix((dim, dim), dtype=complex)
        for i_index in range(n_i):
            operator_list = [identity] * (n_i + n_j)
            operator_list[i_index] = operator
            operator_sum = csr_add(operator_sum, kronecker_n(operator_list))
        return operator_sum

    i_alpha = [i_alpha_component(operator) for operator in (sigma_x, sigma_y, sigma_z)]

    def j_alpha_component(operator: sp.csr_matrix) -> sp.csr_matrix:
        operator_sum = sp.csr_matrix((dim, dim), dtype=complex)
        for j_index in range(n_j):
            operator_list = [identity] * (n_i + n_j)
            operator_list[n_i + j_index] = operator
            operator_sum = csr_add(operator_sum, kronecker_n(operator_list))
        return operator_sum

    j_alpha = [j_alpha_component(operator) for operator in (sigma_x, sigma_y, sigma_z)]

    return i_alpha, j_alpha


def quadrupole_block_majorana(n_i: int, n_j: int, d: float) -> sp.csr_matrix:
    """Return the d-term CSR matrix for given (N_I,N_J)."""
    i_alpha, j_alpha = collective_ij_majorana(n_i, n_j)

    # I⋅J -------------------------------------------------------------------
    i_dot_j = reduce(csr_add, map(sparse_matmul, i_alpha, j_alpha))

    # (I⋅J)^2 ---------------------------------------------------------------
    ij_sq = sparse_matmul(i_dot_j, i_dot_j)

    # I² and J² -----------------------------------------------------------
    i_sq = reduce(csr_add, map(sparse_matmul, i_alpha, i_alpha))
    j_sq = reduce(csr_add, map(sparse_matmul, j_alpha, j_alpha))

    hamiltonian = csr_scale(
        csr_subtract(
            csr_add(csr_scale(ij_sq, 3), csr_scale(i_dot_j, 1.5)),
            sparse_matmul(i_sq, j_sq),
        ),
        (5 * d / ((n_j - 1) * (n_j + 3))),
    )
    hamiltonian.eliminate_zeros()  # keep it neat
    return hamiltonian


def diatomic_hamiltonian_majorana(
    n_i: int,
    n_j: int,
    coefficients: tuple[float, float, float, float],
    b_vec: tuple[float, float, float],
) -> sp.csr_matrix:
    """Construct the diatomic Hamiltonian for two spin systems."""
    a, b, c, d = coefficients
    zeeman = zeeman_hamiltonian_majorana(n_i=n_i, n_j=n_j, a=a, b=b, b_vec=b_vec)
    rotational = spin_rotational_block_majorana(n_i=n_i, n_j=n_j, c=c)
    quadrupole = quadrupole_block_majorana(n_i=n_i, n_j=n_j, d=d)
    return csr_add(csr_add(zeeman, rotational), quadrupole)
