from __future__ import annotations

from functools import cache, reduce

import numpy as np
import scipy.sparse as sp  # type: ignore[import]

from spinecho_sim.util import (
    csr_add,
    csr_diags,
    csr_eye,
    csr_hermitian,
    csr_kron,
    csr_scale,
    csr_subtract,
    sparse_matmul,
)


@cache
def single_spin_ops_sparse(
    s: float,
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Generate sparse matrices for single spin operators Sx, Sy, Sz."""
    dim = int(2 * s + 1)
    mj_values = np.arange(s, -s - 1, -1)
    j_z = csr_diags(mj_values, 0)

    # ladder matrices
    j_plus_lil = sp.lil_matrix((dim, dim), dtype=complex)
    for m in range(dim - 1):
        j_plus_lil[m, m + 1] = np.sqrt(
            s * (s + 1) - mj_values[m + 1] * (mj_values[m + 1] - 1)
        )
    j_plus = sp.csr_matrix(j_plus_lil.tocsr())
    j_minus = csr_hermitian(j_plus)

    j_x = csr_scale(csr_add(j_plus, j_minus), 0.5)
    j_y = csr_scale(csr_subtract(j_plus, j_minus), -0.5j)
    return j_x, j_y, j_z


@cache
def collective_ops_sparse(
    i: float, j: float
) -> tuple[list[sp.csr_matrix], list[sp.csr_matrix]]:
    """Generate sparse matrices for single spin operators acting on the space of two spins."""
    i_x, i_y, i_z = single_spin_ops_sparse(i)
    j_x, j_y, j_z = single_spin_ops_sparse(j)

    dim_i, dim_j = 2 * i + 1, 2 * j + 1
    identity_i = csr_eye(dim_i)
    identity_j = csr_eye(dim_j)

    operators_i = [csr_kron(op, identity_j) for op in (i_x, i_y, i_z)]
    operators_j = [csr_kron(identity_i, op) for op in (j_x, j_y, j_z)]
    return operators_i, operators_j


def zeeman_hamiltonian_dicke(
    ops: list[sp.csr_matrix], b_vec: tuple[float, float, float]
) -> sp.csr_matrix:
    """Generate and cache the Zeeman Hamiltonian for two spin systems."""
    x, y, z = ops
    return csr_add(
        csr_add(csr_scale(x, b_vec[0]), csr_scale(y, b_vec[1])),
        csr_scale(z, b_vec[2]),
    )


@cache
def spin_rotation_hamiltonian_dicke(i: float, j: float) -> sp.csr_matrix:
    """Generate the spin-rotation Hamiltonian for two spin systems."""
    i_ops, j_ops = collective_ops_sparse(i, j)
    return reduce(csr_add, map(sparse_matmul, i_ops, j_ops))


@cache
def quadrupole_hamiltonian_dicke(i: float, j: float) -> sp.csr_matrix:
    """Generate the quadrupole Hamiltonian for two spin systems."""
    i_ops, j_ops = collective_ops_sparse(i, j)
    i_dot_j = spin_rotation_hamiltonian_dicke(i, j)
    ij_sq = sparse_matmul(i_dot_j, i_dot_j)
    i_sq = reduce(csr_add, map(sparse_matmul, i_ops, i_ops))
    j_sq = reduce(csr_add, map(sparse_matmul, j_ops, j_ops))
    return csr_subtract(
        csr_add(csr_scale(ij_sq, 3), csr_scale(i_dot_j, 1.5)),
        sparse_matmul(i_sq, j_sq),
    )


@cache
def cache_terms_hamiltonian_dicke(
    i: float, j: float, c: float, d: float
) -> sp.csr_matrix:
    """Generate the cache terms Hamiltonian for two spin systems."""
    # 2) spin-rotation
    hamiltonian_spin_rotation = csr_scale(spin_rotation_hamiltonian_dicke(i, j), -c)

    # 3) quadrupole / spin-spin
    hamiltonian_quadrupole = csr_scale(
        quadrupole_hamiltonian_dicke(i, j),
        (5 * d / ((2 * j - 1) * (2 * j + 3))),
    )
    return csr_add(hamiltonian_spin_rotation, hamiltonian_quadrupole)


def diatomic_hamiltonian_dicke(
    i: float,
    j: float,
    coefficients: tuple[float, float, float, float],
    b_vec: tuple[float, float, float],
) -> sp.csr_matrix:
    """Generate the Ramsey Hamiltonian as a sparse matrix."""
    a, b, c, d = coefficients
    # Generate spin operators
    i_ops, j_ops = collective_ops_sparse(i, j)

    # Linear Zeeman terms
    hamiltonian_i = csr_scale(zeeman_hamiltonian_dicke(i_ops, b_vec), a)
    hamiltonian_j = csr_scale(zeeman_hamiltonian_dicke(j_ops, b_vec), b)

    return csr_add(
        csr_add(hamiltonian_i, hamiltonian_j), cache_terms_hamiltonian_dicke(i, j, c, d)
    )
