from __future__ import annotations

import matplotlib.pyplot as plt

from spinecho_sim.molecule import diatomic_hamiltonian_dicke
from spinecho_sim.molecule.hamiltonian_dicke import (
    collective_ops_sparse,
    single_spin_ops_sparse,
)
from spinecho_sim.util import plot_complex_heatmap, to_array

if __name__ == "__main__":
    result = single_spin_ops_sparse(1.0)
    fig, ax = plot_complex_heatmap(to_array(result[0]))
    ax.set_title(r"$J_x$, $J=1$")
    fig, ax = plot_complex_heatmap(to_array(result[1]))
    ax.set_title(r"$J_y$, $J=1$")
    fig, ax = plot_complex_heatmap(to_array(result[2]))
    ax.set_title(r"$J_z$, $J=1$")

    i_ops, j_ops = collective_ops_sparse(i=1.0, j=1.0)
    i_x, i_y, i_z, j_x, j_y, j_z = (*i_ops, *j_ops)  # unpack
    fig, ax = plot_complex_heatmap(to_array(i_x))
    ax.set_title(r"$I_x \otimes 1$, $I=1$, $J=1$")
    fig, ax = plot_complex_heatmap(to_array(i_y))
    ax.set_title(r"$I_y \otimes 1$, $I=1$, $J=1$")
    fig, ax = plot_complex_heatmap(to_array(i_z))
    ax.set_title(r"$I_z \otimes 1$, $I=1$, $J=1$")
    fig, ax = plot_complex_heatmap(to_array(j_x))
    ax.set_title(r"$1 \otimes J_x$, $I=1$, $J=1$")
    fig, ax = plot_complex_heatmap(to_array(j_y))
    ax.set_title(r"$1 \otimes J_y$, $I=1$, $J=1$")
    fig, ax = plot_complex_heatmap(to_array(j_z))
    ax.set_title(r"$1 \otimes J_z$, $I=1$, $J=1$")

    # Example usage:
    H = 300
    a, b, c, d = 4.258 * H, 0.6717 * H, 113.8, 57.68
    full_result_sparse = diatomic_hamiltonian_dicke(
        1, 1, coefficients=(a, b, c, d), b_vec=(0.2, 0.2, 1.0)
    )
    full_result_array = to_array(full_result_sparse)
    fig, ax = plot_complex_heatmap(full_result_array)
    ax.set_title(
        r"$H$, $I=1$, $J=1$, "
        f"{full_result_sparse.nnz} nnz elements, "
        f"{full_result_sparse.nnz / (full_result_array.size):.1%}"
    )
    plt.show()
