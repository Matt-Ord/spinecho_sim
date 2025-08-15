from __future__ import annotations

import matplotlib.pyplot as plt

from spinecho_sim.molecule import diatomic_hamiltonian_majorana
from spinecho_sim.molecule.hamiltonian_majorana import (
    quadrupole_block_majorana,
    spin_rotational_block_majorana,
    zeeman_hamiltonian_majorana,
)
from spinecho_sim.util import plot_complex_heatmap, to_array

if __name__ == "__main__":
    H = 300
    a, b, c, d = 4.258 * H, 0.6717 * H, 113.8, 57.68
    test_result = to_array(
        zeeman_hamiltonian_majorana(n_i=2, n_j=2, a=a, b=b, b_vec=(0.2, 0.2, 1.0))
    )
    fig, ax = plot_complex_heatmap(test_result)
    ax.set_title(r"Linear Field Terms $-a\mathbf{I \cdot H} - b\mathbf{J \cdot H}$")

    test_result = to_array(spin_rotational_block_majorana(n_i=2, n_j=2, c=c))
    fig, ax = plot_complex_heatmap(test_result)
    ax.set_title(r"Spin Rotational Term $-c\mathbf{I \cdot J}$")

    test_result = to_array(quadrupole_block_majorana(n_i=2, n_j=2, d=d))
    fig, ax = plot_complex_heatmap(test_result)
    ax.set_title(
        r"Quadrupole Term $+\frac{5d}{(2J-1)(2J+3)} [3(\mathbf{I \cdot J})^2 + \frac{3}{2} \mathbf{I \cdot J} -\mathbf{I}^2 \mathbf{J}^2]$"
    )

    test_result = diatomic_hamiltonian_majorana(
        n_i=2, n_j=2, coefficients=(a, b, c, d), b_vec=(0.2, 0.2, 1.0)
    )
    test_result_array = to_array(test_result)
    print(test_result.nnz)
    print(
        "sparsity:",
        f"{test_result.nnz} nnz elements,",
        f"{test_result.nnz / test_result_array.size:.1%}",
    )
    fig, ax = plot_complex_heatmap(test_result_array)

    plt.show()
