from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from spinecho_sim.molecule.hamiltonian_dicke import (
    collective_ops_sparse,
    diatomic_hamiltonian_dicke,
)
from spinecho_sim.util import SolveIVPResult, solve_ivp_typed, sparse_apply, to_array

if TYPE_CHECKING:
    import scipy.sparse as sp  # pyright: ignore[reportMissingTypeStubs]


def evolve_quantum_state(
    hamiltonian: sp.csr_matrix,
    initial_state: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray | None = None,
) -> SolveIVPResult:
    """
    Evolve a quantum state using the Schrödinger equation.

    Args:
        hamiltonian: The Hamiltonian operator as a sparse matrix
        initial_state: Initial quantum state vector
        t_span: (t_start, t_end) tuple for time evolution
        t_eval: Optional specific time points to evaluate

    Returns
    -------
        Dictionary containing results of the ODE integration
    """
    # Convert initial_state to complex array and flatten
    psi0 = np.asarray(initial_state, dtype=complex).flatten()

    # Normalize the initial state
    psi0 /= np.linalg.norm(psi0)

    def schrodinger_eq(_t: float, psi: np.ndarray) -> np.ndarray:
        # Note: we set ħ = 1 for simplicity
        result = sparse_apply(hamiltonian, psi)
        # Convert back to numpy array for ODE solver compatibility
        return -1j * result

    # Solve the ODE
    return solve_ivp_typed(
        schrodinger_eq,
        t_span,
        psi0,
        t_eval=t_eval,
        method="RK45",  # Can use 'DOP853' for higher accuracy
    )


# Example usage:
if __name__ == "__main__":
    # Create Hamiltonian
    H = 300
    a, b, c, d = 4.258 * H, 0.6717 * H, 113.8, 57.68
    hamiltonian = diatomic_hamiltonian_dicke(
        1, 1, coefficients=(a, b, c, d), b_vec=(0.2, 0.2, 1.0)
    )

    # Dimension of the system
    dim = to_array(hamiltonian).shape[0]

    # Initial state: create a superposition state or specific state
    # For example, |00⟩ (ground state)
    initial_state = np.zeros(dim, dtype=complex)
    initial_state[0] = 1.0

    # Time points (in natural units)
    t_max = 0.1  # Adjust based on your energy scales
    t_points = np.linspace(0, t_max, 1000)

    # Solve Schrödinger equation
    result = evolve_quantum_state(
        hamiltonian, initial_state, (0, t_max), t_eval=t_points
    )

    # Analyze results
    if result.success:
        # Extract solutions at each time point
        psi_t = result.y
        t = result.t

        # Calculate observables (example: expectation values of i_z and j_z)
        i_ops, j_ops = collective_ops_sparse(i=1.0, j=1.0)
        i_z = i_ops[2]
        j_z = j_ops[2]

        i_z_expectation: list[np.floating] = []
        j_z_expectation: list[np.floating] = []

        for i in range(len(t)):
            psi = psi_t[:, i]

            # Calculate expectation values <ψ|O|ψ>
            i_z_exp = np.real(np.vdot(psi, sparse_apply(i_z, psi)))
            j_z_exp = np.real(np.vdot(psi, sparse_apply(j_z, psi)))

            i_z_expectation.append(i_z_exp)
            j_z_expectation.append(j_z_exp)

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(t, i_z_expectation, label=r"$\langle I_z \rangle$")
        plt.plot(t, j_z_expectation, label=r"$\langle J_z \rangle$")
        plt.xlabel("Time")
        plt.ylabel("Expectation Value")
        plt.title("Time Evolution of Spin Expectation Values")
        plt.legend()
        plt.grid(visible=True)
        plt.show()
    else:
        print("ODE solver failed:", result.message)  # noqa: T201
