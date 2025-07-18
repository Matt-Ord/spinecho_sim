from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from spinecho_sim import (
    Solenoid,
)
from spinecho_sim.majorana import simulate_trajectories_majorana
from spinecho_sim.state import (
    sample_gaussian_velocities,
    sample_uniform_displacement,
)

if __name__ == "__main__":
    particle_velocity = 714
    num_spins = 4
    velocities = sample_gaussian_velocities(
        num_spins, particle_velocity, 0.225 * particle_velocity
    )
    displacements = sample_uniform_displacement(num_spins, 1.16e-3)
    spinor_states = np.array(
        [
            [(1.0 + 0.0j) / np.sqrt(2), (0.0 + 0.0j), (1.0 + 0.0j) / np.sqrt(2)],
            [(0.0 + 0.0j), (1.0 + 0.0j) / np.sqrt(2), (1.0 + 0.0j) / np.sqrt(2)],
            [(-1.0 + 0.0j) / np.sqrt(2), (0.0 + 0.0j), (1.0 + 0.0j) / np.sqrt(2)],
            [(0.0 + 0.0j), (-1.0 + 0.0j) / np.sqrt(2), (1.0 + 0.0j) / np.sqrt(2)],
        ],
    )
    # spinor_states = np.array(
    #     [
    #         [(1.0 + 0.0j) / np.sqrt(2), (1.0 + 0.0j) / np.sqrt(2)],
    #         [(1.0 + 0.0j) / np.sqrt(2), (-1.0 + 0.0j) / np.sqrt(2)],
    #         [(1.0 + 0.0j), (0.00 + 0.0j)],
    #     ],
    # )
    solenoid = Solenoid.from_experimental_parameters(
        length=0.75,
        magnetic_constant=3.96e-3,
        current=0.1,
    )
    reconstructed_states = simulate_trajectories_majorana(
        spinor_states,
        displacements=displacements,
        velocities=velocities,
        solenoid=solenoid,
        n_steps=1000,
    )
    print("reconstructed_states:\n", reconstructed_states)
    n = 3

    # reconstructed_states shape: (n_states, n_steps, 2J+1)
    # We'll plot the real part of each coefficient as a "component"
    n_states, n_steps, n_components = reconstructed_states.shape
    time = np.linspace(0, solenoid.length, n_steps)

    fig, axes = plt.subplots(n, 1, figsize=(10, 8), sharex=True)
    component_labels = [f"Component {i}" for i in range(n)]
    for i in range(n):
        ax = axes[i]
        ax.set_ylim(0, 1)
        for state_idx in range(n_states):
            ax.plot(
                time,
                np.real(reconstructed_states[state_idx, :, i]),
                label=f"State {state_idx + 1}",
            )
        ax.set_ylabel(component_labels[i])
        ax.legend()
        ax.grid(visible=True)

    axes[-1].set_xlabel("Position along solenoid (m)")
    fig.suptitle("Quantum State Components Over Time")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()
