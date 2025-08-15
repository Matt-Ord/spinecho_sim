from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from spinecho_sim import (
    MonatomicParticleState,
)
from spinecho_sim.solenoid import (
    MonatomicSolenoid,
    plot_expectation_angles,
    plot_expectation_values,
    plot_spin_states,
)
from spinecho_sim.state import (
    CoherentSpin,
    sample_gaussian_velocities,
    sample_uniform_displacement,
)

if __name__ == "__main__":
    particle_velocity = 714
    num_spins = 10
    initial_states = [
        MonatomicParticleState(
            _spin_angular_momentum=CoherentSpin(theta=np.pi / 2, phi=0).as_generic(
                n_stars=1
            ),
            displacement=displacement,
            parallel_velocity=velocity,
            gyromagnetic_ratio=-2.04e8,
        )
        for displacement, velocity in zip(
            sample_uniform_displacement(num_spins, 1.16e-3),
            sample_gaussian_velocities(
                num_spins, particle_velocity, 0.225 * particle_velocity
            ),
            strict=True,
        )
    ]

    solenoid = MonatomicSolenoid.from_experimental_parameters(
        length=0.75,
        magnetic_constant=3.96e-3,
        current=0.1,
    )
    result = solenoid.simulate_trajectories(initial_states, n_steps=1000)

    n_stars = result.spin.n_stars
    S = n_stars / 2
    S_label = f"{S:.0f}" if S is int else f"{S:.1f}"

    fig, ax = plot_spin_states(result)
    fig.suptitle(
        r"Classical Larmor Precession of ${}^3$He in a Sinusoidal Magnetic Field, "
        r"$\mathbf{{B}} \approx B_0 \mathbf{z}$, "
        f"{num_spins} spins, $S={S_label}$",
    )
    output_path = (
        f"./examples/classical_solenoid.state.{num_spins}-spins_S-{S_label}.pdf"
    )
    plt.savefig(output_path, dpi=600, bbox_inches="tight")

    fig, ax = plot_expectation_values(result)
    fig.suptitle(
        r"Classical Larmor Precession of ${}^3$He in a Sinusoidal Magnetic Field, "
        r"$\mathbf{{B}} \approx B_0 \mathbf{z}$, "
        f"{num_spins} spins, $S={S_label}$",
    )
    output_path = (
        f"./examples/classical_solenoid.expectation.{num_spins}-spins_S-{S_label}.pdf"
    )
    plt.savefig(output_path, dpi=600, bbox_inches="tight")

    fig, ax = plot_expectation_angles(result)
    fig.suptitle(
        r"Classical Larmor Precession of ${}^3$He in a Sinusoidal Magnetic Field, "
        r"$\mathbf{{B}} \approx B_0 \mathbf{z}$, "
        f"{num_spins} spins, $S={S_label}$",
    )
    output_path = (
        f"./examples/classical_solenoid.angles.{num_spins}-spins_S-{S_label}.pdf"
    )
    plt.savefig(output_path, dpi=600, bbox_inches="tight")

    plt.show()
