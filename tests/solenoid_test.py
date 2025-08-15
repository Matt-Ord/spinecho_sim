from __future__ import annotations

from itertools import starmap
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp  # type: ignore[import-untyped]

from spinecho_sim.solenoid import (
    MonatomicSolenoid,
    MonatomicSolenoidTrajectory,
    Solenoid,
    SolenoidTrajectory,
)
from spinecho_sim.state import (
    CoherentSpin,
    MonatomicParticleState,
    MonatomicTrajectory,
    ParticleDisplacement,
    Spin,
    sample_gaussian_velocities,
    sample_uniform_displacement,
)


def _get_field(
    z: float,
    displacement: ParticleDisplacement,
    solenoid: Solenoid,
    dz: float = 1e-5,
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    if displacement.r == 0:
        return solenoid.field(z)

    # Assuming that there is no current in the solenoid, we can
    # calculate the field at any point using grad.B = 0. We do this
    b_z_values = [solenoid.field(zi)[2] for zi in (z - dz, z, z + dz)]

    b0_p = (b_z_values[1] - b_z_values[-1]) / (2 * dz)
    b0_pp = (b_z_values[2] - 2 * b_z_values[1] + b_z_values[0]) / (dz**2)

    b_r = -0.5 * displacement.r * b0_p
    db_z = -0.25 * displacement.r**2 * b0_pp

    return np.array(
        [
            b_r * np.cos(displacement.theta),
            b_r * np.sin(displacement.theta),
            b_z_values[1] + db_z,
        ]
    )


def simulate_trajectory_cartesian(
    solenoid: Solenoid,
    initial_state: MonatomicParticleState,
    n_steps: int = 100,
) -> SolenoidTrajectory:
    """Run the spin echo simulation using configured parameters."""
    z_points = np.linspace(0, solenoid.length, n_steps + 1, endpoint=True)

    gyromagnetic_ratio = -2.04e8  # gyromagnetic ratio (rad s^-1 T^-1)

    def _ds_dx(
        z: float,
        spin: tuple[float, float, float],
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        field = _get_field(z, initial_state.displacement, solenoid)
        velocity = initial_state.parallel_velocity

        return (gyromagnetic_ratio / velocity) * np.cross(spin, field)

    sol = solve_ivp(  # type: ignore[return-value]
        fun=_ds_dx,
        t_span=(z_points[0], z_points[-1]),
        y0=initial_state.spin.item(0).cartesian,
        t_eval=z_points,
        vectorized=False,
        rtol=1e-8,
    )
    spins = Spin.from_iter(
        [x.as_generic() for x in starmap(CoherentSpin.from_cartesian, sol.y.T)]  # type: ignore[return-value]
    )
    return MonatomicSolenoidTrajectory(
        trajectory=MonatomicTrajectory(
            _spin_angular_momentum=spins,
            displacement=initial_state.displacement,
            parallel_velocity=initial_state.parallel_velocity,
        ),
        positions=z_points,
    )


def test_simulate_trajectory() -> None:
    particle_velocity = 714

    initial_state = MonatomicParticleState(
        _spin_angular_momentum=CoherentSpin(theta=np.pi / 2, phi=0).as_generic(),
        displacement=sample_uniform_displacement(1, 1.16e-3)[0],
        parallel_velocity=sample_gaussian_velocities(
            1, particle_velocity, 0.225 * particle_velocity
        )[0],
    )

    solenoid = MonatomicSolenoid.from_experimental_parameters(
        length=0.75,
        magnetic_constant=3.96e-3,
        current=0.01,
    )
    n_steps = 300
    result = solenoid.simulate_trajectory(initial_state, n_steps=n_steps)

    assert result.spin.cartesian.shape == (3, n_steps + 1, 1)

    expected = simulate_trajectory_cartesian(solenoid, initial_state, n_steps=n_steps)
    np.testing.assert_allclose(
        result.spin.cartesian,
        expected.spin.cartesian,
        atol=1e-4,
    )


def test_simulate_trajectories() -> None:
    particle_velocity = 714

    initial_state = MonatomicParticleState(
        _spin_angular_momentum=CoherentSpin(theta=np.pi / 2, phi=0).as_generic(
            n_stars=2
        ),
        displacement=sample_uniform_displacement(1, 1.16e-3)[0],
        parallel_velocity=sample_gaussian_velocities(
            1, particle_velocity, 0.225 * particle_velocity
        )[0],
    )
    solenoid = MonatomicSolenoid.from_experimental_parameters(
        length=0.75,
        magnetic_constant=3.96e-3,
        current=0.01,
    )
    result = solenoid.simulate_trajectories([initial_state], n_steps=300)
    expected = solenoid.simulate_trajectory(initial_state, n_steps=300)

    # Both theta and phi should be the same for all stars
    np.testing.assert_allclose(
        result.spin.theta[0, ..., 0],
        expected.spin.theta[..., 1],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        result.spin.phi[0, ..., 0],
        expected.spin.phi[..., 1],
        atol=1e-4,
    )


def test_simulate_trajectory_high_spin() -> None:
    particle_velocity = 714

    initial_state = MonatomicParticleState(
        _spin_angular_momentum=CoherentSpin(theta=np.pi / 2, phi=0).as_generic(
            n_stars=2
        ),
        displacement=sample_uniform_displacement(1, 1.16e-3)[0],
        parallel_velocity=sample_gaussian_velocities(
            1, particle_velocity, 0.225 * particle_velocity
        )[0],
    )
    solenoid = MonatomicSolenoid.from_experimental_parameters(
        length=0.75,
        magnetic_constant=3.96e-3,
        current=0.01,
    )
    result = solenoid.simulate_trajectory(initial_state, n_steps=300)

    # Both theta and phi should be the same for all stars
    np.testing.assert_allclose(
        result.spin.theta[0, ..., 0],
        result.spin.theta[1, ..., 1],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        result.spin.phi[0, ..., 0],
        result.spin.phi[1, ..., 1],
        atol=1e-4,
    )

    initial_state_1 = MonatomicParticleState(
        _spin_angular_momentum=CoherentSpin(theta=np.pi / 2, phi=0).as_generic(
            n_stars=1
        ),
        displacement=initial_state.displacement,
        parallel_velocity=initial_state.parallel_velocity,
    )
    result_1 = solenoid.simulate_trajectory(initial_state_1, n_steps=300)

    # Both theta and phi should be the same for all stars
    np.testing.assert_allclose(
        result_1.spin.theta[..., 0],
        result.spin.theta[..., 1],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        result_1.spin.phi[..., 0],
        result.spin.phi[..., 1],
        atol=1e-4,
    )
