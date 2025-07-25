"""Core simulation functionality for spin echo experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.integrate import solve_ivp  # type: ignore[import-untyped]
from tqdm import tqdm

from spinecho_sim.state import (
    ParticleDisplacement,
    ParticleDisplacementList,
    ParticleState,
    Spin,
    Trajectory,
    TrajectoryList,
)
from spinecho_sim.state._spin import expectation_values
from spinecho_sim.util import timed

if TYPE_CHECKING:
    from collections.abc import Callable

    from spinecho_sim.state._state import CoherentParticleState


@dataclass(kw_only=True, frozen=True)
class Solenoid:
    """Dataclass representing a solenoid with its parameters."""

    length: float
    field: Callable[[float], np.ndarray[Any, np.dtype[np.floating]]]

    @classmethod
    def with_uniform_z(cls, length: float, strength: float) -> Solenoid:
        """Build a solenoid with a uniform field along the z-axis."""
        return cls(length=length, field=lambda _z: np.array([0.0, 0.0, strength]))

    @classmethod
    def with_nonuniform_z(
        cls, length: float, strength: Callable[[float], float]
    ) -> Solenoid:
        """Build a solenoid with a non-uniform field along the z-axis."""
        return cls(length=length, field=lambda z: np.array([0.0, 0.0, strength(z)]))

    @classmethod
    def from_experimental_parameters(
        cls, *, length: float, magnetic_constant: float, current: float
    ) -> Solenoid:
        """Build a solenoid from an experimental magnetic constant and current."""
        b_z = np.pi * magnetic_constant * current / (2 * length)
        return cls.with_nonuniform_z(
            length=length,
            strength=lambda z: b_z * np.sin(np.pi * z / length) ** 2,
        )

    def _simulate_coherent_trajectory(
        self,
        initial_state: CoherentParticleState,
        n_steps: int = 100,
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.floating]],
        np.ndarray[Any, np.dtype[np.floating]],
    ]:
        z_points = np.linspace(0, self.length, n_steps + 1, endpoint=True)

        gyromagnetic_ratio = -2.04e8  # gyromagnetic ratio for 3He (rad s^-1 T^-1)
        effective_ratio = gyromagnetic_ratio / initial_state.parallel_velocity

        def _d_angles_dx(
            z: float, angles: tuple[float, float]
        ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
            theta, phi = angles
            # TODO: can we find B_phi and B_theta analytically to make this faster?  # noqa: FIX002
            field = _get_field(z, initial_state.displacement, self)

            # Ensure theta is not too close to 0 or pi to avoid coordinate singularity
            epsilon = 1e-12
            if np.abs(theta) < epsilon:
                theta = epsilon
            elif np.abs(theta - np.pi) < epsilon:
                theta = np.pi - epsilon

            # d_theta / dt = B_x sin phi - B_y cos phi
            d_theta = field[0] * np.sin(phi) - field[1] * np.cos(phi)
            # d_phi / dt = tan theta * (B_x cos phi + B_y sin phi) - B_z
            d_phi_xy = (field[0] * np.cos(phi) + field[1] * np.sin(phi)) / np.tan(theta)
            d_phi = d_phi_xy - field[2]
            return effective_ratio * np.array([d_theta, d_phi])

        y0 = np.array([initial_state.spin.theta.item(), initial_state.spin.phi.item()])

        sol = solve_ivp(  # type: ignore[return-value]
            fun=_d_angles_dx,
            t_span=(z_points[0], z_points[-1]),
            y0=y0,
            t_eval=z_points,
            vectorized=False,
            rtol=1e-8,
        )
        return np.array(sol.y)[0], np.array(sol.y)[1]  # type: ignore[return-value]

    def simulate_trajectory(
        self,
        initial_state: ParticleState,
        n_steps: int = 100,
    ) -> SolenoidTrajectory:
        """Run the spin echo simulation using configured parameters."""
        data = np.empty((n_steps + 1, initial_state.spin.size, 2), dtype=np.float64)
        for i, s in enumerate(initial_state.as_coherent()):
            thetas, phis = self._simulate_coherent_trajectory(s, n_steps)
            data[:, i, 0] = thetas
            data[:, i, 1] = phis

        spins = Spin[tuple[int, int]](data)
        z_points = np.linspace(0, self.length, n_steps + 1, endpoint=True)

        return SolenoidTrajectory(
            trajectory=Trajectory(
                spins=spins,
                displacement=initial_state.displacement,
                parallel_velocity=initial_state.parallel_velocity,
            ),
            positions=z_points,
        )

    @timed
    def simulate_trajectories(
        self,
        initial_states: list[ParticleState],
        n_steps: int = 100,
    ) -> SolenoidSimulationResult:
        """Run a solenoid simulation for multiple initial states."""
        z_points = np.linspace(0, self.length, n_steps + 1, endpoint=True)
        return SolenoidSimulationResult(
            trajectories=TrajectoryList.from_trajectories(
                [
                    self.simulate_trajectory(state, n_steps).trajectory
                    for state in tqdm(initial_states, desc="Simulating Trajectories")
                ]
            ),
            positions=z_points,
        )


@dataclass(kw_only=True, frozen=True)
class SolenoidTrajectory:
    """Represents the trajectory of a particle as it moves through the simulation."""

    trajectory: Trajectory
    positions: np.ndarray[Any, np.dtype[np.floating]]

    @property
    def spins(self) -> Spin[tuple[int, int]]:
        """The spin components from the simulation states."""
        return self.trajectory.spins

    @property
    def displacement(self) -> ParticleDisplacement:
        """The displacement of the particle at the end of the trajectory."""
        return self.trajectory.displacement


@dataclass(kw_only=True, frozen=True)
class SolenoidSimulationResult:
    """Represents the result of a solenoid simulation."""

    trajectories: TrajectoryList
    positions: np.ndarray[Any, np.dtype[np.floating]]

    @property
    def spins(self) -> Spin[tuple[int, int, int]]:
        """Extract the spin components from the simulation states."""
        return self.trajectories.spins

    @property
    def displacements(self) -> ParticleDisplacementList:
        """Extract the displacements from the simulation states."""
        return self.trajectories.displacements

    @property
    def spin_expectations(
        self,
    ) -> np.ndarray[tuple[Literal[3], int, int], np.dtype[np.floating]]:
        """Extract the transverse spin expectation values from the simulation states."""
        return expectation_values(self.trajectories.spins)


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
