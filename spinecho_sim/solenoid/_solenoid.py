"""Core simulation functionality for spin echo experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

import numpy as np
from tqdm import tqdm

from spinecho_sim.molecule import diatomic_hamiltonian_dicke
from spinecho_sim.state import (
    EmptySpinList,
    EmptySpinListList,
    MonatomicParticleState,
    MonatomicTrajectory,
    MonatomicTrajectoryList,
    ParticleDisplacement,
    ParticleDisplacementList,
    ParticleState,
    Spin,
    StateVectorParticleState,
    StateVectorTrajectory,
    StateVectorTrajectoryList,
    Trajectory,
    TrajectoryList,
)
from spinecho_sim.util import solve_ivp_typed, sparse_apply, timed

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from spinecho_sim.state._state import (
        CoherentMonatomicParticleState,
    )


@dataclass(kw_only=True, frozen=True)
class SolenoidTrajectory:
    """Represents the trajectory of a diatomic particle in a solenoid."""

    trajectory: Trajectory
    positions: np.ndarray[Any, np.dtype[np.floating]]

    @property
    def spin(self) -> Spin[tuple[int, int]]:
        """The spin components from the simulation states."""
        return self.trajectory.spin

    @property
    def rotational_angular_momentum(self) -> Spin[tuple[int, int]]:
        """The rotational angular momentum of the particle."""
        return self.trajectory.rotational_angular_momentum

    @property
    def displacement(self) -> ParticleDisplacement:
        """The displacement of the particle at the end of the trajectory."""
        return self.trajectory.displacement


@dataclass(kw_only=True, frozen=True)
class MonatomicSolenoidTrajectory(SolenoidTrajectory):
    """Represents the trajectory of a monatomic particle in a solenoid."""

    trajectory: MonatomicTrajectory

    @property
    @override
    def rotational_angular_momentum(self) -> Spin[tuple[int, int]]:
        """The rotational angular momentum of the particle."""
        return EmptySpinList(self.spin.shape)


@dataclass(kw_only=True, frozen=True)
class StateVectorSolenoidTrajectory(SolenoidTrajectory):
    """Represents a trajectory in a solenoid using state vectors instead of spins."""

    trajectory: StateVectorTrajectory
    positions: np.ndarray[Any, np.dtype[np.floating]]

    @property
    def state_vectors(self) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        """Get the state vectors from the trajectory."""
        return self.trajectory.state_vectors

    @property
    def hilbert_space_dims(self) -> tuple[int, int]:
        """Get the Hilbert space dimensions of the state vectors."""
        return self.trajectory.hilbert_space_dims

    @property
    @override
    def spin(self) -> Spin[tuple[int, int]]:
        msg = "Spin components are not available in state vector representation."
        raise NotImplementedError(msg)

    @property
    @override
    def rotational_angular_momentum(self) -> Spin[tuple[int, int]]:
        msg = "Rotational angular momentum is not available in state vector representation."
        raise NotImplementedError(msg)

    @staticmethod
    def from_solenoid_trajectory(
        solenoid_trajectory: SolenoidTrajectory,
        hilbert_space_dims: tuple[int, int],
    ) -> StateVectorSolenoidTrajectory:
        """Create a StateVectorSolenoidTrajectory from a SolenoidTrajectory."""
        return StateVectorSolenoidTrajectory(
            trajectory=StateVectorTrajectory.from_trajectory(
                solenoid_trajectory.trajectory, hilbert_space_dims
            ),
            positions=solenoid_trajectory.positions,
        )


@dataclass(kw_only=True, frozen=True)
class SolenoidSimulationResult:
    """Represents the result of a solenoid simulation."""

    trajectories: TrajectoryList
    positions: np.ndarray[Any, np.dtype[np.floating]]

    @property
    def spin(self) -> Spin[tuple[int, int, int]]:
        return self.trajectories.spin

    @property
    def rotational_angular_momentum(self) -> Spin[tuple[int, int, int]]:
        return self.trajectories.rotational_angular_momentum

    @property
    def displacements(self) -> ParticleDisplacementList:
        """Extract the displacements from the simulation states."""
        return self.trajectories.displacements


@dataclass(kw_only=True, frozen=True)
class MonatomicSolenoidSimulationResult(SolenoidSimulationResult):
    trajectories: MonatomicTrajectoryList

    @property
    @override
    def rotational_angular_momentum(self) -> Spin[tuple[int, int, int]]:
        return EmptySpinListList(self.spin.shape)


@dataclass(kw_only=True, frozen=True)
class StateVectorSolenoidSimulationResult(SolenoidSimulationResult):
    """Represents the result of a solenoid simulation using state vectors."""

    trajectories: StateVectorTrajectoryList
    positions: np.ndarray[Any, np.dtype[np.floating]]

    @property
    def state_vectors(
        self,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
        """Get the state vectors from the trajectories."""
        return self.trajectories.state_vectors

    @property
    def hilbert_space_dims(self) -> tuple[int, int]:
        """Get the Hilbert space dimensions of the state vectors."""
        return self.trajectories.hilbert_space_dims

    @property
    @override
    def spin(self) -> Spin[tuple[int, int, int]]:
        msg = "Spin components are not available in state vector representation."
        raise NotImplementedError(msg)

    @property
    @override
    def rotational_angular_momentum(self) -> Spin[tuple[int, int, int]]:
        msg = "Rotational angular momentum is not available in state vector representation."
        raise NotImplementedError(msg)

    @staticmethod
    def from_simulation_result(
        result: SolenoidSimulationResult,
        hilbert_space_dims: tuple[int, int],
    ) -> StateVectorSolenoidSimulationResult:
        """Convert a regular SolenoidSimulationResult to a StateVectorSolenoidSimulationResult."""
        # Create state vector trajectories from regular trajectories
        state_vector_trajectories: list[StateVectorTrajectory] = []
        for i in range(len(result.trajectories)):
            trajectory = result.trajectories[i]
            sv_trajectory = StateVectorTrajectory.from_trajectory(
                trajectory, hilbert_space_dims
            )
            state_vector_trajectories.append(sv_trajectory)

        return StateVectorSolenoidSimulationResult(
            trajectories=StateVectorTrajectoryList.from_state_vector_trajectories(
                state_vector_trajectories
            ),
            positions=result.positions,
        )


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

    def simulate_trajectory(
        self,
        initial_state: ParticleState,
        n_steps: int = 100,
    ) -> SolenoidTrajectory:
        assert isinstance(initial_state, StateVectorParticleState)
        i = initial_state.spin.size / 2 - 1
        j = initial_state.rotational_angular_momentum.size / 2 - 1

        z_points = np.linspace(0, self.length, n_steps + 1, endpoint=True)

        def schrodinger_eq(z: float, psi: np.ndarray) -> np.ndarray:
            field = _get_field(z, initial_state.displacement, self)
            b_vec = (field[0], field[1], field[2])
            hamiltonian = diatomic_hamiltonian_dicke(
                i, j, initial_state.coefficients, b_vec
            )
            result = sparse_apply(hamiltonian, psi)
            return -1j * result

        psi0: np.ndarray[tuple[int], np.dtype[np.complex128]] = np.kron(
            initial_state.spin.momentum_states,
            initial_state.rotational_angular_momentum.momentum_states,
        ).astype(np.complex128)

        sol = solve_ivp_typed(
            fun=schrodinger_eq,
            t_span=(z_points[0], z_points[-1]),
            y0=psi0,
            t_eval=z_points,
            rtol=1e-8,
        )

        state_vectors: np.ndarray[tuple[int, int], np.dtype[np.complex128]] = (
            np.transpose(sol.y).astype(np.complex128)
        )

        return StateVectorSolenoidTrajectory(
            trajectory=StateVectorTrajectory(
                state_vectors=state_vectors,
                hilbert_space_dims=initial_state.hilbert_space_dims,
                displacement=initial_state.displacement,
                parallel_velocity=initial_state.parallel_velocity,
            ),
            positions=z_points,
        )

    @timed
    def simulate_trajectories(
        self,
        initial_states: Sequence[ParticleState],
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


@dataclass(kw_only=True, frozen=True)
class MonatomicSolenoid(Solenoid):
    def _simulate_coherent_trajectory(
        self,
        initial_state: CoherentMonatomicParticleState,
        n_steps: int = 100,
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.floating]],
        np.ndarray[Any, np.dtype[np.floating]],
    ]:
        z_points = np.linspace(0, self.length, n_steps + 1, endpoint=True)

        gyromagnetic_ratio = initial_state.gyromagnetic_ratio
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

        y0 = np.array(
            [
                initial_state.spin.theta.item(),
                initial_state.spin.phi.item(),
            ]
        )

        sol = solve_ivp_typed(
            fun=_d_angles_dx,  # pyright: ignore[reportArgumentType]
            t_span=(z_points[0], z_points[-1]),
            y0=y0,
            t_eval=z_points,
            vectorized=False,
            rtol=1e-8,
        )
        return sol.y[0], sol.y[1]

    @override
    def simulate_trajectory(
        self,
        initial_state: ParticleState,
        n_steps: int = 100,
    ) -> MonatomicSolenoidTrajectory:
        """Run the spin echo simulation using configured parameters."""
        assert isinstance(initial_state, MonatomicParticleState), (
            "Expected a coherent monatomic particle state."
        )

        data = np.empty((n_steps + 1, initial_state.spin.size, 2), dtype=np.float64)
        for i, s in enumerate(initial_state.as_coherent()):
            thetas, phis = self._simulate_coherent_trajectory(s, n_steps)
            data[:, i, 0] = thetas
            data[:, i, 1] = phis

        spins = Spin[tuple[int, int]](data)
        z_points = np.linspace(0, self.length, n_steps + 1, endpoint=True)

        return MonatomicSolenoidTrajectory(
            trajectory=MonatomicTrajectory(
                _spin_angular_momentum=spins,
                displacement=initial_state.displacement,
                parallel_velocity=initial_state.parallel_velocity,
            ),
            positions=z_points,
        )

    @timed
    @override
    def simulate_trajectories(
        self,
        initial_states: Sequence[ParticleState],
        n_steps: int = 100,
    ) -> MonatomicSolenoidSimulationResult:
        """Run a solenoid simulation for multiple initial states."""
        mono_initial_states = [
            state
            for state in initial_states
            if isinstance(state, MonatomicParticleState)
        ]
        assert mono_initial_states, "No MonatomicParticleState instances provided."

        z_points = np.linspace(0, self.length, n_steps + 1, endpoint=True)
        return MonatomicSolenoidSimulationResult(
            trajectories=MonatomicTrajectoryList.from_monatomic_trajectories(
                [
                    self.simulate_trajectory(state, n_steps).trajectory
                    for state in tqdm(
                        mono_initial_states, desc="Simulating Trajectories"
                    )
                ]
            ),
            positions=z_points,
        )
