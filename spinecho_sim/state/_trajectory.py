from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, overload, override

import numpy as np

from spinecho_sim.state import (
    MonatomicParticleState,
    ParticleDisplacementList,
    ParticleState,
    Spin,
    StateVectorParticleState,
)
from spinecho_sim.state._spin import EmptySpinList, EmptySpinListList

if TYPE_CHECKING:
    from spinecho_sim.state import (
        GenericSpinList,
        ParticleDisplacement,
    )


@dataclass(frozen=True, kw_only=True)
class Trajectory(Sequence[Any]):
    """A trajectory of a diatomic particle through the simulation."""

    _spin_angular_momentum: GenericSpinList
    _rotational_angular_momentum: GenericSpinList

    displacement: ParticleDisplacement
    parallel_velocity: float

    @staticmethod
    def from_states(
        states: Iterable[ParticleState],
    ) -> Trajectory:
        """Create a Trajectory from a list of ParticleStates."""
        velocities = np.array([state.parallel_velocity for state in states])
        assert np.allclose(velocities, velocities[0]), (
            "All states must have the same velocity."
        )
        displacements = [state.displacement for state in states]
        assert all(d == displacements[0] for d in displacements), (
            "All states must have the same displacement."
        )

        return Trajectory(
            _spin_angular_momentum=Spin.from_iter(s.spin for s in states),
            _rotational_angular_momentum=Spin.from_iter(
                s.rotational_angular_momentum for s in states
            ),
            displacement=displacements[0],
            parallel_velocity=velocities[0],
        )

    @override
    def __len__(self) -> int:
        return self.spin.shape[0]

    @property
    def spin(self) -> GenericSpinList:
        return self._spin_angular_momentum

    @property
    def rotational_angular_momentum(self) -> GenericSpinList:
        return self._rotational_angular_momentum

    @overload
    def __getitem__(self: Trajectory, index: int) -> ParticleState: ...

    @overload
    def __getitem__(self, index: slice | int) -> Trajectory: ...

    @override
    def __getitem__(self, index: int | slice) -> ParticleState | Trajectory:
        if isinstance(index, int):
            return ParticleState(
                _spin_angular_momentum=self.spin[index],
                _rotational_angular_momentum=self.rotational_angular_momentum[index],
                displacement=self.displacement,
                parallel_velocity=self.parallel_velocity,
            )

        return Trajectory(
            _spin_angular_momentum=self.spin[index],
            _rotational_angular_momentum=self.rotational_angular_momentum[index],
            displacement=self.displacement,
            parallel_velocity=self.parallel_velocity,
        )


@dataclass(frozen=True, kw_only=True)
class MonatomicTrajectory(Trajectory):
    _rotational_angular_momentum: EmptySpinList = field(
        init=False
    )  # Automatically set later

    def __post_init__(self) -> None:
        """Automatically set rotational angular momentum to an EmptySpinList with the same shape as spin."""
        object.__setattr__(
            self,
            "_rotational_angular_momentum",
            EmptySpinList(self._spin_angular_momentum.shape),
        )

    @staticmethod
    def from_monatomic_states(
        states: Iterable[MonatomicParticleState],
    ) -> MonatomicTrajectory:
        """Create a Trajectory from a list of ParticleStates."""
        velocities = np.array([state.parallel_velocity for state in states])
        assert np.allclose(velocities, velocities[0]), (
            "All states must have the same velocity."
        )
        displacements = [state.displacement for state in states]
        assert all(d == displacements[0] for d in displacements), (
            "All states must have the same displacement."
        )

        return MonatomicTrajectory(
            _spin_angular_momentum=Spin.from_iter(s.spin for s in states),
            displacement=displacements[0],
            parallel_velocity=velocities[0],
        )

    @property
    @override
    def rotational_angular_momentum(self) -> GenericSpinList:
        return EmptySpinList(self.rotational_angular_momentum.shape)

    @overload
    def __getitem__(
        self: MonatomicTrajectory, index: int
    ) -> MonatomicParticleState: ...

    @overload
    def __getitem__(self, index: slice | int) -> MonatomicTrajectory: ...

    @override
    def __getitem__(
        self, index: int | slice
    ) -> MonatomicParticleState | MonatomicTrajectory:
        if isinstance(index, int):
            return MonatomicParticleState(
                _spin_angular_momentum=self.spin[index],
                displacement=self.displacement,
                parallel_velocity=self.parallel_velocity,
            )

        return MonatomicTrajectory(
            _spin_angular_momentum=self.spin[index],
            displacement=self.displacement,
            parallel_velocity=self.parallel_velocity,
        )


class StateVectorTrajectory(Trajectory):
    """A trajectory that uses state vector representation instead of Spin classes."""

    # Override these fields with None to prevent initialization
    _spin_angular_momentum: EmptySpinList = field(init=False)
    _rotational_angular_momentum: EmptySpinList = field(init=False)
    state_vectors: np.ndarray[
        tuple[int, int], np.dtype[np.complex128]
    ]  # [timestep, state_vector_element]
    hilbert_space_dims: tuple[int, int]  # Dimensions of the subsystems

    displacement: ParticleDisplacement
    parallel_velocity: float

    def __init__(
        self,
        state_vectors: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
        hilbert_space_dims: tuple[int, int],
        displacement: ParticleDisplacement,
        parallel_velocity: float,
    ) -> None:
        self.state_vectors = state_vectors
        self.hilbert_space_dims = hilbert_space_dims
        self.displacement = displacement
        self.parallel_velocity = parallel_velocity

        expected_dim = np.prod(self.hilbert_space_dims)

        # Check state_vectors shape is compatible
        if self.state_vectors.shape[1] != expected_dim:
            msg = (
                f"State vector elements size {self.state_vectors.shape[1]} doesn't match "
                f"the expected dimension {expected_dim} from hilbert_space_dims {self.hilbert_space_dims}"
            )
            raise ValueError(msg)

        # Normalize each state vector
        normalized_vectors = np.zeros_like(self.state_vectors)
        for i, state in enumerate(self.state_vectors):
            normalized_vectors[i] = state / np.linalg.norm(state)

        self.state_vectors = normalized_vectors

        # Initialize empty spins for compatibility with parent class
        self._spin_angular_momentum = EmptySpinList((self.state_vectors.shape[0], 0))
        self._rotational_angular_momentum = EmptySpinList(
            (self.state_vectors.shape[0], 0)
        )

    @staticmethod
    def from_state_vector_states(
        states: Iterable[StateVectorParticleState],
    ) -> StateVectorTrajectory:
        """Create a StateVectorTrajectory from a list of StateVectorParticleStates."""
        velocities = np.array([state.parallel_velocity for state in states])
        assert np.allclose(velocities, velocities[0]), (
            "All states must have the same velocity."
        )
        displacements = [state.displacement for state in states]
        assert all(d == displacements[0] for d in displacements), (
            "All states must have the same displacement."
        )

        # Extract and stack state vectors
        state_vectors = np.vstack(
            [state.state_vector[np.newaxis, :] for state in states]
        )

        # Get hilbert_space_dims from the first state
        first_state = next(iter(states))
        hilbert_space_dims = first_state.hilbert_space_dims

        return StateVectorTrajectory(
            state_vectors=state_vectors,
            hilbert_space_dims=hilbert_space_dims,
            displacement=displacements[0],
            parallel_velocity=velocities[0],
        )

    @staticmethod
    def from_trajectory(
        trajectory: Trajectory, hilbert_space_dims: tuple[int, int]
    ) -> StateVectorTrajectory:
        """Convert a regular Trajectory to a StateVectorTrajectory."""
        state_vectors = np.array(
            [
                StateVectorParticleState.from_spin_state(trajectory[i]).state_vector
                for i in range(len(trajectory))
            ]
        )

        return StateVectorTrajectory(
            state_vectors=state_vectors,
            hilbert_space_dims=hilbert_space_dims,
            displacement=trajectory.displacement,
            parallel_velocity=trajectory.parallel_velocity,
        )

    @property
    @override
    def spin(self) -> Spin[tuple[int, int]]:
        """The spin components are not directly accessible in state vector representation."""
        msg = "Spin components are not available in state vector representation."
        raise NotImplementedError(msg)

    @property
    @override
    def rotational_angular_momentum(self) -> Spin[tuple[int, int]]:
        """The rotational angular momentum is not directly accessible in state vector representation."""
        msg = "Rotational angular momentum is not available in state vector representation."
        raise NotImplementedError(msg)

    @override
    def __len__(self) -> int:
        return self.state_vectors.shape[0]

    @overload
    def __getitem__(
        self: StateVectorTrajectory, index: int
    ) -> StateVectorParticleState: ...
    @overload
    def __getitem__(self, index: slice | int) -> StateVectorTrajectory: ...

    @override
    def __getitem__(
        self, index: int | slice
    ) -> StateVectorParticleState | StateVectorTrajectory:
        if isinstance(index, int):
            return StateVectorParticleState(
                state_vector=self.state_vectors[index],
                hilbert_space_dims=self.hilbert_space_dims,
                displacement=self.displacement,
                parallel_velocity=self.parallel_velocity,
            )

        return StateVectorTrajectory(
            state_vectors=self.state_vectors[index],
            hilbert_space_dims=self.hilbert_space_dims,
            displacement=self.displacement,
            parallel_velocity=self.parallel_velocity,
        )


@dataclass(kw_only=True, frozen=True)
class TrajectoryList(Sequence[Trajectory]):
    """A list of diatomic trajectories."""

    _spin_angular_momentum: Spin[tuple[int, int, int]]
    _rotational_angular_momentum: Spin[tuple[int, int, int]]
    displacements: ParticleDisplacementList
    parallel_velocities: np.ndarray[Any, np.dtype[np.floating]]

    def __post_init__(self) -> None:
        if (
            self.parallel_velocities.ndim != 1
            or self.parallel_velocities.shape != self.displacements.shape
            or self.parallel_velocities.size != self._spin_angular_momentum.shape[0]
            or self.parallel_velocities.size
            != self.rotational_angular_momentum.shape[0]
        ):
            msg = "Spins must be a 2D array, parallel velocities and displacements must be 1D arrays, and their shapes must match."
            raise ValueError(msg)

    @property
    def spin(self) -> Spin[tuple[int, int, int]]:
        return self._spin_angular_momentum

    @property
    def rotational_angular_momentum(self) -> Spin[tuple[int, int, int]]:
        return self._rotational_angular_momentum

    @staticmethod
    def from_trajectories(
        trajectories: Iterable[Trajectory],
    ) -> TrajectoryList:
        """Create a DiatomicTrajectoryList from a list of DiatomicTrajectories."""
        nuclear_spins = Spin.from_iter(t.spin for t in trajectories)
        rotational_spins = Spin.from_iter(
            t.rotational_angular_momentum for t in trajectories
        )
        displacements = ParticleDisplacementList.from_displacements(
            t.displacement for t in trajectories
        )
        parallel_velocities = np.array([t.parallel_velocity for t in trajectories])
        return TrajectoryList(
            _spin_angular_momentum=nuclear_spins,
            _rotational_angular_momentum=rotational_spins,
            displacements=displacements,
            parallel_velocities=parallel_velocities,
        )

    @override
    def __len__(self) -> int:
        return len(self.parallel_velocities)

    @overload
    def __getitem__(self, index: int) -> Trajectory: ...
    @overload
    def __getitem__(self, index: slice) -> TrajectoryList: ...

    @override
    def __getitem__(self, index: int | slice) -> Trajectory | TrajectoryList:
        if isinstance(index, slice):
            return TrajectoryList(
                _spin_angular_momentum=self.spin[index],
                _rotational_angular_momentum=self.rotational_angular_momentum[index],
                displacements=self.displacements[index],
                parallel_velocities=self.parallel_velocities[index],
            )
        return Trajectory(
            _spin_angular_momentum=self.spin[index],
            _rotational_angular_momentum=self.rotational_angular_momentum[index],
            displacement=self.displacements[index],
            parallel_velocity=self.parallel_velocities[index],
        )

    @override
    def __iter__(self) -> Iterator[Trajectory]:
        for i in range(len(self)):
            yield Trajectory(
                _spin_angular_momentum=self.spin[i],
                _rotational_angular_momentum=self.rotational_angular_momentum[i],
                displacement=self.displacements[i],
                parallel_velocity=self.parallel_velocities[i],
            )


@dataclass(frozen=True, kw_only=True)
class MonatomicTrajectoryList(TrajectoryList):
    """A list of monatomic trajectories."""

    _rotational_angular_momentum: EmptySpinListList = field(
        init=False
    )  # Automatically set later

    @override
    def __post_init__(self) -> None:
        if (
            self.parallel_velocities.ndim != 1
            or self.parallel_velocities.shape != self.displacements.shape
            or self.parallel_velocities.size != self.spin.shape[0]
        ):
            msg = "Spins must be a 2D array, parallel velocities and displacements must be 1D arrays, and their shapes must match."
            raise ValueError(msg)
        """Automatically set rotational angular momentum to an EmptySpinList with the same shape as spin."""
        object.__setattr__(
            self,
            "_rotational_angular_momentum",
            EmptySpinListList(self._spin_angular_momentum.shape),
        )

    @property
    @override
    def rotational_angular_momentum(self) -> Spin[tuple[int, int, int]]:
        return self._rotational_angular_momentum

    @staticmethod
    def from_monatomic_trajectories(
        trajectories: Iterable[MonatomicTrajectory],
    ) -> MonatomicTrajectoryList:
        """Create a MonatomicTrajectoryList from a list of MonatomicTrajectories."""
        spins = Spin.from_iter(t.spin for t in trajectories)
        displacements = ParticleDisplacementList.from_displacements(
            t.displacement for t in trajectories
        )
        parallel_velocities = np.array([t.parallel_velocity for t in trajectories])
        return MonatomicTrajectoryList(
            _spin_angular_momentum=spins,
            displacements=displacements,
            parallel_velocities=parallel_velocities,
        )

    @overload
    def __getitem__(self, index: int) -> MonatomicTrajectory: ...
    @overload
    def __getitem__(self, index: slice) -> MonatomicTrajectoryList: ...

    @override
    def __getitem__(
        self, index: int | slice
    ) -> MonatomicTrajectory | MonatomicTrajectoryList:
        if isinstance(index, slice):
            return MonatomicTrajectoryList(
                _spin_angular_momentum=self.spin[index],
                displacements=self.displacements[index],
                parallel_velocities=self.parallel_velocities[index],
            )
        return MonatomicTrajectory(
            _spin_angular_momentum=self.spin[index],
            displacement=self.displacements[index],
            parallel_velocity=self.parallel_velocities[index],
        )

    @override
    def __iter__(self) -> Iterator[MonatomicTrajectory]:
        for i in range(len(self)):
            yield MonatomicTrajectory(
                _spin_angular_momentum=self.spin[i],
                displacement=self.displacements[i],
                parallel_velocity=self.parallel_velocities[i],
            )


class StateVectorTrajectoryList(TrajectoryList):
    """A list of state vector trajectories."""

    state_vectors: np.ndarray[
        tuple[int, int, int], np.dtype[np.complex128]
    ]  # [trajectory, timestep, state_vector_element]
    hilbert_space_dims: tuple[int, int]  # [nuclear, rotational]
    displacements: ParticleDisplacementList
    parallel_velocities: np.ndarray[Any, np.dtype[np.floating]]

    def __init__(
        self,
        state_vectors: np.ndarray,
        hilbert_space_dims: tuple[int, int],
        displacements: ParticleDisplacementList,
        parallel_velocities: np.ndarray,
    ) -> None:
        self.state_vectors = state_vectors
        self.hilbert_space_dims = hilbert_space_dims
        self.displacements = displacements
        self.parallel_velocities = parallel_velocities

        if (
            self.parallel_velocities.ndim != 1
            or self.parallel_velocities.shape[0] != self.state_vectors.shape[0]
            or self.parallel_velocities.shape[0] != len(self.displacements)
        ):
            msg = "State vectors, parallel velocities and displacements must have compatible shapes."
            raise ValueError(msg)

    @staticmethod
    def from_state_vector_trajectories(
        trajectories: Iterable[StateVectorTrajectory],
    ) -> StateVectorTrajectoryList:
        """Create a StateVectorTrajectoryList from a list of StateVectorTrajectories."""
        # Get the first trajectory to extract properties
        trajectory_list = list(trajectories)

        # Check that all trajectories have the same number of steps and vector dimensions
        first_trajectory = trajectory_list[0]
        n_steps = len(first_trajectory)
        vector_dim = first_trajectory.state_vectors.shape[1]
        hilbert_space_dims = first_trajectory.hilbert_space_dims

        for t in trajectory_list:
            assert len(t) == n_steps, f"Expected {n_steps} steps, got {len(t)}"
            assert t.state_vectors.shape[1] == vector_dim, (
                f"Expected state vector dimension {vector_dim}, got {t.state_vectors.shape[1]}"
            )
            assert t.hilbert_space_dims == hilbert_space_dims, (
                f"Expected Hilbert space dimensions {hilbert_space_dims}, got {t.hilbert_space_dims}"
            )

        # Stack state vectors
        state_vectors = np.stack([t.state_vectors for t in trajectory_list])

        # Collect displacements and velocities
        displacements = ParticleDisplacementList.from_displacements(
            t.displacement for t in trajectory_list
        )
        parallel_velocities = np.array([t.parallel_velocity for t in trajectory_list])

        return StateVectorTrajectoryList(
            state_vectors=state_vectors,
            hilbert_space_dims=hilbert_space_dims,
            displacements=displacements,
            parallel_velocities=parallel_velocities,
        )

    @override
    def __len__(self) -> int:
        return self.state_vectors.shape[0]

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

    @overload
    def __getitem__(self, index: int) -> StateVectorTrajectory: ...
    @overload
    def __getitem__(self, index: slice) -> StateVectorTrajectoryList: ...

    @override
    def __getitem__(
        self, index: int | slice
    ) -> StateVectorTrajectory | StateVectorTrajectoryList:
        if isinstance(index, int):
            return StateVectorTrajectory(
                state_vectors=self.state_vectors[index],
                hilbert_space_dims=self.hilbert_space_dims,
                displacement=self.displacements[index],
                parallel_velocity=self.parallel_velocities[index],
            )

        return StateVectorTrajectoryList(
            state_vectors=self.state_vectors[index],
            hilbert_space_dims=self.hilbert_space_dims,
            displacements=self.displacements[index],
            parallel_velocities=self.parallel_velocities[index],
        )

    @override
    def __iter__(self) -> Iterator[StateVectorTrajectory]:
        for i in range(len(self)):
            yield StateVectorTrajectory(
                state_vectors=self.state_vectors[i],
                hilbert_space_dims=self.hilbert_space_dims,
                displacement=self.displacements[i],
                parallel_velocity=self.parallel_velocities[i],
            )
