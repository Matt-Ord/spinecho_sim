from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

import numpy as np

from spinecho_sim.state._displacement import ParticleDisplacement
from spinecho_sim.state._spin import EmptySpin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from spinecho_sim.state._spin import GenericSpin


@dataclass(frozen=True, kw_only=True)
class ParticleState:
    """State of a diatomic particle."""

    displacement: ParticleDisplacement = field(default_factory=ParticleDisplacement)
    parallel_velocity: float
    _spin_angular_momentum: GenericSpin
    _rotational_angular_momentum: GenericSpin
    coefficients: tuple[float, float, float, float] = (4.258, 0.6717, 113.8, 57.68)

    @property
    def spin(self) -> GenericSpin:
        return self._spin_angular_momentum

    @property
    def rotational_angular_momentum(self) -> GenericSpin:
        return self._rotational_angular_momentum


@dataclass(frozen=True, kw_only=True)
class MonatomicParticleState(ParticleState):
    gyromagnetic_ratio: float = -2.04e8  # default value for 3He
    _rotational_angular_momentum: EmptySpin = field(init=False, default=EmptySpin())

    @property
    @override
    def spin(self) -> GenericSpin:
        return self._spin_angular_momentum

    def as_coherent(self) -> Sequence[CoherentMonatomicParticleState]:
        return [
            CoherentMonatomicParticleState(
                _spin_angular_momentum=s.as_generic(),
                displacement=self.displacement,
                parallel_velocity=self.parallel_velocity,
                gyromagnetic_ratio=self.gyromagnetic_ratio,
            )
            for s in self._spin_angular_momentum.flat_iter()
        ]

    @property
    @override
    def rotational_angular_momentum(self) -> GenericSpin:
        return EmptySpin()


@dataclass(frozen=True, kw_only=True)
class CoherentMonatomicParticleState(MonatomicParticleState):
    _rotational_angular_momentum: EmptySpin = field(init=False, default=EmptySpin())

    def __post_init__(self) -> None:
        assert self.spin.size == 1, (
            "CoherentParticleState must represent a single coherent spin."
        )
        assert self.rotational_angular_momentum.size == 1, (
            "CoherentParticleState must represent a single coherent spin."
        )
        assert self._spin_angular_momentum.size == 1, (
            "CoherentParticleState must represent a single coherent spin."
        )


class StateVectorParticleState(ParticleState):
    """Particle state that directly uses state vector arrays instead of Spin classes."""

    # Override these fields with None to prevent initialization
    _spin_angular_momentum: GenericSpin = field(init=False, default=EmptySpin())
    _rotational_angular_momentum: GenericSpin = field(init=False, default=EmptySpin())

    state_vector: np.ndarray[tuple[int], np.dtype[np.complex128]]
    hilbert_space_dims: tuple[int, int]  # Dimensions of the subsystems
    displacement: ParticleDisplacement
    parallel_velocity: float

    def __init__(
        self,
        state_vector: np.ndarray[tuple[int], np.dtype[np.complex128]],
        hilbert_space_dims: tuple[int, int],
        displacement: ParticleDisplacement,
        parallel_velocity: float,
    ) -> None:
        self.state_vector = state_vector
        self.hilbert_space_dims = hilbert_space_dims
        self.displacement = displacement
        self.parallel_velocity = parallel_velocity
        normalized_vector = self.state_vector / np.linalg.norm(self.state_vector)
        object.__setattr__(self, "state_vector", normalized_vector)
        expected_dim = np.prod(self.hilbert_space_dims)
        if self.state_vector.size != expected_dim:
            msg = (
                f"State vector size {self.state_vector.size} doesn't match "
                f"the expected dimension {expected_dim} from hilbert_space_dims {self.hilbert_space_dims}"
            )
            raise ValueError(msg)

    @property
    @override
    def spin(self) -> GenericSpin:
        msg = "StateVectorParticleState does not use Spin representations."
        raise NotImplementedError(msg)

    @property
    @override
    def rotational_angular_momentum(self) -> GenericSpin:
        msg = "StateVectorParticleState does not use Spin representations."
        raise NotImplementedError(msg)

    @staticmethod
    def from_spin_state(state: ParticleState) -> StateVectorParticleState:
        """Create a StateVectorParticleState from a traditional ParticleState."""
        state_vector = np.kron(
            state.spin.momentum_states,
            state.rotational_angular_momentum.momentum_states,
        ).astype(np.complex128)  # Explicitly cast to complex128
        return StateVectorParticleState(
            state_vector=state_vector,
            hilbert_space_dims=(
                state.spin.size,
                state.rotational_angular_momentum.size,
            ),
            displacement=state.displacement,
            parallel_velocity=state.parallel_velocity,
        )
