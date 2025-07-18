from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, overload, override

import numpy as np


@dataclass(kw_only=True, frozen=True)
class ParticleDisplacement:
    """Represents the displacement of a particle in the simulation.

    This defines the displacement in the x-y plane, perpendicular to the particle's velocity.
    The displacement is stored in polar coordinates (r, theta), where:
    - r is the radial distance from the origin (0, 0) in the x-y plane.
    - theta is the angle from the positive x-axis in the x-y plane.
    """

    r: float
    theta: float

    @property
    def x(self) -> float:
        """Get the x-coordinate of the displacement."""
        return self.r * np.cos(self.theta)

    @property
    def y(self) -> float:
        """Get the y-coordinate of the displacement."""
        return self.r * np.sin(self.theta)

    @staticmethod
    def from_cartesian(x: float, y: float) -> ParticleDisplacement:
        """Create a ParticleDisplacement from Cartesian coordinates."""
        return ParticleDisplacement(r=np.sqrt(x**2 + y**2), theta=np.arctan2(y, x))


@dataclass(kw_only=True, frozen=True)
class ParticleDisplacementList(Sequence[ParticleDisplacement]):
    """A list of particle displacements."""

    r: np.ndarray[Any, np.dtype[np.floating]]
    theta: np.ndarray[Any, np.dtype[np.floating]]

    def __post_init__(self) -> None:
        if self.theta.shape != self.r.shape:
            msg = "Theta and r arrays must have the same shape."
            raise ValueError(msg)

    @staticmethod
    def from_displacements(
        displacements: Iterable[ParticleDisplacement],
    ) -> ParticleDisplacementList:
        """Create a ParticleDisplacementList from a list of ParticleDisplacements."""
        displacements = list(displacements)
        return ParticleDisplacementList(
            r=np.array([d.r for d in displacements]),
            theta=np.array([d.theta for d in displacements]),
        )

    @override
    def __len__(self) -> int:
        return self.theta.shape[0]

    @overload
    def __getitem__(self, index: int) -> ParticleDisplacement: ...

    @overload
    def __getitem__(self, index: slice) -> ParticleDisplacementList: ...

    @override
    def __getitem__(
        self, index: int | slice
    ) -> ParticleDisplacement | ParticleDisplacementList:
        if isinstance(index, slice):
            return ParticleDisplacementList(r=self.r[index], theta=self.theta[index])
        return ParticleDisplacement(r=self.r.item(index), theta=self.theta.item(index))

    @override
    def __iter__(self) -> Iterator[ParticleDisplacement]:
        for r, t in zip(self.r, self.theta, strict=True):
            yield ParticleDisplacement(r=r, theta=t)

    @property
    def x(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Get the x-displacement of the particles."""
        return np.array([displacement.x for displacement in self])

    @property
    def y(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Get the y-displacement of the particles."""
        return np.array([displacement.y for displacement in self])

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the displacement list."""
        return self.r.shape


@dataclass(kw_only=True, frozen=True)
class CoherentSpin:
    r"""A vector representing the spin of a particle in a classical coherent state.

    Spin in a classical coherent state can be treated as a vector on the unit sphere.
    The theta and phi angles are defined in spherical coordinates, where theta is the polar angle
    (angle from the positive z-axis) and phi is the azimuthal angle
    (angle in the x-y plane from the positive x-axis).

    Classical coherent states are often used to model the behavior of spins in magnetic fields,
    as the dynamics of a coherent spin state can be described by the Larmor precession formula
    in the presence of a magnetic field.
    """

    theta: float
    phi: float

    @property
    def x(self) -> float:
        """Get the x-component of the spin vector."""
        return np.sin(self.theta) * np.cos(self.phi)

    @property
    def y(self) -> float:
        """Get the y-component of the spin vector."""
        return np.sin(self.theta) * np.sin(self.phi)

    @property
    def z(self) -> float:
        """Get the z-component of the spin vector."""
        return np.cos(self.theta)

    @property
    def cartesian(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Get the Cartesian coordinates of the spin vector."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @staticmethod
    def from_cartesian(x: float, y: float, z: float) -> CoherentSpin:
        """Create a Spin from Cartesian coordinates."""
        r = np.sqrt(x**2 + y**2 + z**2)
        assert np.isclose(r, 1, rtol=1e-3), f"Spin vector must be normalized. r = {r}"
        return CoherentSpin(theta=np.arccos(z / r), phi=np.arctan2(y, x))


@dataclass(kw_only=True, frozen=True)
class CoherentSpinList(Sequence[CoherentSpin]):
    """A sequence of `CoherentSpin` states."""

    theta: np.ndarray[Any, np.dtype[np.floating]]
    phi: np.ndarray[Any, np.dtype[np.floating]]

    def __post_init__(self) -> None:
        if self.theta.shape != self.phi.shape:
            msg = "Theta and phi arrays must have the same shape."
            raise ValueError(msg)

    @staticmethod
    def from_spins(states: Iterable[CoherentSpin]) -> CoherentSpinList:
        """Create a CoherentSpinList from a list of CoherentSpins."""
        states = list(states)
        return CoherentSpinList(
            theta=np.array([state.theta for state in states]),
            phi=np.array([state.phi for state in states]),
        )

    @override
    def __len__(self) -> int:
        return self.theta.shape[0]

    @overload
    def __getitem__(self, index: int) -> CoherentSpin: ...

    @overload
    def __getitem__(self, index: slice) -> CoherentSpinList: ...

    @override
    def __getitem__(self, index: int | slice) -> CoherentSpin | CoherentSpinList:
        if isinstance(index, slice):
            return CoherentSpinList(theta=self.theta[index], phi=self.phi[index])
        return CoherentSpin(theta=self.theta.item(index), phi=self.phi.item(index))

    @override
    def __iter__(self) -> Iterator[CoherentSpin]:
        for t, p in zip(self.theta.ravel(), self.phi.ravel(), strict=True):
            yield CoherentSpin(theta=t.item(), phi=p.item())

    @property
    def x(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Get the x-components of the spins."""
        return np.array([spin.x for spin in self])

    @property
    def y(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Get the y-components of the spins."""
        return np.array([spin.y for spin in self])

    @property
    def z(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Get the z-components of the spins."""
        return np.array([spin.z for spin in self])

    @property
    def cartesian(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Get the Cartesian coordinates of the spins."""
        return np.array([spin.cartesian for spin in self]).reshape((*self.shape, 3))

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the spin list."""
        return self.theta.shape


@dataclass(kw_only=True, frozen=True)
class ParticleState:
    """Represents the state of a particle in the simulation."""

    spin: CoherentSpin
    displacement: ParticleDisplacement
    parallel_velocity: float


@dataclass(kw_only=True, frozen=True)
class Trajectory(Sequence[ParticleState]):
    """A trajectory of a particle through the simulation."""

    spins: CoherentSpinList
    displacement: ParticleDisplacement
    parallel_velocity: float

    def __post_init__(self) -> None:
        if self.spins.theta.ndim != 1:
            msg = "Spins must be a 1D array."
            raise ValueError(msg)

    @staticmethod
    def from_states(
        states: Iterable[ParticleState],
    ) -> Trajectory:
        """Create a Trajectory from a list of ParticleStates."""
        states = list(states)
        velocities = np.array([state.parallel_velocity for state in states])
        assert np.allclose(velocities, velocities[0]), (
            "All states must have the same velocity."
        )
        displacements = [state.displacement for state in states]
        assert all(d == displacements[0] for d in displacements), (
            "All states must have the same displacement."
        )
        return Trajectory(
            spins=CoherentSpinList.from_spins(s.spin for s in states),
            displacement=displacements[0],
            parallel_velocity=velocities[0],
        )

    @override
    def __len__(self) -> int:
        return self.spins.shape[0]

    @overload
    def __getitem__(self, index: int) -> ParticleState: ...

    @overload
    def __getitem__(self, index: slice) -> Trajectory: ...

    @override
    def __getitem__(self, index: int | slice) -> ParticleState | Trajectory:
        if isinstance(index, slice):
            return Trajectory(
                spins=self.spins[index],
                displacement=self.displacement,
                parallel_velocity=self.parallel_velocity,
            )
        return ParticleState(
            spin=self.spins[index],
            displacement=self.displacement,
            parallel_velocity=self.parallel_velocity,
        )


@dataclass(kw_only=True, frozen=True)
class TrajectoryList(Sequence[Trajectory]):
    """A list of trajectories."""

    spins: CoherentSpinList
    displacements: ParticleDisplacementList
    parallel_velocities: np.ndarray[Any, np.dtype[np.floating]]

    def __post_init__(self) -> None:
        if (
            self.spins.theta.ndim != 2  # noqa: PLR2004
            or self.parallel_velocities.ndim != 1
            or self.parallel_velocities.shape != self.displacements.shape
            or self.parallel_velocities.size != self.spins.shape[0]
        ):
            msg = "Spins must be a 2D array, parallel velocities must be a 1D array, and their shapes must match."
            raise ValueError(msg)

    @staticmethod
    def from_trajectories(
        trajectories: Iterable[Trajectory],
    ) -> TrajectoryList:
        """Create a TrajectoryList from a list of Trajectories."""
        trajectories = list(trajectories)
        spins = CoherentSpinList(
            theta=np.array([t.spins.theta for t in trajectories]),
            phi=np.array([t.spins.phi for t in trajectories]),
        )
        displacements = ParticleDisplacementList.from_displacements(
            t.displacement for t in trajectories
        )
        parallel_velocities = np.array([t.parallel_velocity for t in trajectories])
        return TrajectoryList(
            spins=spins,
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
                spins=self.spins[index],
                displacements=self.displacements[index],
                parallel_velocities=self.parallel_velocities[index],
            )
        return Trajectory(
            spins=CoherentSpinList(
                theta=self.spins.theta[index],
                phi=self.spins.phi[index],
            ),
            displacement=self.displacements[index],
            parallel_velocity=self.parallel_velocities[index],
        )

    @override
    def __iter__(self) -> Iterator[Trajectory]:
        for i in range(len(self)):
            yield Trajectory(
                spins=CoherentSpinList(
                    theta=self.spins.theta[i],
                    phi=self.spins.phi[i],
                ),
                displacement=self.displacements[i],
                parallel_velocity=self.parallel_velocities[i],
            )


def sample_uniform_spin(n: int) -> list[CoherentSpin]:
    """Sample N uniform random directions on the unit sphere."""
    rng = np.random.default_rng()
    phi = rng.uniform(0, 2 * np.pi, size=n)
    theta = rng.uniform(0, np.pi, size=n)
    return [CoherentSpin(theta=t, phi=p) for t, p in zip(theta, phi, strict=True)]


def sample_s_unit_circle(n: int) -> list[CoherentSpin]:
    """Sample N uniform random directions on the unit circle normal to z-axis."""
    rng = np.random.default_rng()
    phi = rng.uniform(0, 2 * np.pi, size=n)
    return [CoherentSpin(theta=np.pi / 2, phi=p) for p in phi]


def sample_uniform_displacement(n: int, r_max: float) -> list[ParticleDisplacement]:
    """Sample N random radii uniformly distributed over a disk of radius r_max."""
    rng = np.random.default_rng()
    # angles uniform on [0,2π)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    # radii with PDF ∝ r  →  CDF ∝ r^2  →  r = R * sqrt(u)
    u = rng.uniform(0, 1, size=n)
    r = r_max * np.sqrt(u)
    return [
        ParticleDisplacement(r=r_i, theta=theta_i)
        for r_i, theta_i in zip(r, theta, strict=True)
    ]


def sample_boltzmann_velocities(
    n: int, temperature: float, mass: float
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Sample N velocities from a Boltzmann distribution."""
    k_b = 1  # 1.380649e-23  # Boltzmann constant (J/K)

    sigma = np.sqrt(k_b * temperature / mass)
    rng = np.random.default_rng()
    velocities = rng.normal(loc=0, scale=sigma, size=(n))
    return np.linalg.norm(velocities, axis=1)


def sample_gaussian_velocities(
    n: int, average_velocity: float, std_velocity: float
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Sample N velocities from a Gaussian distribution."""
    rng = np.random.default_rng()
    return rng.normal(loc=average_velocity, scale=std_velocity, size=n)
