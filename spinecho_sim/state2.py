from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, overload, override

import numpy as np

from spinecho_sim.majorana import majorana_stars, stars_to_state


class Spin:
    """A class representing a spin with theta and phi angles."""

    _majorana_theta: np.ndarray[tuple[int], np.dtype[np.float64]]
    _majorana_phi: np.ndarray[tuple[int], np.dtype[np.float64]]

    def __init__(
        self,
        majorana_theta: np.ndarray[Any, np.dtype[np.float64]],
        majorana_phi: np.ndarray[Any, np.dtype[np.float64]],
    ) -> None:
        self._majorana_theta = majorana_theta.ravel()
        self._majorana_phi = majorana_phi.ravel()

    def as_momentum_state(self) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        """Convert the majorana star representation to a momentum state."""
        stars_array = np.column_stack(
            (self._majorana_theta, self._majorana_phi)
        ).astype(np.float64)
        state = stars_to_state(stars_array)
        return np.asarray(state, dtype=np.complex128)

    @property
    def n_stars(self) -> int:
        """Get the number of majorana stars represented by this spin."""
        return len(self._majorana_theta)

    @property
    def majorana_theta(self) -> np.ndarray:
        """Get the polar angles of the majorana stars."""
        return self._majorana_theta

    @property
    def majorana_phi(self) -> np.ndarray:
        """Get the azimuthal angles of the majorana stars."""
        return self._majorana_phi

    def as_majorana_stars(self) -> list[CoherentSpin]:
        """Get the Majorana stars as a 2D array of shape (n_stars, 2)."""
        return [
            CoherentSpin(theta=theta, phi=phi, n_stars=1)
            for theta, phi in zip(self._majorana_theta, self._majorana_phi, strict=True)
        ]

    @staticmethod
    def from_momentum_state(
        momentum: np.ndarray[tuple[int], np.dtype[np.complex128]],
    ) -> Spin:
        """Create a Spin from a momentum state."""
        stars = majorana_stars(momentum)  # shape: (n_states, n_stars, 2)
        # Expect only one state, use stars[0]
        theta = stars[0, :, 0]
        phi = stars[0, :, 1]
        return Spin(
            majorana_theta=np.array(theta, dtype=np.float64),
            majorana_phi=np.array(phi, dtype=np.float64),
        )

    @staticmethod
    def from_majorana_stars(stars: list[CoherentSpin]) -> Spin:
        """Create a Spin from Majorana stars."""
        return Spin(
            majorana_theta=np.array(
                [(star.theta,) * star.n_stars for star in stars]
            ).ravel(),
            majorana_phi=np.array(
                [(star.phi,) * star.n_stars for star in stars]
            ).ravel(),
        )


class CoherentSpin(Spin):
    """A class representing a coherent spin with theta and phi angles."""

    def __init__(
        self,
        theta: float,
        phi: float,
        n_stars: int = 1,
    ) -> None:
        self._n_stars = n_stars
        self._theta = theta
        self._phi = phi

    @override
    def as_momentum_state(self) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        """Convert the coherent spin representation to a momentum state."""
        thetas = np.full(self._n_stars, self._theta, dtype=np.float64)
        phis = np.full(self._n_stars, self._phi, dtype=np.float64)
        stars_array = np.column_stack((thetas, phis))
        state = stars_to_state(stars_array)
        return np.asarray(state, dtype=np.complex128)

    @override
    def as_majorana_stars(self) -> list[CoherentSpin]:
        """Get the Majorana stars as a list of CoherentSpin."""
        return [CoherentSpin(self._theta, self._phi, n_stars=self._n_stars)]

    @property
    @override
    def n_stars(self) -> int:
        """Get the number of majorana stars represented by this coherent spin."""
        return self._n_stars

    @property
    def theta(self) -> float:
        """Get the polar angle of the spin vector."""
        return self._theta

    @property
    def phi(self) -> float:
        """Get the azimuthal angle of the spin vector."""
        return self._phi

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
    def cartesian(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """Get the Cartesian coordinates of the spin vector."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @staticmethod
    def from_cartesian(
        x: float, y: float, z: float, *, n_stars: int = 1
    ) -> CoherentSpin:
        """Create a Spin from Cartesian coordinates."""
        r = np.sqrt(x**2 + y**2 + z**2)
        assert np.isclose(r, 1, rtol=1e-3), f"Spin vector must be normalized. r = {r}"
        return CoherentSpin(
            theta=np.arccos(z / r), phi=np.arctan2(y, x), n_stars=n_stars
        )


@dataclass(kw_only=True, frozen=True)
class SpinList(Sequence[Spin]):
    """A class representing a list of Spin objects."""

    _majorana_theta: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    _majorana_phi: np.ndarray[tuple[int, int], np.dtype[np.float64]]

    def __post_init__(self) -> None:
        if self._majorana_theta.shape != self._majorana_phi.shape:
            msg = "Theta and phi arrays must have the same shape."
            raise ValueError(msg)

    @staticmethod
    def from_spins(spins: Iterable[Spin]) -> SpinList:
        """Create a SpinList from an iterable of Spin objects."""
        theta = np.array([spin.majorana_theta for spin in spins], dtype=np.float64)
        phi = np.array([spin.majorana_phi for spin in spins], dtype=np.float64)
        return SpinList(_majorana_theta=theta, _majorana_phi=phi)

    @override
    def __len__(self) -> int:
        """Get the number of spins in the list."""
        return self._majorana_theta.shape[0]

    @overload
    def __getitem__(self, index: int) -> Spin: ...

    @overload
    def __getitem__(self, index: slice) -> SpinList: ...

    @override
    def __getitem__(self, index: int | slice) -> Spin | SpinList:
        if isinstance(index, slice):
            return SpinList(
                _majorana_theta=self._majorana_theta[index],
                _majorana_phi=self._majorana_phi[index],
            )
        return Spin(
            majorana_theta=np.array(
                [self._majorana_theta.item(index)], dtype=np.float64
            ),
            majorana_phi=np.array([self._majorana_phi.item(index)], dtype=np.float64),
        )

    @override
    def __iter__(self) -> Iterator[Spin]:
        """Iterate over the spins in the list."""
        for i in range(len(self)):
            yield Spin(
                majorana_theta=np.array([self._majorana_theta[i]], dtype=np.float64),
                majorana_phi=np.array([self._majorana_phi[i]], dtype=np.float64),
            )

    @property
    def majorana_theta(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Get the polar angles of the majorana stars."""
        return self._majorana_theta

    @property
    def majorana_phi(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Get the azimuthal angles of the majorana stars."""
        return self._majorana_phi

    @property
    def n_stars(self) -> int:
        """Get the number of majorana stars represented by the first spin."""
        if len(self) == 0:
            return 0
        return self[0].n_stars

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the majorana stars."""
        return self._majorana_theta.shape

    def as_majorana_stars(self) -> list[CoherentSpin]:
        """Get the Majorana stars as a list of CoherentSpin."""
        return [
            CoherentSpin(theta=theta, phi=phi, n_stars=self.n_stars)
            for theta, phi in zip(
                self._majorana_theta.ravel(), self._majorana_phi.ravel(), strict=True
            )
        ]


@dataclass(kw_only=True, frozen=True)
class CoherentSpinList(Sequence[CoherentSpin]):
    """A class representing a list of CoherentSpin objects."""

    theta: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    phi: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    n_stars: np.ndarray[tuple[int], np.dtype[np.int_]]

    def __post_init__(self) -> None:
        if self.theta.shape != self.phi.shape:
            msg = "Theta and phi arrays must have the same shape."
            raise ValueError(msg)
        if self.theta.shape[0] != self.n_stars.shape[0]:
            msg = "Theta and phi arrays must have the same number of spins as n_stars."
            raise ValueError(msg)

    @staticmethod
    def from_coherent_spins(spins: Iterable[CoherentSpin]) -> CoherentSpinList:
        """Create a CoherentSpinList from an iterable of CoherentSpin objects."""
        theta = np.array([spin.theta for spin in spins], dtype=np.float64)
        phi = np.array([spin.phi for spin in spins], dtype=np.float64)
        n_stars = np.array([spin.n_stars for spin in spins], dtype=np.int_)
        return CoherentSpinList(theta=theta, phi=phi, n_stars=n_stars)

    @override
    def __len__(self) -> int:
        """Get the number of coherent spins in the list."""
        return self.theta.shape[0]

    @overload
    def __getitem__(self, index: int) -> CoherentSpin: ...

    @overload
    def __getitem__(self, index: slice) -> CoherentSpinList: ...

    @override
    def __getitem__(self, index: int | slice) -> CoherentSpin | CoherentSpinList:
        if isinstance(index, slice):
            return CoherentSpinList(
                theta=self.theta[index],
                phi=self.phi[index],
                n_stars=self.n_stars[index],
            )
        return CoherentSpin(
            theta=self.theta.item(index),
            phi=self.phi.item(index),
            n_stars=self.n_stars.item(index),
        )

    @override
    def __iter__(self) -> Iterator[CoherentSpin]:
        """Iterate over the coherent spins in the list."""
        for i in range(len(self)):
            yield CoherentSpin(
                theta=self.theta.item(i),
                phi=self.phi.item(i),
                n_stars=self.n_stars.item(i),
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the coherent spins."""
        return self.theta.shape

    def as_momentum_states(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        """Convert the coherent spin representations to momentum states."""
        states: list[np.ndarray[Any, np.dtype[np.complex128]]] = []
        for spin in self:
            # Use the as_momentum_state method of CoherentSpin
            state = spin.as_momentum_state()
            states.append(state)
        return np.stack(states)
