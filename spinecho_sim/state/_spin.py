from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from functools import reduce
from typing import Any, Literal, cast, overload, override

import numpy as np
from scipy.special import comb  # type: ignore[import]

from spinecho_sim.measurement import transverse_expectation
from spinecho_sim.state._majorana import majorana_stars


def _get_polynomial_product(
    states: Spin[tuple[int]],
) -> np.ndarray[tuple[int], np.dtype[np.complexfloating]]:
    """
    Compute the coefficients of product polynomial.

    P(z) = ∏ (b_i - a_i z), returned as a vector of coefficients.
    """
    a = np.sin(states.theta / 2) * np.exp(1j * states.phi)
    b = -np.cos(states.theta / 2)
    return reduce(np.convolve, np.stack([a, b], axis=-1))[::-1]


def _majorana_polynomial_components(
    states: Spin[tuple[int]],
) -> np.ndarray[tuple[int], np.dtype[np.complexfloating]]:
    """
    Compute A_m using the polynomial representation.

    Returns
    -------
    A : np.ndarray, shape (N+1,)
        Coefficients A_m for m = -j to j
    """
    coefficients = _get_polynomial_product(states)
    k = np.arange(states.size + 1)
    binomial_weights = np.sqrt(np.asarray(comb(states.size, k), dtype=np.float64))
    state = coefficients / binomial_weights
    return state / np.linalg.norm(state)


class Spin[S: tuple[int, ...]](Sequence[Any]):  # noqa: PLR0904
    """A class representing a collection of lists of CoherentSpin objects."""

    def __init__[*S_](
        self: Spin[tuple[*S_]],  # type: ignore[override]
        spins: np.ndarray[tuple[*S_, int], np.dtype[np.float64]],  # type: ignore[override]
    ) -> None:
        self._spins = spins
        # Spins are stored as an ndarray of shape (..., 2)
        # Where spin[..., 0] is theta and spin[..., 1] is phi
        assert self._spins.shape[-1] == 2  # noqa: PLR2004
        assert self._spins.ndim > 1, "Spins must have at least 2 dimensions."

    @override
    def __eq__(self, value: object) -> bool:
        if isinstance(value, Spin):
            value = cast("Spin[tuple[int, ...]]", value)
            return np.array_equal(self.theta, value.theta) and np.array_equal(
                self.phi, value.phi
            )
        return False

    @override
    def __hash__(self) -> int:
        return hash((self.theta, self.phi))

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the spins array."""
        return self._spins.ndim - 1

    @override
    def __len__(self) -> int:
        """Total number of CoherentSpin objects."""
        return self.shape[0]

    @property
    def theta(self) -> np.ndarray[tuple[*S], np.dtype[np.floating]]:
        """Return the theta angle of the spin."""
        return self._spins[..., 0]

    @property
    def phi(self) -> np.ndarray[tuple[*S], np.dtype[np.floating]]:
        """Return the phi angle of the spin."""
        return self._spins[..., 1]

    @overload
    def __getitem__(self: Spin[tuple[int]], index: int) -> CoherentSpin: ...

    @overload
    def __getitem__[*S_](
        self: Spin[tuple[int, *S_]],  # type: ignore[override]
        index: int,
    ) -> Spin[tuple[*S_]]: ...  # type: ignore[override]

    @overload
    def __getitem__[*S_](
        self: Spin[tuple[int, *S_]],  # type: ignore[override]
        index: slice,
    ) -> Spin[tuple[int, *S_]]: ...  # type: ignore[override]

    @override
    def __getitem__(self, index: int | slice) -> CoherentSpin | Spin[tuple[Any, ...]]:
        """Get a single CoherentSpin object by index."""
        if isinstance(index, int) and self._spins.ndim == 2:  # noqa: PLR2004
            theta, phi = self._spins[index]
            return CoherentSpin(theta=theta, phi=phi)
        # If index is a slice, return a new Spin object with sliced data
        return Spin(self._spins[index])

    @override
    def __iter__[*S_](self: Spin[tuple[int, *S_]]) -> Iterator[Spin[tuple[*S_]]]:  # type: ignore[override]
        """Iterate over all CoherentSpin objects (flattened)."""
        for group in self._spins:
            yield from group

    def item(self, index: int) -> CoherentSpin:
        """Iterate over all CoherentSpin objects."""
        return CoherentSpin(theta=self.theta.item(index), phi=self.phi.item(index))

    def flat_iter(self) -> Iterator[CoherentSpin]:
        """Iterate over all CoherentSpin objects in a flat manner."""
        for i in range(self.size):
            yield self.item(i)

    @property
    def shape(self) -> tuple[*S]:
        """Return the shape of the spin list."""
        return self._spins.shape[:-1]  # type: ignore[return-value]

    @property
    def n_stars(self) -> int:
        """Return the number of components in each spin momentum state (e.g., 2J+1 for spin-J)."""
        return self.shape[-1]

    @property
    def size(self) -> int:
        """Return the total number of spins."""
        return np.prod(self.shape).item()

    @property
    def x(self) -> np.ndarray[tuple[*S], np.dtype[np.floating]]:
        """Get the x-component of the spin vector."""
        return np.sin(self.theta) * np.cos(self.phi)

    @property
    def y(self) -> np.ndarray[tuple[*S], np.dtype[np.floating]]:
        """Get the y-component of the spin vector."""
        return np.sin(self.theta) * np.sin(self.phi)

    @property
    def z(self) -> np.ndarray[tuple[*S], np.dtype[np.floating]]:
        """Get the z-component of the spin vector."""
        return np.cos(self.theta)

    @property
    def cartesian(self) -> np.ndarray[tuple[int, *S], np.dtype[np.floating]]:
        """Get the Cartesian coordinates of the spin vector."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @property
    def momentum_states[*S_](
        self: Spin[tuple[*S_, int]],  # type: ignore[override]
    ) -> np.ndarray[tuple[int, *S_], np.dtype[np.complex128]]:  # type: ignore[override]
        """Convert the spin representation to a momentum state."""
        # Flatten to (n_spins, n_stars, 2)
        stars = self._spins.reshape(-1, self.n_stars, 2)
        state_list = [
            _majorana_polynomial_components(Spin[tuple[int]](stars[i]))
            for i in range(stars.shape[0])
        ]
        # Undo the flattening and reshape to match the original shape
        return np.stack(state_list, axis=-1).reshape(-1, *self.shape[:-1])  # type: ignore[return-value]

    @staticmethod
    def from_momentum_state(
        spin_coefficients: np.ndarray[tuple[int], np.dtype[np.complex128]],
    ) -> Spin[tuple[int]]:
        """Create a Spin from a series of momentum states represented by complex coefficients.

        This function takes a list of spin coefficients
        ```python
        spin_coefficients[i,j]
        ```
        where i is the state index and j is the list index.

        """
        assert spin_coefficients.ndim == 1
        stars_array = majorana_stars(np.array([spin_coefficients]).T)
        return Spin(stars_array.reshape(-1, 2))

    @staticmethod
    def from_iter[S_: tuple[int, ...]](
        spins: Iterable[Spin[S_]],
    ) -> Spin[tuple[int, *S_]]:
        """Create a Spin from a nested list of CoherentSpin objects."""
        spins = list(spins)
        spins_array = np.array(
            [
                np.column_stack((spin.theta, spin.phi)).reshape(*spin.shape, 2)
                for spin in spins
            ],
            dtype=np.float64,
        )
        return Spin(spins_array)  # type: ignore[return-value]


def get_spin_expectation_values(
    state: np.ndarray[Any, np.dtype[np.complex128]],
) -> tuple[float, float, float]:
    # Compute expectation values from the momentum states of the input Spin
    jx, jy, jz = transverse_expectation(state)
    return jx, jy, jz


def expectation_values[*S_](
    spins: Spin[tuple[*S_, int]],  # type: ignore[override]
) -> np.ndarray[tuple[Literal[3], *S_], np.dtype[np.floating]]:  # type: ignore[override]
    """Get the expectation values of the spin."""
    momentum_states = spins.momentum_states
    momentum_states = momentum_states.reshape(momentum_states.shape[0], -1)
    expectation_values_list = [
        get_spin_expectation_values(momentum_states[:, i])
        for i in range(momentum_states.shape[1])
    ]
    return np.stack(expectation_values_list, axis=-1, dtype=np.float64).reshape(
        3, *spins.shape[:-1]
    )  # type: ignore[return-value]


class CoherentSpin(Spin[tuple[()]]):
    """A class representing a single coherent spin with theta and phi angles."""

    def __init__(
        self,
        theta: float,
        phi: float,
    ) -> None:
        self._theta = theta
        self._phi = phi

    @property
    @override
    def theta(self) -> np.ndarray[tuple[()], np.dtype[np.floating]]:
        """Return the theta angle of the spin."""
        return np.array(self._theta)

    @property
    @override
    def phi(self) -> np.ndarray[tuple[()], np.dtype[np.floating]]:
        """Return the phi angle of the spin."""
        return np.array(self._phi)

    @property
    @override
    def shape(self) -> tuple[()]:
        """Return the shape of a single coherent spin."""
        return ()

    def as_generic(self, *, n_stars: int = 1) -> GenericSpin:
        """Return a generic Spin representation of this coherent spin."""
        return Spin.from_iter((self,) * n_stars)

    @staticmethod
    def from_cartesian(x: float, y: float, z: float) -> CoherentSpin:
        """Create a Spin from Cartesian coordinates."""
        r = np.sqrt(x**2 + y**2 + z**2)
        assert np.isclose(r, 1, rtol=1e-3), (
            f"Spin vector must be normalized. r = {r}, inputs: x={x}, y={y}, z={z}"
        )
        return CoherentSpin(theta=np.arccos(z / r), phi=np.arctan2(y, x))


type GenericSpin = Spin[tuple[int]]
type GenericSpinList = Spin[tuple[int, int]]
type CoherentSpinList = Spin[tuple[int]]
