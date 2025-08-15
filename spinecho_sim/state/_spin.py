from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from functools import cache, reduce
from typing import Any, Literal, cast, overload, override

import numpy as np
from scipy.special import comb  # type: ignore[import]

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


def bargmann_inner_product(
    a: np.ndarray[tuple[int], np.dtype[np.complex128]],
    b: np.ndarray[tuple[int], np.dtype[np.complex128]],
) -> np.complexfloating:
    """Compute the inner product of two polynomials in z."""
    two_j = len(a) - 1
    k = np.arange(two_j + 1)
    w = 1 / np.asarray(comb(two_j, k), dtype=np.complex128)
    return np.vdot(a, b * w)  # vdot = conjugate(a)·b


def _polynomial_z_derivative(
    a: np.ndarray[tuple[int], np.dtype[np.complex128]],
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    k = np.arange(len(a) - 1)
    da = (k + 1) * a[k + 1]
    return np.concatenate((da, np.zeros(1, dtype=np.complex128)))


def _polynomial_z_multiplication(
    a: np.ndarray[tuple[int], np.dtype[np.complex128]], shift: int = 1
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    return np.concatenate((np.zeros(shift, dtype=np.complex128), a[:-shift]))


def dicke_to_poly(
    c: np.ndarray[tuple[int], np.dtype[np.complex128]],
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    two_j = len(c) - 1
    k = np.arange(two_j + 1)
    return np.sqrt(np.asarray(comb(two_j, k), dtype=np.float64)) * c[two_j - k]


def _s_minus(
    a: np.ndarray[tuple[int], np.dtype[np.complex128]],
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    return _polynomial_z_derivative(a)  # Eq. (1) rightmost


def _s_plus(
    a: np.ndarray[tuple[int], np.dtype[np.complex128]],
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    two_j = len(a) - 1
    term1 = -_polynomial_z_multiplication(
        _polynomial_z_derivative(a), shift=2
    )  # -ħ z^2 dP/dz
    term2 = two_j * _polynomial_z_multiplication(a, shift=1)  # +2ħj z P
    return term1 + term2


def _s_z(
    a: np.ndarray[tuple[int], np.dtype[np.complex128]],
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    j = (len(a) - 1) / 2
    term = _polynomial_z_multiplication(_polynomial_z_derivative(a), shift=1)  # z dP/dz
    return term - j * a


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
            [np.stack([spin.theta, spin.phi], axis=-1) for spin in spins],
            dtype=np.float64,
        )
        return Spin(spins_array)  # type: ignore[return-value]


@cache
def _j_plus_factors(two_j: int) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Return a sparse array of J_+ ladder factors."""
    j = two_j / 2
    m = np.arange(-j, j)  # length 2j   (stops at j-1)
    return np.sqrt((j - m) * (j + m + 1))


def _get_expectation(
    state_coefficients: np.ndarray[Any, np.dtype[np.complex128]],
) -> tuple[float, float, float]:
    """Return the expectation values of S_x, S_y, and S_z for a given state vector using cached arrays."""
    state_coefficients /= np.linalg.norm(state_coefficients)  # Normalize the state
    two_j = state_coefficients.size - 1
    factors = _j_plus_factors(two_j)  # sparse array

    inner = np.conjugate(state_coefficients[:-1]) * state_coefficients[1:] * factors
    j_plus = inner.sum()

    jx = -float(j_plus.real)
    jy = -float(j_plus.imag)

    m_z = np.arange(two_j / 2, -two_j / 2 - 1, -1, dtype=np.float64)
    jz = float(np.sum(np.abs(state_coefficients) ** 2 * m_z))
    return jx, jy, jz


def _get_bargmann_expectation(
    state_coefficients: np.ndarray[Any, np.dtype[np.complex128]],
) -> tuple[float, float, float]:
    """Return the expectation values of S_x, S_y, and S_z for a given state vector using bargmann representation operators."""
    polynomial_coefficients = dicke_to_poly(state_coefficients)

    # operator actions
    a_minus = _s_minus(polynomial_coefficients)
    a_plus = _s_plus(polynomial_coefficients)
    a_z = _s_z(polynomial_coefficients)

    sx = 0.5 * bargmann_inner_product(polynomial_coefficients, a_plus + a_minus)
    sy = -0.5j * bargmann_inner_product(polynomial_coefficients, a_plus - a_minus)
    sx *= -1  # Aligns with convention of other code
    sy *= -1
    sz = bargmann_inner_product(polynomial_coefficients, a_z)
    return float(sx.real), float(sy.real), float(sz.real)


def get_expectation_values[*S_](
    spins: Spin[tuple[*S_, int]],  # type: ignore[override]
) -> np.ndarray[tuple[Literal[3], *S_], np.dtype[np.floating]]:  # type: ignore[override]
    """Get the expectation values of the spin.

    Returns an array of shape (3, *spins.shape) where the first dimension corresponds to
    the expectation values for S_x, S_y, and S_z.
    """
    momentum_states = spins.momentum_states
    momentum_states = momentum_states.reshape(momentum_states.shape[0], -1)
    expectation_values_list = [
        _get_expectation(momentum_states[:, i]) for i in range(momentum_states.shape[1])
    ]
    return np.stack(expectation_values_list, axis=-1, dtype=np.float64).reshape(
        3, *spins.shape[:-1]
    )  # type: ignore[return-value]


def get_bargmann_expectation_values[*S_](
    spins: Spin[tuple[*S_, int]],  # type: ignore[override]
) -> np.ndarray[tuple[Literal[3], *S_], np.dtype[np.floating]]:  # type: ignore[override]
    """Get the expectation values of the spin.

    Returns an array of shape (3, *spins.shape) where the first dimension corresponds to
    the expectation values for S_x, S_y, and S_z.
    """
    momentum_states = spins.momentum_states
    momentum_states = momentum_states.reshape(momentum_states.shape[0], -1)
    expectation_values_list = [
        _get_bargmann_expectation(momentum_states[:, i])
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


class EmptySpin(Spin[tuple[int]]):
    """Represents the absence of rotational angular momentum with zero Majorana stars."""

    def __init__(self) -> None:
        """Initialize EmptySpin with zero Majorana stars."""
        # Create an array with zero rows and 2 columns
        spins = np.zeros((0, 2), dtype=np.float64)
        super().__init__(spins=spins)

    @property
    @override
    def n_stars(self) -> int:
        """Override the number of Majorana stars to return zero."""
        return 0

    @override
    def flat_iter(self) -> Iterator[CoherentSpin]:
        """Return an empty iterator since there are no Majorana stars."""
        return iter([])


class EmptySpinList(Spin[tuple[int, int]]):
    """Represents a list of empty spins with zero Majorana stars."""

    def __init__(self, shape: tuple[int, int]) -> None:
        """Initialize EmptySpinList with a given number of empty spins."""
        # Create an array with n_rows rows, n_cols columns, and 2 spin components, filled with zeros
        spins = np.zeros(
            (shape[0], shape[1], 2), dtype=np.float64
        )  # Match the expected 3D shape
        super().__init__(spins=spins)

    @property
    @override
    def n_stars(self) -> int:
        """Override the number of Majorana stars to return zero."""
        return 0

    @override
    def flat_iter(self) -> Iterator[CoherentSpin]:
        """Return an iterator over all empty spins."""
        return EmptySpin().flat_iter()


class EmptySpinListList(Spin[tuple[int, int, int]]):
    """Represents the absence of rotational angular momentum with zero Majorana stars in a 3D structure."""

    def __init__(self, shape: tuple[int, int, int]) -> None:
        """Initialize EmptySpin3D with a given 3D shape."""
        # Create an array with the specified shape and 2 spin components, filled with zeros
        spins = np.zeros((*shape, 2), dtype=np.float64)
        super().__init__(spins=spins)

    @property
    @override
    def n_stars(self) -> int:
        """Override the number of Majorana stars to return zero."""
        return 0

    @override
    def flat_iter(self) -> Iterator[CoherentSpin]:
        """Return an iterator over all empty spins."""
        return EmptySpin().flat_iter()  # No Majorana stars, so return an empty iterator
