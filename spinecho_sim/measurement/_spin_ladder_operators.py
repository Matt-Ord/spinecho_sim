from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def transverse_expectation(
    c: NDArray[np.complex128], hbar: float = 1.0
) -> tuple[float, float, float]:
    """Return the expectation values of S_x, S_y, and S_z for a given state vector c."""
    two_j = c.size - 1
    j = two_j / 2

    m = np.arange(-j, j)  # length 2j   (stops at j-1)
    factors = np.sqrt((j - m) * (j + m + 1))  # length 2j
    inner = np.conjugate(c[:-1]) * c[1:] * factors

    j_plus = hbar * inner.sum()
    jx = float(j_plus.real)
    jy = float(j_plus.imag)

    # S_z operator
    m_z = np.arange(-j, j + 1)
    jz = float(hbar * np.sum(np.abs(c) ** 2 * m_z[::-1]))

    return jx, jy, jz
