from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import comb  # type: ignore[import]

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _majorana_precompute(
    two_j: int,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    k = np.arange(two_j + 1, dtype=int)  # k = 0, 1, ..., 2J
    w = np.sqrt(np.asarray(comb(two_j, k), dtype=np.float64))  # √(2J choose k)

    # Hessenberg companion template for the monic polynomial
    c_template = np.zeros((two_j, two_j), dtype=np.complex128)
    c_template[1:, :-1] = np.eye(two_j - 1, dtype=np.complex128)

    return w, c_template


def majorana_stars(
    spin_coefficients: np.ndarray[Any, np.dtype[np.complex128]], z_tol: float = 1e8
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compute Majorana points for multiple sets of spin coefficients."""
    # Calculate j from the length of the spin vector
    two_j = spin_coefficients.shape[1] - 1  # Spin-j vector has 2j+1 coefficients
    w, c0 = _majorana_precompute(two_j)
    points_list = _majorana_roots(spin_coefficients, w, c0)

    # Take absolute value and argument for all points
    abs_z = np.abs(points_list)
    angle_z = np.angle(points_list)

    theta = np.where(abs_z > z_tol, np.pi, 2 * np.arctan(abs_z))
    phi = np.where(abs_z > z_tol, 0, angle_z % (2 * np.pi))

    all_stars = np.empty((len(points_list), two_j, 2), dtype=np.float64)
    for i in range(len(points_list)):
        stars = np.column_stack((theta[i], phi[i]))  # shape (n_points, 2)
        while stars.shape[0] < two_j:
            stars = np.vstack((stars, [np.pi, 0.0]))
        all_stars[i, :, :] = stars
    return all_stars  # shape (n_states, two_j, 2)


def _majorana_roots(
    spin_coefficients: NDArray[np.complex128],  # c_m  with m = -J…J, length 2J+1
    w: NDArray[np.float64],  # from majorana_precompute
    c_template: NDArray[np.complex128],  # from majorana_precompute
) -> NDArray[np.complex128]:
    two_j = len(w) - 1  # = 2J

    roots_array = np.empty((spin_coefficients.shape[0], two_j), dtype=np.complex128)
    for idx, coefficients in enumerate(spin_coefficients):
        if len(coefficients) != (two_j + 1):
            msg = "coefficients must have length 2J+1."
            raise ValueError(msg)

        # ---- step (i) : polynomial coefficients (highest power first) ----
        p = w * coefficients[::-1]

        # --  peel off vanishing highest powers -----------------------------
        m_inf = 0
        while len(p) > 1 and np.isclose(p[-1], 0.0):
            p = p[:-1]
            m_inf += 1

        # --  all roots at infinity?  (state lives entirely in +z subspace) --
        if len(p) == 1:
            roots = np.full(m_inf, np.inf, dtype=np.complex128)
        else:
            c_monic = p[:-1] / p[-1]
            n_eff = len(c_monic)
            c = c_template[:n_eff, :n_eff].copy()
            c[0, :] = -c_monic[::-1]
            if n_eff > 1:
                c[1:, :-1] = np.eye(n_eff - 1)
            roots = np.linalg.eigvals(c)
            if m_inf:
                roots = np.concatenate(
                    [roots, np.full(m_inf, np.inf, dtype=np.complex128)]
                )
        roots_array[idx, :] = roots
    return roots_array
