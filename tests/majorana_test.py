from __future__ import annotations

import numpy as np

from spinecho_sim.majorana import (
    majorana_stars,
    stars_to_states,
)


def test_spin_states_roundtrip() -> None:
    spin_states = np.array(
        [
            [(1.0 + 0.0j) / np.sqrt(2), (0.0 + 0.0j), (1.0 + 0.0j) / np.sqrt(2)],
            [(0.0 + 0.0j), (1.0 + 0.0j) / np.sqrt(2), (1.0 + 0.0j) / np.sqrt(2)],
        ],
    )
    # Convert to Majorana points
    majorana_points = majorana_stars(spin_states)
    # Convert back to states
    recovered_states = stars_to_states(majorana_points)
    # Check if recovered states match original (up to global phase)
    for original, recovered in zip(spin_states, recovered_states, strict=False):
        # Normalize both
        original_norm = original / np.linalg.norm(original)
        recovered_norm = recovered / np.linalg.norm(recovered)
        # Remove global phase
        phase: np.complex128 = np.exp(
            -1j * np.angle(np.vdot(original_norm, recovered_norm))
        )
        recovered_norm *= phase
        np.testing.assert_allclose(
            original_norm,
            recovered_norm,
            atol=1e-8,
            err_msg=f"Round-trip failed: {original_norm} vs {recovered_norm}",
        )
