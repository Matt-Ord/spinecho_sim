"""Provide functions to convert between Majorana stars and quantum state coefficients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from spinecho_sim import (
    ParticleState,
    Solenoid,
)
from spinecho_sim.majorana._companion_helper import (
    majorana_stars,
)
from spinecho_sim.majorana._majorana_representation import (
    stars_to_state,
)
from spinecho_sim.state import (
    CoherentSpin,
    ParticleDisplacement,
    TrajectoryList,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _group_majorana_stars_by_index(
    spinor_states: np.ndarray[Any, np.dtype[np.complex128]],
    z_tol: float = 1e8,
) -> list[np.ndarray[Any, np.dtype[np.float64]]]:
    # Step 1: Get Majorana stars for each state
    stars_array = majorana_stars(
        spinor_states, z_tol=z_tol
    )  # shape: (n_states, n_stars, 2)

    _, n_stars, _ = stars_array.shape

    # Step 2: Group by star index
    return [stars_array[:, i, :] for i in range(n_stars)]


def _make_majorana_particle_groups(
    spinor_states: np.ndarray[Any, np.dtype[np.complex128]],
    displacements: Sequence[ParticleDisplacement],
    velocities: np.ndarray[Any, np.dtype[np.floating]],
    z_tol: float = 1e8,
) -> list[list[ParticleState]]:
    grouped_stars = _group_majorana_stars_by_index(spinor_states, z_tol=z_tol)
    all_groups: list[list[ParticleState]] = []
    n_groups = len(grouped_stars)
    n_states = grouped_stars[0].shape[0]

    for i in range(n_groups):
        group = grouped_stars[i]  # shape: (n_states, 2)
        group_states = [
            ParticleState(
                spin=CoherentSpin(theta=group[j, 0], phi=group[j, 1]),
                displacement=displacements[j],
                parallel_velocity=velocities[j],
            )
            for j in range(n_states)
        ]
        all_groups.append(group_states)
    return all_groups


def _simulate_majorana_groups(
    grouped_states: list[list[ParticleState]],
    solenoid: Solenoid,
    n_steps: int = 100,
) -> list[TrajectoryList]:
    """Simulate trajectories for each group of ParticleState objects and return a list of TrajectoryList objects."""
    all_trajectorylists: list[TrajectoryList] = []
    for group in grouped_states:
        # Simulate trajectories for each ParticleState in the group
        result = solenoid.simulate_trajectories(group, n_steps=n_steps)
        trajectory_list = result.trajectories
        all_trajectorylists.append(trajectory_list)
    return all_trajectorylists


def _reconstruct_spin_states_from_trajectories(
    grouped_trajectorylists: list[TrajectoryList],
    tol: float = 1e-10,
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    """
    Given a list of TrajectoryList objects (each for a Majorana star index), reconstruct the spin-J state coefficients for each initial state at every trajectory step.

    Returns an array of shape (n_states, n_steps, 2J+1) with the reconstructed spin-J coefficients
    for each state at each step along the trajectory.
    """
    n_groups = len(grouped_trajectorylists)  # number of Majorana stars (2J)
    n_states = len(grouped_trajectorylists[0])
    n_steps = len(
        grouped_trajectorylists[0][0].spins
    )  # assumes all trajectories have same length

    # Prepare array to hold Majorana stars for each state, step, and star index
    all_stars = np.empty((n_states, n_steps, n_groups, 2), dtype=np.float64)

    for i, trajectory_list in enumerate(grouped_trajectorylists):  # i: star index
        for j, trajectory in enumerate(trajectory_list):  # j: state index
            for k in range(n_steps):  # k: trajectory step
                spin = trajectory.spins[k]
                all_stars[j, k, i, 0] = spin.theta
                all_stars[j, k, i, 1] = spin.phi

    # For each state and each step, reconstruct the 2J+1 spin-J coefficients from the Majorana stars
    reconstructed_states = np.empty(
        (n_states, n_steps, n_groups + 1), dtype=np.complex128
    )
    for j in range(n_states):
        for k in range(n_steps):
            reconstructed_states[j, k] = stars_to_state(all_stars[j, k], tol=tol)

    return reconstructed_states


def simulate_trajectories_majorana(
    spinor_states: np.ndarray[Any, np.dtype[np.complex128]],
    displacements: Sequence[ParticleDisplacement],
    velocities: np.ndarray[Any],
    solenoid: Solenoid,
    n_steps: int = 100,
    tol: float = 1e-10,
) -> np.ndarray:
    grouped_states = _make_majorana_particle_groups(
        spinor_states, displacements, velocities
    )
    grouped_trajectories = _simulate_majorana_groups(
        grouped_states, solenoid, n_steps=n_steps
    )
    return _reconstruct_spin_states_from_trajectories(grouped_trajectories, tol=tol)
