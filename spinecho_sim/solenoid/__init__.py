"""Module for simulating and plotting solenoid magnetic fields and particle trajectories."""

from __future__ import annotations

from spinecho_sim.solenoid._plot import (
    plot_expectation_angles,
    plot_expectation_trajectories,
    plot_expectation_trajectory,
    plot_expectation_values,
    plot_spin_states,
)
from spinecho_sim.solenoid._solenoid import (
    MonatomicSolenoid,
    MonatomicSolenoidSimulationResult,
    MonatomicSolenoidTrajectory,
    Solenoid,
    SolenoidSimulationResult,
    SolenoidTrajectory,
)

__all__ = [
    "MonatomicSolenoid",
    "MonatomicSolenoidSimulationResult",
    "MonatomicSolenoidTrajectory",
    "Solenoid",
    "SolenoidSimulationResult",
    "SolenoidTrajectory",
    "plot_expectation_angles",
    "plot_expectation_trajectories",
    "plot_expectation_trajectory",
    "plot_expectation_values",
    "plot_spin_states",
]
