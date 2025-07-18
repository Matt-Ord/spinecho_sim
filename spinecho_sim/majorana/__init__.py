"""Module for simulating and plotting solenoid magnetic fields and particle trajectories."""

from __future__ import annotations

from spinecho_sim.majorana._companion_helper import (
    majorana_stars,
)
from spinecho_sim.majorana._majorana import (
    simulate_trajectories_majorana,
)
from spinecho_sim.majorana._majorana_representation import (
    stars_to_state,
    stars_to_states,
)

__all__ = [
    "majorana_stars",
    "simulate_trajectories_majorana",
    "stars_to_state",
    "stars_to_states",
]
