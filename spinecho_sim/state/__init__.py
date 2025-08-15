"""Module for representing and manipulating spin states.

In this package, spin states are represented by the `Spin` class,
which encapsulates the properties of a spin.
An example of its useage can be seen below.

.. literalinclude:: ../../examples/spin_representation.py
    :language: python
    :lineno-start: 8
    :lines: 8-43
    :dedent: 4

"""

from __future__ import annotations

from spinecho_sim.state._displacement import (
    ParticleDisplacement,
    ParticleDisplacementList,
)
from spinecho_sim.state._samples import (
    sample_gaussian_velocities,
    sample_uniform_displacement,
)
from spinecho_sim.state._spin import (
    CoherentSpin,
    CoherentSpinList,
    EmptySpin,
    EmptySpinList,
    EmptySpinListList,
    GenericSpin,
    GenericSpinList,
    Spin,
    get_bargmann_expectation_values,
    get_expectation_values,
)
from spinecho_sim.state._state import (
    MonatomicParticleState,
    ParticleState,
    StateVectorParticleState,
)
from spinecho_sim.state._trajectory import (
    MonatomicTrajectory,
    MonatomicTrajectoryList,
    StateVectorTrajectory,
    StateVectorTrajectoryList,
    Trajectory,
    TrajectoryList,
)

__all__ = [
    "CoherentSpin",
    "CoherentSpinList",
    "EmptySpin",
    "EmptySpinList",
    "EmptySpinListList",
    "GenericSpin",
    "GenericSpinList",
    "MonatomicParticleState",
    "MonatomicTrajectory",
    "MonatomicTrajectoryList",
    "ParticleDisplacement",
    "ParticleDisplacementList",
    "ParticleState",
    "Spin",
    "StateVectorParticleState",
    "StateVectorTrajectory",
    "StateVectorTrajectoryList",
    "Trajectory",
    "TrajectoryList",
    "get_bargmann_expectation_values",
    "get_expectation_values",
    "sample_gaussian_velocities",
    "sample_uniform_displacement",
]
