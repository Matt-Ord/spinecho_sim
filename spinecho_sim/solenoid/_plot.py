from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from matplotlib import pyplot as plt

from spinecho_sim.state import get_expectation_values
from spinecho_sim.util import Measure, get_figure, get_measure_label, measure_data

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]

    from spinecho_sim.solenoid._solenoid import (
        SolenoidSimulationResult,
    )
    from spinecho_sim.state._trajectory import (
        Trajectory,
        TrajectoryList,
    )


def plot_spin_state(
    result: SolenoidSimulationResult,
    idx: int,
    *,
    measure: Measure = "abs",
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    positions = result.positions
    states = result.spin.momentum_states[idx, :, :]
    state_measure = measure_data(states, measure)

    average_state_measure = np.average(state_measure, axis=0)

    n_stars = result.spin.n_stars
    s = n_stars / 2
    ms_values = np.linspace(s, -s, n_stars + 1, endpoint=True)
    ms_labels = [
        rf"$|m_S={int(m)} \rangle$"
        if m.is_integer()
        else rf"$|m_S={2 * m:.0f}/2 \rangle$"
        for m in ms_values
    ]

    # Plot phase
    (measure_line,) = ax.plot(positions, average_state_measure, label="Mean")
    color_measure = measure_line.get_color()
    ax.plot(
        positions,
        np.swapaxes(state_measure, 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color_measure,
    )

    # Standard error of the mean for phase
    std_states_measure = np.std(state_measure, axis=0) / np.sqrt(len(states))
    ax.fill_between(
        positions,
        (average_state_measure - std_states_measure).ravel(),
        (average_state_measure + std_states_measure).ravel(),
        alpha=0.2,
        linestyle=":",
        color=color_measure,
        label=r"Mean $\pm 1\sigma$",
    )

    ax.set_ylabel(f"{ms_labels[idx]} {get_measure_label(measure)}")
    ax.legend(loc="lower right")
    ax.set_xlim(positions[0], positions[-1])

    return fig, ax


def plot_state_intensity(
    result: SolenoidSimulationResult, idx: int, *, ax: Axes | None = None
) -> tuple[Figure | SubFigure, Axes, Line2D]:
    fig, ax = get_figure(ax)

    positions = result.positions
    states = result.spin.momentum_states[idx]
    average_state_abs = np.average(np.abs(states) ** 2, axis=0)

    (line,) = ax.plot(
        positions,
        average_state_abs,
        color="black",
        linestyle="--",
        label=r"$|m_S\rangle$ Intensity",
    )
    ax.set_ylabel(r"$|m_S\rangle$ Intensity")
    ax.set_xlim(positions[0], positions[-1])
    ax.legend(loc="center right")

    return fig, ax, line


def plot_spin_states(result: SolenoidSimulationResult) -> tuple[Figure, Axes]:
    n_stars = result.spin.n_stars
    fig, axes = plt.subplots(n_stars + 1, 2, figsize=(10, 6), sharex=True)

    for idx, (ax_abs, ax_arg) in enumerate(axes):
        plot_spin_state(result, idx, measure="abs", ax=ax_abs)
        plot_spin_state(result, idx, measure="arg", ax=ax_arg)
    for ax in axes[-1]:
        ax.set_xlabel(r"Distance $z$ along Solenoid Axis")
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title
    return fig, axes


def plot_expectation_value(
    result: SolenoidSimulationResult,
    idx: int,
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    positions = result.positions
    expectation_values = get_expectation_values(result.spin)[idx, :]

    average_state_measure = np.average(expectation_values, axis=0)
    labels = [
        r"\langle S_x \rangle",
        r"\langle S_y \rangle",
        r"\langle S_z \rangle",
    ]

    (measure_line,) = ax.plot(positions, average_state_measure)
    measure_line.set_label(rf"$\overline{{{labels[idx]}}} / \hbar$")
    color_measure = measure_line.get_color()
    ax.plot(
        positions,
        np.swapaxes(expectation_values, 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color_measure,
    )

    # Standard error of the mean for phase
    std_states_measure = np.std(expectation_values, axis=0) / np.sqrt(
        len(expectation_values)
    )
    ax.fill_between(
        positions,
        (average_state_measure - std_states_measure).ravel(),
        (average_state_measure + std_states_measure).ravel(),
        alpha=0.2,
        linestyle=":",
        color=color_measure,
        label=rf"$\overline{{{labels[idx]}}} / \hbar \pm 1\sigma$",
    )

    ax.set_ylabel(rf"${labels[idx]} / \hbar$")
    ax.legend(loc="center left")
    ax.set_xlim(positions[0], positions[-1])

    return fig, ax


def plot_expectation_values(
    result: SolenoidSimulationResult,
) -> tuple[Figure, Axes]:
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    for idx, ax in enumerate(axes):
        plot_expectation_value(result, idx, ax=ax)
    axes[-1].set_xlabel(r"Distance $z$ along Solenoid Axis")
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title
    return fig, axes


def plot_expectation_phi(
    result: SolenoidSimulationResult,
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    positions = result.positions
    expectation_values = get_expectation_values(result.spin)

    wrapped_phi = np.arctan2(
        expectation_values[1, :], expectation_values[0, :]
    )  # atan2(y, x) gives the angle in radians
    phi = np.unwrap(wrapped_phi, axis=1) / np.pi  # Unwrap and normalize to [0, 2π)

    average_phi = np.average(phi, axis=0)

    (average_line,) = ax.plot(positions, average_phi)
    average_line.set_label(r"$\overline{\langle \phi \rangle}$")
    color = average_line.get_color()

    ax.plot(
        positions,
        np.swapaxes(phi, 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color,
    )
    # Standard error of the mean
    std_spins = np.std(phi, axis=0) / np.sqrt(len(phi))
    ax.fill_between(
        positions,
        (average_phi - std_spins).ravel(),
        (average_phi + std_spins).ravel(),
        alpha=0.2,
        linestyle="--",
        color=color,
        label=r"$\overline{\langle \phi \rangle} \pm 1 \sigma$",
    )
    ax.legend(loc="upper right")
    ax.set_ylabel(r"$\langle \phi \rangle$ Azimuthal Angle (radians/$\pi$)")
    ax.set_xlim(positions[0], positions[-1])
    return fig, ax


def plot_expectation_theta(
    result: SolenoidSimulationResult,
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    positions = result.positions
    expectation_values = get_expectation_values(result.spin)

    wrapped_theta = np.arctan2(
        np.sqrt(expectation_values[0, :] ** 2 + expectation_values[1, :] ** 2),
        expectation_values[2, :],
    )
    theta = np.unwrap(wrapped_theta, axis=1) / np.pi

    average_theta = np.average(theta, axis=0)

    (average_line,) = ax.plot(positions, average_theta)
    average_line.set_label(r"$\overline{\langle \theta \rangle}$")
    color = average_line.get_color()

    ax.plot(
        positions,
        np.swapaxes(theta, 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color,
    )
    # Standard error of the mean
    std_spins = np.std(theta, axis=0) / np.sqrt(len(theta))
    ax.fill_between(
        positions,
        (average_theta - std_spins).ravel(),
        (average_theta + std_spins).ravel(),
        alpha=0.2,
        linestyle="--",
        color=color,
        label=r"$\overline{\langle \theta \rangle} \pm 1 \sigma$",
    )
    ax.legend(loc="upper left")
    ax.set_ylabel(r"$\langle \theta \rangle$ Polar Angle (radians/$\pi$)")
    ax.set_xlim(positions[0], positions[-1])
    return fig, ax


def plot_expectation_angles(
    result: SolenoidSimulationResult,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_expectation_theta(result, ax=ax)
    plot_expectation_phi(result, ax=ax.twinx())
    ax.set_xlabel(r"Distance $z$ along Solenoid Axis")
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title

    return fig, ax


def plot_expectation_trajectory(
    trajectory: Trajectory,
) -> tuple[Figure, Axes3D, Line2D]:
    fig = plt.figure(figsize=(6, 6))
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))

    expectations = get_expectation_values(trajectory.spin)

    # Plot the trajectory as a 3D curve
    (line,) = ax.plot(expectations[0, :], expectations[1, :], expectations[2, :])
    ax.set_xlabel(r"$\langle S_x \rangle$")
    ax.set_ylabel(r"$\langle S_y \rangle$")
    ax.set_zlabel(r"$\langle S_z \rangle$")
    return fig, ax, line


def plot_expectation_trajectories(
    trajectories: TrajectoryList,
) -> tuple[Figure, Axes3D]:
    fig = plt.figure(figsize=(8, 8))
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))

    expectations = get_expectation_values(trajectories.spin)
    # Average over samples (axis=1), shape: (3, n_positions)
    avg_expectations = np.average(expectations, axis=1)

    # Unpack components
    x = avg_expectations[0, :]
    y = avg_expectations[1, :]
    z = avg_expectations[2, :]

    # Plot the trajectory as a 3D curve
    (average_line,) = ax.plot(x, y, z)
    average_line.set_label(r"$\overline{\langle \mathbf{S} \rangle}$")
    color = average_line.get_color()
    for trajectory in trajectories:
        _, _, line = plot_expectation_trajectory(trajectory)
        line.set_alpha(0.1)
        line.set_color(color)

    ax.set_xlabel(r"$\langle S_x \rangle$")
    ax.set_ylabel(r"$\langle S_y \rangle$")
    ax.set_zlabel(r"$\langle S_z \rangle$")
    ax.legend()
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title
    return fig, ax
