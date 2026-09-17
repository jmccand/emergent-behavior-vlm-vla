"""Shared data types passed between the simulator, reasoning agent, and action agent.

These are the only objects the three subsystems agree on. Keeping them as plain
dataclasses (not tied to any model or sim library) is what lets any of the three
be swapped independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Observation:
    """A single timestep of simulator output."""

    images: dict[str, np.ndarray]
    """Camera name -> HWC uint8 RGB array, e.g. {"image": agentview, "image2": wrist}."""

    state: np.ndarray | None
    """Proprioceptive state (e.g. end-effector pose + gripper), if the sim provides one."""

    task_description: str
    """Fixed natural-language description of the overall task, from the simulator."""

    step: int
    """Steps elapsed since the last reset."""

    done: bool
    info: dict = field(default_factory=dict)


@dataclass
class Plan:
    """Output of the reasoning agent; input (alongside Observation) to the action agent.

    `instruction` is the only field the action agent is required to use. Everything
    else is for logging/inspection. A future latent-conditioned action agent can add
    a `latent: np.ndarray | None` field here without changing this contract.
    """

    instruction: str
    reasoning: str | None
    step_issued: int
    metadata: dict = field(default_factory=dict)


@dataclass
class Action:
    """A single low-level control command."""

    array: np.ndarray
    """Continuous control vector, e.g. 7-dim: 6D end-effector delta + 1D gripper."""


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    total_reward: float
    plans: list[Plan]
