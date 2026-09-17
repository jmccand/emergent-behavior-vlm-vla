"""Abstract interface for the environment the agent stack acts in.

Implementations are free to be a local sim, a client to a sim running elsewhere
(e.g. in Docker), or eventually a real-robot driver — the orchestrator only
depends on this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vlm_vla.core.types import Action, Observation


class Simulator(ABC):
    @abstractmethod
    def reset(self, task_id: int | None = None) -> Observation:
        """Start a new episode, optionally selecting a specific task by id."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """Apply an action and advance one timestep. Returns (obs, reward, done, info)."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any resources (connections, processes). Optional to override."""
