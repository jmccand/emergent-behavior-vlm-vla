"""Abstract interface for the high-level ("slow thinking") reasoning agent.

Any VLM can implement this: wrap it in a class with a `plan` method and it can
be dropped into the orchestrator via config, with no other code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vlm_vla.core.types import Observation, Plan


class ReasoningAgent(ABC):
    @abstractmethod
    def plan(self, observation: Observation, task: str, history: list[Plan]) -> Plan:
        """Produce the next subgoal given the current observation and plan history.

        Called once at episode start and again every `replan_every` steps
        (see `vlm_vla.orchestrator`), not on every environment step.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any per-episode state (e.g. conversation history). Optional to override."""
