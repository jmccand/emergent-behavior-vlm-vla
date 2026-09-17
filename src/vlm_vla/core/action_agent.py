"""Abstract interface for the low-level ("fast control") action agent.

Any VLA can implement this: wrap it in a class with an `act` method and it can
be dropped into the orchestrator via config, with no other code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vlm_vla.core.types import Action, Observation, Plan


class ActionAgent(ABC):
    @abstractmethod
    def act(self, observation: Observation, plan: Plan) -> Action:
        """Produce the next low-level control command, conditioned on the current plan.

        Called every environment step.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any per-episode state (e.g. action chunk buffer). Optional to override."""
