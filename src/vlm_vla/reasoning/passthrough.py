"""A trivial reasoning agent that echoes the simulator's ground-truth task
description instead of generating subgoals -- used to run the action agent
(VLA) alone, without loading any VLM, as an apples-to-apples baseline against
what the underlying VLA's own paper reports.
"""

from __future__ import annotations

from vlm_vla.core.reasoning_agent import ReasoningAgent
from vlm_vla.core.types import Observation, Plan


class PassthroughReasoningAgent(ReasoningAgent):
    def plan(self, observation: Observation, task: str, history: list[Plan]) -> Plan:
        return Plan(instruction=task, reasoning=None, step_issued=observation.step)
