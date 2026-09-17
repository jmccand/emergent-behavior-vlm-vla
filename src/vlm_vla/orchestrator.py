"""The dual-system control loop: slow reasoning, fast action.

This is the only place that knows both agents exist. It has no knowledge of
Qwen, SmolVLA, or LIBERO specifically -- it only talks to the three ABCs in
`vlm_vla.core`, so any backend combination selected via config runs through
the same loop unmodified.
"""

from __future__ import annotations

import logging

from vlm_vla.core import (
    Action,
    ActionAgent,
    EpisodeRecorder,
    EpisodeResult,
    Plan,
    ReasoningAgent,
    Simulator,
)

logger = logging.getLogger(__name__)


class HierarchicalAgent:
    def __init__(
        self,
        reasoning_agent: ReasoningAgent,
        action_agent: ActionAgent,
        simulator: Simulator,
        replan_every: int = 30,
    ) -> None:
        self.reasoning_agent = reasoning_agent
        self.action_agent = action_agent
        self.simulator = simulator
        self.replan_every = replan_every

    def run_episode(
        self,
        task_id: int | None = None,
        max_steps: int = 300,
        recorder: EpisodeRecorder | None = None,
    ) -> EpisodeResult:
        self.reasoning_agent.reset()
        self.action_agent.reset()

        obs = self.simulator.reset(task_id)
        if recorder is not None:
            recorder.record_frame(obs)

        plans: list[Plan] = []
        total_reward = 0.0
        success = False

        plan = self.reasoning_agent.plan(obs, obs.task_description, history=plans)
        plans.append(plan)
        logger.info("step=%d new plan: %r", obs.step, plan.instruction)
        if recorder is not None:
            recorder.record_plan(plan)

        for t in range(max_steps):
            if t > 0 and t % self.replan_every == 0:
                plan = self.reasoning_agent.plan(obs, obs.task_description, history=plans)
                plans.append(plan)
                logger.info("step=%d new plan: %r", obs.step, plan.instruction)
                if recorder is not None:
                    recorder.record_plan(plan)

            action: Action = self.action_agent.act(obs, plan)
            obs, reward, done, info = self.simulator.step(action)
            total_reward += reward
            if recorder is not None:
                recorder.record_step(obs, plan, action, reward, done, info)

            if done:
                success = bool(info.get("success", False))
                break

        return EpisodeResult(
            success=success,
            steps=obs.step,
            total_reward=total_reward,
            plans=plans,
        )
