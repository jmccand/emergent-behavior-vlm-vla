#!/usr/bin/env python
"""Run one or more LIBERO episodes through the full VLM -> VLA hierarchical loop.

Example:
    python scripts/run_episode.py --config configs/default.yaml \
        --task-id 0 --episodes 1
"""

from __future__ import annotations

import argparse
import logging

from vlm_vla.config import build_action_agent, build_reasoning_agent, build_simulator, load_config
from vlm_vla.orchestrator import HierarchicalAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)

    reasoning_agent = build_reasoning_agent(cfg.reasoning)
    action_agent = build_action_agent(cfg.action)
    simulator = build_simulator(cfg.simulator)

    agent = HierarchicalAgent(
        reasoning_agent=reasoning_agent,
        action_agent=action_agent,
        simulator=simulator,
        replan_every=cfg.orchestrator.replan_every,
    )

    for episode in range(args.episodes):
        result = agent.run_episode(task_id=args.task_id, max_steps=cfg.orchestrator.max_steps)
        print(
            f"[episode {episode}] success={result.success} steps={result.steps} "
            f"total_reward={result.total_reward:.3f} num_plans={len(result.plans)}"
        )
        for plan in result.plans:
            print(f"  step {plan.step_issued}: {plan.instruction!r}")

    simulator.close()


if __name__ == "__main__":
    main()
