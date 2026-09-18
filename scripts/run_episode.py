#!/usr/bin/env python
"""Run one or more LIBERO episodes through the full VLM -> VLA hierarchical loop.

Example:
    python scripts/run_episode.py --config configs/default.yaml \
        --task-id 0 --episodes 1
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from vlm_vla.config import build_action_agent, build_reasoning_agent, build_simulator, load_config
from vlm_vla.core import EpisodeRecorder
from vlm_vla.orchestrator import HierarchicalAgent

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("vlm_vla").setLevel(logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save an mp4 rollout + plans/steps JSONL logs per episode under --output-dir",
    )
    parser.add_argument("--output-dir", default="outputs", help="Base directory for --record output")
    parser.add_argument("--camera", default="image", help="Observation camera key to record frames from")
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
        recorder = None
        if args.record:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            episode_dir = Path(args.output_dir) / f"{timestamp}-ep{episode}"
            recorder = EpisodeRecorder(episode_dir, camera=args.camera)

        result = agent.run_episode(
            task_id=args.task_id, max_steps=cfg.orchestrator.max_steps, recorder=recorder
        )

        if recorder is not None:
            recorder.close()

        print(
            f"[episode {episode}] success={result.success} steps={result.steps} "
            f"total_reward={result.total_reward:.3f} num_plans={len(result.plans)}"
        )
        for plan in result.plans:
            print(f"  step {plan.step_issued}: {plan.instruction!r}")
        if recorder is not None:
            print(f"  recorded to {recorder.output_dir}")

    simulator.close()


if __name__ == "__main__":
    main()
