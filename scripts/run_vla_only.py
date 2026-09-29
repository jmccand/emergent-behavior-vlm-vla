#!/usr/bin/env python
"""Run one or more LIBERO episodes through SmolVLA alone, with no VLM in the loop.

This is the no-VLM baseline: the action agent is fed the simulator's own
ground-truth task description directly (via `PassthroughReasoningAgent`)
instead of a paraphrased subgoal from a reasoning agent, matching how the
SmolVLA paper's own LIBERO eval works. It exists to establish an
apples-to-apples comparison against those reported numbers, and to
sanity-check the harness itself, without the cost of loading a VLM.

Example:
    python scripts/run_vla_only.py --config configs/default.yaml \
        --task-id 0 --episodes 1
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from vlm_vla.config import build_action_agent, build_simulator, load_config
from vlm_vla.core import EpisodeRecorder, SummaryLog
from vlm_vla.orchestrator import HierarchicalAgent
from vlm_vla.reasoning.passthrough import PassthroughReasoningAgent

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("vlm_vla").setLevel(logging.INFO)

# Number of tasks in each standard LIBERO suite (see benchmark.get_benchmark_dict
# in the LIBERO package). Mirrors the ranges already documented in README.md.
_SUITE_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}


def _num_tasks(task_suite: str) -> int:
    try:
        return _SUITE_TASK_COUNTS[task_suite]
    except KeyError:
        raise ValueError(
            f"Unknown task count for suite {task_suite!r}; known suites: "
            f"{sorted(_SUITE_TASK_COUNTS)}. Pass --task-id explicitly instead of --all-tasks."
        ) from None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run every task in the configured task suite (overrides --task-id); "
        "--episodes still controls how many episodes run per task",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save an mp4 rollout + plans/steps JSONL logs per episode under --output-dir",
    )
    parser.add_argument("--output-dir", default="outputs", help="Base directory for --record output")
    parser.add_argument("--camera", default="image", help="Observation camera key to record frames from")
    parser.add_argument(
        "--second-camera",
        default="image2",
        help="Additional observation camera key placed side-by-side with --camera in the recorded "
        "video (defaults to the wrist camera); pass an empty string to disable",
    )
    args = parser.parse_args()

    if args.all_tasks and args.task_id is not None:
        parser.error("--all-tasks and --task-id are mutually exclusive")

    cfg = load_config(args.config)

    summary_path = Path(args.output_dir) / "summaries" / f"{time.strftime('%Y%m%d-%H%M%S')}-run_vla_only.jsonl"
    summary_log = SummaryLog(summary_path)

    action_agent = build_action_agent(cfg.action)
    simulator = build_simulator(cfg.simulator)

    agent = HierarchicalAgent(
        reasoning_agent=PassthroughReasoningAgent(),
        action_agent=action_agent,
        simulator=simulator,
        replan_every=cfg.orchestrator.max_steps + 1,
    )

    task_ids = list(range(_num_tasks(cfg.simulator.task_suite))) if args.all_tasks else [args.task_id]

    for task_id in task_ids:
        for episode in range(args.episodes):
            recorder = None
            if args.record:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                suffix = f"-task{task_id}" if args.all_tasks else ""
                episode_dir = Path(args.output_dir) / f"{timestamp}{suffix}-ep{episode}"
                recorder = EpisodeRecorder(
                    episode_dir, camera=args.camera, second_camera=args.second_camera or None
                )

            result = agent.run_episode(
                task_id=task_id, max_steps=cfg.orchestrator.max_steps, recorder=recorder
            )

            if recorder is not None:
                recorder.close()

            summary_log.write(
                mode="vla_only",
                config_path=args.config,
                task_suite=cfg.simulator.task_suite,
                task_id=task_id if task_id is not None else 0,
                episode=episode,
                success=result.success,
                steps=result.steps,
                total_reward=result.total_reward,
                num_plans=len(result.plans),
                replan_every=None,
                max_steps=cfg.orchestrator.max_steps,
                action_checkpoint=cfg.action.checkpoint,
                reasoning_model_id=None,
                n_action_steps=None,
            )

            label = f"task {task_id} episode {episode}" if task_id is not None else f"episode {episode}"
            print(
                f"[{label}] success={result.success} steps={result.steps} "
                f"total_reward={result.total_reward:.3f} num_plans={len(result.plans)}"
            )
            for plan in result.plans:
                print(f"  step {plan.step_issued}: {plan.instruction!r}")
            if recorder is not None:
                print(f"  recorded to {recorder.output_dir}")

    simulator.close()
    summary_log.close()


if __name__ == "__main__":
    main()
