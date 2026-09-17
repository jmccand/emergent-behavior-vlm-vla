#!/usr/bin/env python
"""Sanity-check each piece of the stack independently before running a full episode.

Example:
    python scripts/check_setup.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def check_mps() -> bool:
    import torch

    ok = torch.backends.mps.is_available()
    print(f"[mps]        available={ok}")
    return ok


def check_sim(cfg) -> bool:
    try:
        from vlm_vla.config import build_simulator

        sim = build_simulator(cfg.simulator)
        obs = sim.reset(task_id=0)
        print(
            f"[simulator]  OK - task={obs.task_description!r} "
            f"images={list(obs.images.keys())} state_dim={None if obs.state is None else obs.state.shape}"
        )
        sim.close()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[simulator]  FAILED - {exc!r}")
        return False


def check_reasoning(cfg) -> bool:
    try:
        from vlm_vla.config import build_reasoning_agent
        from vlm_vla.core.types import Observation

        agent = build_reasoning_agent(cfg.reasoning)
        dummy_obs = Observation(
            images={"image": np.zeros((256, 256, 3), dtype=np.uint8)},
            state=None,
            task_description="pick up the red block",
            step=0,
            done=False,
        )
        plan = agent.plan(dummy_obs, dummy_obs.task_description, history=[])
        print(f"[reasoning]  OK - instruction={plan.instruction!r}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[reasoning]  FAILED - {exc!r}")
        return False


def check_action(cfg) -> bool:
    try:
        from vlm_vla.config import build_action_agent
        from vlm_vla.core.types import Observation, Plan

        agent = build_action_agent(cfg.action)
        agent.reset()
        dummy_obs = Observation(
            images={
                "image": np.zeros((256, 256, 3), dtype=np.uint8),
                "image2": np.zeros((256, 256, 3), dtype=np.uint8),
            },
            state=np.zeros(8, dtype=np.float32),
            task_description="pick up the red block",
            step=0,
            done=False,
        )
        plan = Plan(instruction="pick up the red block", reasoning=None, step_issued=0)
        action = agent.act(dummy_obs, plan)
        print(f"[action]     OK - action shape={action.array.shape}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[action]     FAILED - {exc!r}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--skip-vla", action="store_true")
    parser.add_argument("--skip-sim", action="store_true")
    args = parser.parse_args()

    from vlm_vla.config import load_config

    cfg = load_config(args.config)

    results = [check_mps()]
    if not args.skip_sim:
        results.append(check_sim(cfg))
    if not args.skip_vlm:
        results.append(check_reasoning(cfg))
    if not args.skip_vla:
        results.append(check_action(cfg))

    if not all(results):
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
