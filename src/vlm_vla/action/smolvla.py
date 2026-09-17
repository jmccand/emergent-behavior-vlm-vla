"""SmolVLA as the low-level action agent.

Swap in a different VLA by writing another `ActionAgent` implementation and
pointing `configs/*.yaml` -> `action.backend` at it -- see
`vlm_vla.config.build_action_agent`.

Uses LeRobot's own pre/post-processor pipeline (`make_pre_post_processors`)
rather than hand-rolled normalization: the pretrained checkpoint ships
dataset-derived normalization stats for both observations and actions, and
reimplementing that here would silently diverge from them.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from vlm_vla.core.action_agent import ActionAgent
from vlm_vla.core.types import Action, Observation, Plan

logger = logging.getLogger(__name__)


class SmolVLAActionAgent(ActionAgent):
    def __init__(self, checkpoint: str = "lerobot/smolvla_libero", device: str = "mps") -> None:
        self.device = device
        logger.info("Loading action VLA %s on %s", checkpoint, device)
        # `device=` alone doesn't override the checkpoint's saved config (only `cli_overrides`
        # reaches draccus before its device-availability check), so the checkpoint's default
        # `device: cuda` would otherwise trip lerobot's "switching to 'mps'" warning here.
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, cli_overrides=[f"--device={device}"])
        self.policy.to(device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def act(self, observation: Observation, plan: Plan) -> Action:
        batch = {
            "observation.state": _to_tensor(observation.state).unsqueeze(0),
            "task": [plan.instruction],
        }
        for cam_name, image in observation.images.items():
            key = f"observation.images.{cam_name}"
            batch[key] = _image_to_tensor(image).unsqueeze(0)

        batch = self.preprocessor(batch)
        with torch.no_grad():
            action = self.policy.select_action(batch)
        action = self.postprocessor(action)

        action_array = action.squeeze(0).to("cpu").numpy()
        return Action(array=action_array)


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32)


def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> CHW float32 in [0, 1], matching LeRobotDataset's image convention."""
    chw = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
    return torch.as_tensor(chw)
