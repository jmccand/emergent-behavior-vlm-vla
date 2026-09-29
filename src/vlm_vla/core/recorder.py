"""Optional episode recorder: rollout video + step-by-step JSONL logs.

Kept independent of `HierarchicalAgent` so recording is strictly opt-in --
passing `recorder=None` (the default) costs nothing and the control loop
never has to know how frames/logs are persisted. Plans and steps are logged
to separate files because a plan only changes every `replan_every` steps;
folding it into the per-step log would repeat the same reasoning text on
every row for no benefit.
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vlm_vla.core.types import Action, Observation, Plan

logger = logging.getLogger(__name__)


def _burn_in_plan_text(frame: np.ndarray, text: str) -> np.ndarray:
    """Return a copy of `frame` with `text` drawn in a band along the bottom."""
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size

    chars_per_line = max(1, width // 6)
    lines = textwrap.wrap(text, width=chars_per_line) or [text]
    line_height = 11
    band_height = min(height, line_height * len(lines) + 8)

    draw.rectangle([0, height - band_height, width, height], fill=(0, 0, 0, 170))
    for i, line in enumerate(lines):
        draw.text((4, height - band_height + 4 + i * line_height), line, fill=(255, 255, 255, 255))

    return np.array(image)


class EpisodeRecorder:
    def __init__(self, output_dir: str | Path, camera: str = "image", fps: int = 10) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera = camera
        self.fps = fps
        self._frames: list[np.ndarray] = []
        self._current_plan_text: str | None = None
        self._plans_file = (self.output_dir / "plans.jsonl").open("w")
        self._steps_file = (self.output_dir / "steps.jsonl").open("w")

    def record_frame(self, observation: Observation) -> None:
        frame = observation.images[self.camera]
        if self._current_plan_text is not None:
            frame = _burn_in_plan_text(frame, self._current_plan_text)
        self._frames.append(frame)

    def record_plan(self, plan: Plan) -> None:
        self._current_plan_text = plan.instruction
        entry = {
            "step_issued": plan.step_issued,
            "instruction": plan.instruction,
            "reasoning": plan.reasoning,
            "metadata": plan.metadata,
        }
        self._plans_file.write(json.dumps(entry) + "\n")
        self._plans_file.flush()

    def record_step(
        self,
        observation: Observation,
        plan: Plan,
        action: Action,
        reward: float,
        done: bool,
        info: dict,
    ) -> None:
        self.record_frame(observation)
        entry = {
            "step": observation.step,
            "instruction": plan.instruction,
            "action": action.array.tolist(),
            "reward": reward,
            "done": done,
            "info": info,
        }
        self._steps_file.write(json.dumps(entry) + "\n")
        self._steps_file.flush()

    def close(self) -> None:
        self._plans_file.close()
        self._steps_file.close()
        if not self._frames:
            logger.warning("No frames recorded; skipping video write")
            return

        import imageio.v3 as iio

        video_path = self.output_dir / "episode.mp4"
        iio.imwrite(video_path, np.stack(self._frames), fps=self.fps)
        logger.info("Wrote %d frames to %s", len(self._frames), video_path)
