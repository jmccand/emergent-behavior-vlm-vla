"""Qwen3-VL as the high-level reasoning agent.

Swap in a different VLM by writing another `ReasoningAgent` implementation
(same constructor-ish shape) and pointing `configs/*.yaml` -> `reasoning.backend`
at it -- see `vlm_vla.config.build_reasoning_agent`.
"""

from __future__ import annotations

import json
import logging
import re

from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from vlm_vla.core.reasoning_agent import ReasoningAgent
from vlm_vla.core.types import Observation, Plan

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the high-level reasoning module of a robot control system. You see the "
    "robot's current camera view and know the overall task. You do not control the "
    "robot directly -- you issue a subgoal instruction that a separate low-level "
    "controller (a vision-language-action policy) will execute for the next several "
    "timesteps. That policy was trained on richly descriptive instructions, so write "
    "the instruction the same way: a full imperative sentence that names the specific "
    "object by its visual attributes (color, shape, category) and, whenever more than "
    "one similar object is visible, disambiguates it by its spatial relation to other "
    "objects (e.g. 'the black bowl on the stove' rather than just 'the bowl'). If the "
    "subgoal involves placing or moving something, name the destination just as "
    "concretely (e.g. 'place it on the plate to the left of the ramekin'). Think about "
    "what the single most useful next subgoal is, including creative or resourceful use "
    "of objects in the scene if the direct path is blocked. Respond with a compact JSON "
    'object: {"reasoning": "<brief reasoning trace>", "instruction": "<full descriptive '
    "imperative subgoal, e.g. 'pick up the black bowl between the plate and the ramekin "
    "and place it on the plate'>\"}."
)


class QwenVLReasoningAgent(ReasoningAgent):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "mps",
        max_new_tokens: int = 256,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.device = device
        logger.info("Loading reasoning VLM %s on %s", model_id, device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype="bfloat16"
        ).to(device)
        self.model.eval()

    def plan(self, observation: Observation, task: str, history: list[Plan]) -> Plan:
        image = Image.fromarray(observation.images["image"])
        history_text = (
            "\n".join(f"- {p.instruction}" for p in history[-5:])
            if history
            else "(none yet)"
        )
        user_text = (
            f"Overall task: {task}\n"
            f"Previous subgoals issued so far:\n{history_text}\n"
            "What is the next subgoal?"
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        import torch

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        raw_text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

        instruction, reasoning = _parse_response(raw_text)
        return Plan(
            instruction=instruction,
            reasoning=reasoning,
            step_issued=observation.step,
            metadata={"raw_response": raw_text},
        )


def _parse_response(raw_text: str) -> tuple[str, str | None]:
    """Extract {instruction, reasoning} from the model's JSON reply, tolerating stray text."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            instruction = str(parsed.get("instruction", "")).strip()
            reasoning = parsed.get("reasoning")
            if instruction:
                return instruction, reasoning
        except json.JSONDecodeError:
            pass
    logger.warning("Failed to parse structured plan from VLM output: %r", raw_text)
    return raw_text.strip(), None
