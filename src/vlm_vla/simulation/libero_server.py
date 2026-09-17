"""FastAPI server wrapping LeRobot's LIBERO env. Runs inside the Docker container
(linux/arm64), started as the container's entrypoint. See `docker/Dockerfile.libero`.

Deliberately reuses LeRobot's own observation pipeline (`preprocess_observation`
+ the env's `LiberoProcessorStep`) rather than reading raw robosuite keys
directly: LIBERO's raw state is a nested dict (eef pos/quat, gripper qpos, ...)
that has to be converted to axis-angle and concatenated in a specific order,
and its raw images need a LIBERO-specific 180-degree flip. That logic lives in
LeRobot and is exactly what a pretrained checkpoint (e.g. `smolvla_libero`)
was trained against, so we call it rather than reimplementing it.

Wire protocol (JSON, images as base64 PNG) is consumed by
`vlm_vla.simulation.libero_client.LiberoSimulator` on the macOS side.
"""

from __future__ import annotations

import base64
import io
import logging

import numpy as np
import torch
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env
from lerobot.envs.utils import NEW_ROLLOUT_OPTION, close_envs, preprocess_observation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

_state: dict = {
    "envs_dict": None,
    "env": None,
    "env_preprocessor": None,
    "task_suite": None,
    "task_id": None,
    "step": 0,
}


class ResetRequest(BaseModel):
    task_suite: str = "libero_spatial"
    task_id: int | None = None


class StepRequest(BaseModel):
    action: list[float]


def _build_env(task_suite: str, task_id: int | None):
    resolved_task_id = 0 if task_id is None else task_id
    if _state["env"] is None or _state["task_suite"] != task_suite or _state["task_id"] != resolved_task_id:
        if _state["envs_dict"] is not None:
            close_envs(_state["envs_dict"])
        logger.info("Building LIBERO env: suite=%s task_id=%d", task_suite, resolved_task_id)
        env_cfg = LiberoEnv(task=task_suite, task_ids=[resolved_task_id])
        envs_dict = make_env(env_cfg, n_envs=1)
        env = envs_dict[task_suite][resolved_task_id]
        env_preprocessor, _ = env_cfg.get_env_processors()
        _state.update(
            envs_dict=envs_dict,
            env=env,
            env_preprocessor=env_preprocessor,
            task_suite=task_suite,
            task_id=resolved_task_id,
            step=0,
        )
    return _state["env"], _state["env_preprocessor"]


def _encode_image(array: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _tensor_obs_to_wire(obs: dict) -> tuple[dict, list[float] | None]:
    """LeRobot-format tensor obs (batch=1) -> {cam_name: base64 png}, state list."""
    images: dict[str, str] = {}
    state: list[float] | None = None
    for key, value in obs.items():
        if key.startswith("observation.images."):
            name = key.removeprefix("observation.images.")
            img_chw = value[0].clamp(0, 1)
            img_uint8 = (img_chw * 255).round().to(torch.uint8)
            img_hwc = img_uint8.permute(1, 2, 0).cpu().numpy()
            images[name] = _encode_image(img_hwc)
        elif key == "observation.state":
            state = value[0].cpu().tolist()
    return images, state


def _extract_success(info: dict, n_envs: int) -> bool:
    """Mirrors LeRobot's own rollout(): VectorEnv may report `is_success` directly
    or nested under `final_info` depending on gymnasium version / autoreset state."""
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict):
            is_success = final_info.get("is_success", [False] * n_envs)
            return bool(np.asarray(is_success).reshape(-1)[0])
        item = final_info[0] if len(final_info) else None
        return bool(item.get("is_success", False)) if isinstance(item, dict) else False
    if "is_success" in info:
        return bool(np.asarray(info["is_success"]).reshape(-1)[0])
    return False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest):
    env, env_preprocessor = _build_env(req.task_suite, req.task_id)
    raw_obs, _info = env.reset(options={NEW_ROLLOUT_OPTION: True})
    obs_t = env_preprocessor(preprocess_observation(raw_obs))
    images, state = _tensor_obs_to_wire(obs_t)
    _state["step"] = 0
    task_description = list(env.call("task_description"))[0]
    return {
        "images": images,
        "state": state,
        "task_description": task_description,
        "step": 0,
        "done": False,
        "info": {},
    }


@app.post("/step")
def step(req: StepRequest):
    env = _state["env"]
    env_preprocessor = _state["env_preprocessor"]
    action = np.asarray(req.action, dtype=np.float32).reshape(1, -1)
    raw_obs, reward, terminated, truncated, info = env.step(action)
    _state["step"] += 1

    done = bool(np.asarray(terminated).reshape(-1)[0] or np.asarray(truncated).reshape(-1)[0])
    success = _extract_success(info, n_envs=1)

    obs_t = env_preprocessor(preprocess_observation(raw_obs))
    images, state = _tensor_obs_to_wire(obs_t)
    task_description = list(env.call("task_description"))[0]

    return {
        "observation": {
            "images": images,
            "state": state,
            "task_description": task_description,
            "step": _state["step"],
            "done": done,
            "info": {},
        },
        "reward": float(np.asarray(reward).reshape(-1)[0]),
        "done": done,
        "info": {"success": success},
    }


@app.post("/close")
def close():
    if _state["envs_dict"] is not None:
        close_envs(_state["envs_dict"])
    _state.update(envs_dict=None, env=None, env_preprocessor=None, task_suite=None, task_id=None)
    return {"status": "closed"}
