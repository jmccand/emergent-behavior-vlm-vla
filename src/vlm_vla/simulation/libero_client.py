"""HTTP client for the LIBERO env server running in the Docker container.

Runs natively on macOS. The wire format is plain JSON with base64-encoded PNG
images; see `vlm_vla.simulation.libero_server` for the server side of this
protocol. Swap simulators by writing another `Simulator` implementation and
pointing `configs/*.yaml` -> `simulator.backend` at it.
"""

from __future__ import annotations

import base64
import io
import logging

import numpy as np
import requests
from PIL import Image

from vlm_vla.core.simulator import Simulator
from vlm_vla.core.types import Action, Observation

logger = logging.getLogger(__name__)


class LiberoSimulator(Simulator):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        task_suite: str = "libero_spatial",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = f"http://{host}:{port}"
        self.task_suite = task_suite
        self.timeout = timeout
        self._check_health()

    def _check_health(self) -> None:
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        logger.info("Connected to LIBERO sim server at %s", self.base_url)

    def reset(self, task_id: int | None = None) -> Observation:
        payload = {"task_suite": self.task_suite, "task_id": task_id}
        resp = requests.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return _decode_observation(resp.json())

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        payload = {"action": action.array.tolist()}
        resp = requests.post(f"{self.base_url}/step", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        obs = _decode_observation(data["observation"])
        return obs, float(data["reward"]), bool(data["done"]), dict(data.get("info", {}))

    def close(self) -> None:
        try:
            requests.post(f"{self.base_url}/close", timeout=self.timeout)
        except requests.RequestException:
            pass


def _decode_observation(data: dict) -> Observation:
    images = {name: _decode_image(b64) for name, b64 in data["images"].items()}
    state = np.array(data["state"], dtype=np.float32) if data.get("state") is not None else None
    return Observation(
        images=images,
        state=state,
        task_description=data["task_description"],
        step=int(data["step"]),
        done=bool(data["done"]),
        info=dict(data.get("info", {})),
    )


def _decode_image(b64_png: str) -> np.ndarray:
    raw = base64.b64decode(b64_png)
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
