"""Config-driven backend selection.

Swapping the VLM, the VLA, or the simulator is meant to be a YAML edit, not a
code change. `build_*` functions are the single place that maps a config
string to a concrete class -- add a new backend by adding one branch here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from vlm_vla.core import ActionAgent, ReasoningAgent, Simulator


@dataclass
class ReasoningConfig:
    backend: str = "qwen_vl"
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "mps"
    max_new_tokens: int = 256


@dataclass
class ActionConfig:
    backend: str = "smolvla"
    checkpoint: str = "lerobot/smolvla_libero"
    device: str = "mps"


@dataclass
class SimulatorConfig:
    backend: str = "libero_docker"
    host: str = "localhost"
    port: int = 8000
    task_suite: str = "libero_spatial"


@dataclass
class OrchestratorConfig:
    replan_every: int = 30
    max_steps: int = 300


@dataclass
class Config:
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        reasoning=ReasoningConfig(**raw.get("reasoning", {})),
        action=ActionConfig(**raw.get("action", {})),
        simulator=SimulatorConfig(**raw.get("simulator", {})),
        orchestrator=OrchestratorConfig(**raw.get("orchestrator", {})),
    )


def build_reasoning_agent(cfg: ReasoningConfig) -> ReasoningAgent:
    if cfg.backend == "qwen_vl":
        from vlm_vla.reasoning.qwen_vl import QwenVLReasoningAgent

        return QwenVLReasoningAgent(
            model_id=cfg.model_id, device=cfg.device, max_new_tokens=cfg.max_new_tokens
        )
    raise ValueError(f"Unknown reasoning backend: {cfg.backend!r}")


def build_action_agent(cfg: ActionConfig, n_action_steps: int | None = None) -> ActionAgent:
    if cfg.backend == "smolvla":
        from vlm_vla.action.smolvla import SmolVLAActionAgent

        return SmolVLAActionAgent(checkpoint=cfg.checkpoint, device=cfg.device, n_action_steps=n_action_steps)
    raise ValueError(f"Unknown action backend: {cfg.backend!r}")


def build_simulator(cfg: SimulatorConfig) -> Simulator:
    if cfg.backend == "libero_docker":
        from vlm_vla.simulation.libero_client import LiberoSimulator

        return LiberoSimulator(host=cfg.host, port=cfg.port, task_suite=cfg.task_suite)
    raise ValueError(f"Unknown simulator backend: {cfg.backend!r}")
