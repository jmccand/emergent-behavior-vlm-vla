# emergent-behavior-vlm-vla

Research project studying the ability of VLMs and VLAs to creatively determine resourceful actions to solve complex tasks under physical constraints.

VLMs and VLAs (and cascaded arrangements of the two) are growing popular in modern robotics. One natural question is: how "intelligent" are these systems, really?

The behavior that we seek to study is their ability to act resourcefully and creatively in complex situations to overcome their physical constraints. For instance, a robotic arm grabbing a bowling pin to extend its reach.

## Framework

This repo implements a modular **dual-system** agent, inspired by [ThinkAct](https://arxiv.org/abs/2507.16815): a high-level **reasoning agent** (VLM) periodically issues a natural-language subgoal, and a low-level **action agent** (VLA) executes it at every control step. All three pieces — reasoning agent, action agent, and simulator — are abstract interfaces (`src/vlm_vla/core/`), so any of them can be swapped via `configs/default.yaml` without touching the orchestrator.

Default backends:
- **Reasoning:** [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), run natively via `transformers` (MPS).
- **Action:** [SmolVLA](https://huggingface.co/lerobot/smolvla_libero) (`lerobot/smolvla_libero` checkpoint), run natively via `lerobot` (MPS).
- **Simulator:** [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), run inside a Docker container (Linux-only dependency) and accessed over a local HTTP server.

See `src/vlm_vla/orchestrator.py` for the control loop and `src/vlm_vla/config.py` for backend wiring.

### Setup (one-time)

**1. LIBERO simulator image (Docker):**

```bash
open -a Docker  # start Docker Desktop if not running
docker build -f docker/Dockerfile.libero -t libero-sim .
```

The image build compiles/downloads robosuite, MuJoCo, and torch inside the
container, so the first build can take 15-30+ minutes. Subsequent builds are
cached and fast unless the Dockerfile or LeRobot's pinned commit changes.

**2. Agent stack (native macOS, Apple Silicon MPS):**

```bash
# lerobot requires Python >=3.12
conda create -n vlm-vla-py312 python=3.12 -y
conda activate vlm-vla-py312
pip install -e .

# SmolVLA policy class, installed from a sibling checkout (kept out of this repo):
git clone https://github.com/huggingface/lerobot.git ../lerobot
pip install -e "../lerobot[smolvla]"
```

If you use VS Code / the Claude Code IDE extension, `.vscode/settings.json`
points the Python interpreter at this env automatically. If you use a
different editor or `pyenv`, point it at
`$(conda info --base)/envs/vlm-vla-py312/bin/python` yourself, or make sure
you `conda activate vlm-vla-py312` before running anything below.

### Running

Every run needs **two things active at once**: the LIBERO container, and the
`vlm-vla-py312` conda env. In one terminal:

```bash
docker run --rm -p 8000:8000 --name libero-sim libero-sim
```

Leave that running. In a second terminal, for every new shell:

```bash
conda activate vlm-vla-py312
cd /path/to/emergent-behavior-vlm-vla

# One-time-per-shell sanity check (loads both models + pings the sim server):
python scripts/check_setup.py --config configs/default.yaml

# Run an episode. --task-id selects which task within the suite (0-9 for
# libero_spatial/object/goal/10; 0-89 for libero_90). Suite is set in
# configs/default.yaml under simulator.task_suite.
python scripts/run_episode.py --config configs/default.yaml --task-id 0 --episodes 1

# Run every task in the suite, --episodes times each (a full benchmark sweep):
python scripts/run_episode.py --config configs/default.yaml --all-tasks --episodes 5
```

To inspect a run afterwards -- the rollout video plus every subgoal and raw
model output -- pass `--record`:

```bash
python scripts/run_episode.py --config configs/default.yaml --task-id 0 --episodes 1 --record
```

Each episode is written to its own timestamped directory under `outputs/`
(override with `--output-dir`):

- `episode.mp4` -- the rollout, one frame per step (`--camera` selects the
  primary `Observation.images` key, e.g. `image` vs `image2` for wrist cam;
  `--second-camera` is placed side-by-side with it, defaulting to `image2` so
  the wrist view is included alongside the agentview camera by default -- pass
  an empty string to disable it).
- `plans.jsonl` -- one row per subgoal issued by the reasoning agent
  (`instruction`, `reasoning`, and the raw VLM response in `metadata`).
- `steps.jsonl` -- one row per control step (`action`, `reward`, `done`,
  `info`, and the instruction active at that step).

The first `run_episode.py`/`check_setup.py` call downloads Qwen3-VL-8B-Instruct
(~16GB) and the SmolVLA checkpoint from Hugging Face Hub; later runs use the
local cache (`~/.cache/huggingface`) and start much faster. The first
`/reset` against a fresh container also lazily downloads LIBERO's render/physics
assets (~30-40s one-time cost per container lifetime).

When you're done, stop the container with `docker stop libero-sim`.

### Troubleshooting

- **`ModuleNotFoundError: No module named 'vlm_vla'`** — you're not running
  inside the `vlm-vla-py312` conda env (a fresh terminal, an IDE run button
  using a different interpreter, etc.). Run `conda activate vlm-vla-py312`
  first, or select that interpreter in your editor.
- **`requests.exceptions.ConnectionError` / simulator health check fails** —
  the Docker container isn't running. Start it with the `docker run` command
  above and confirm with `curl localhost:8000/health`.