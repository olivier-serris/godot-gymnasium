"""Export a model of a training run to ONNX for NNController (train.py already exports final.zip).

  uv run game/addons/godot_gymnasium/python/export.py runs/<run>/checkpoints/step_500000_steps.zip
"""

import json
from dataclasses import dataclass
from pathlib import Path

import tyro

from godot_gymnasium.sb3 import export_onnx


@dataclass
class Export:
    """Export a model of a run folder to ONNX: final.zip -> policy.onnx, other.zip -> policy_other.onnx."""
    model: tyro.conf.Positional[str]
    """e.g. runs/<run>/checkpoints/step_500000_steps.zip"""


def export(args: Export) -> None:
    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise SystemExit(f"{args.model}: no such file")
    run_dir = next(p for p in model_path.parents if (p / "config.json").exists())
    config = json.loads((run_dir / "config.json").read_text())
    name = model_path.stem
    export_onnx(
        model_path,
        run_dir / ("policy.onnx" if name == "final" else f"policy_{name}.onnx"),
        godot_metadata=config["godot"],
        # Not the config: it holds absolute paths (godot_exe, project), and the .onnx ships in games.
        run_info={"run_name": run_dir.name, "model": str(model_path.relative_to(run_dir))},
    )


if __name__ == "__main__":
    export(tyro.cli(Export))
