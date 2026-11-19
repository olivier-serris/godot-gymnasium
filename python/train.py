"""Train PPO (Stable-Baselines3) on a Godot env, then export the final model to policy.onnx for NNController.

  uv run game/addons/godot_gymnasium/python/train.py --project game --env-scene res://my_task/my_task.tscn [--ppo.gamma 0.995 ...]

A basic script: for anything its options don't cover, copy it and change the copy.
Each training run writes a new folder <runs-dir>/<run-name>_<date>_<time>/:
  config.json      how the run was made: command options, all PPO hyperparameters, the trained
                   [AgentParameter] values and physics rate, git commit of the project, library versions
  progress.csv, events.out.tfevents.*   logs (tensorboard --logdir runs)
  checkpoints/     step_<n>_steps.zip every --checkpoint-every timesteps (export.py exports them)
  final.zip        the model when training stops (also after Ctrl+C or an error)
  policy.onnx      final.zip exported for NNController, the only file the game needs
Nothing refers to the folder's location: move or rename it freely.
"""

import importlib.metadata
import inspect
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import tyro
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor

from godot_gymnasium import GodotVectorEnv
from godot_gymnasium.sb3 import SB3VecEnv, export_onnx

GODOT_EXE_ENV = "GODOT4"  # environment variable holding the Godot .NET executable


@dataclass
class Ppo:
    """PPO hyperparameters (the others keep SB3's defaults; config.json lists them all)."""
    n_steps: int = 128
    """Rollout length per env; batch = n_steps * number of envs."""
    n_minibatches: int = 8
    """Minibatches per batch."""
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    ent_coef: float = 0.0


@dataclass
class Train:
    """Train PPO on a Godot env, then export the final model to policy.onnx."""
    project: str | None = None
    """Godot project folder (the one with project.godot). Required unless --editor."""
    env_scene: str | None = None
    """res:// path of the env scene (root extends GymEnv). Required unless --editor."""
    editor: bool = False
    """Launch nothing: wait for bridge.tscn run from the editor (F6) on port 11008, for breakpoints
    (see the README's Debug section). Physics runs in real time."""
    godot_exe: str = os.environ.get(GODOT_EXE_ENV, "")
    """Godot .NET executable to launch, default $GODOT4. Ignored with --editor."""
    n_instances: int | None = None
    """Godot processes (about one CPU core each), default 8. With --editor: editor runs to wait for, default 1."""
    envs_per_instance: int = 16
    """Ignored with --editor (set by the editor's launch arguments)."""
    total_timesteps: int = 2_000_000
    seed: int = 0
    headless: bool = True
    """Ignored with --editor."""
    env: list[str] = field(default_factory=list)
    """Env [Export] values, e.g. --env MaxEpisodeSteps=500. Ignored with --editor."""
    agent: list[str] = field(default_factory=list)
    """Agent [AgentParameter] values, e.g. --agent Speed=400. Ignored with --editor."""
    device: str = "cpu"
    """Torch device for the networks and the PPO update (Godot always runs on CPU)."""
    run_name: str | None = None
    """Start of the run folder name, before the date and time. Default: the env scene's name."""
    runs_dir: str = "runs"
    """Where the run folder is created."""
    checkpoint_every: int = 500_000
    """Timesteps between checkpoints, 0 = no checkpoints."""
    ppo: Ppo = field(default_factory=Ppo)


def git_info(folder: Path) -> dict | None:
    """Commit of the git repo holding folder and whether it has uncommitted changes. None if not a git repo."""
    def git(*cmd):
        return subprocess.run(["git", *cmd], cwd=folder, capture_output=True, text=True, check=True).stdout
    try:
        return {"sha": git("rev-parse", "HEAD").strip(), "dirty": bool(git("status", "--porcelain").strip())}
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_parameters(items: list[str]) -> dict:
    """["Speed=400", "Mode=fast"] -> {"Speed": 400, "Mode": "fast"}: JSON values, else text."""
    params = {}
    for item in items:
        name, _, value = item.partition("=")
        try:
            params[name] = json.loads(value)
        except json.JSONDecodeError:
            params[name] = value
    return params


def ppo_hyperparameters(kwargs: dict) -> dict:
    """All PPO constructor arguments: SB3's defaults, replaced by kwargs."""
    params = inspect.signature(PPO.__init__).parameters.values()
    return {p.name: p.default for p in params if p.default is not inspect.Parameter.empty} | kwargs


def train(args: Train) -> None:
    n_instances = args.n_instances or (1 if args.editor else 8)
    if args.editor:
        launch = {}
    else:
        if not (args.project and args.env_scene):
            raise SystemExit("--project and --env-scene are required (or --editor)")
        if not args.godot_exe:
            raise SystemExit(f"No Godot executable: set ${GODOT_EXE_ENV} or pass --godot-exe PATH")
        launch = dict(
            godot_exe=args.godot_exe,
            project_path=args.project,
            env_scene=args.env_scene,
            envs_per_instance=args.envs_per_instance,
            headless=args.headless,
            env_parameters=parse_parameters(args.env),
            agent_parameters=parse_parameters(args.agent),
        )
    venv = GodotVectorEnv(n_instances=n_instances, editor=args.editor, **launch)
    run_name = args.run_name or (Path(args.env_scene).stem if args.env_scene else "editor")
    run_dir = Path(args.runs_dir) / f"{run_name}_{datetime.now():%Y-%m-%d_%H%M%S}"
    run_dir.mkdir(parents=True)  # raises if it exists: a run is never overwritten
    env = VecMonitor(SB3VecEnv(venv))  # logs rollout/ep_rew_mean and ep_len_mean
    n_envs = env.num_envs
    p = args.ppo
    ppo_kwargs = dict(
        policy="MlpPolicy",
        n_steps=p.n_steps,
        batch_size=n_envs * p.n_steps // p.n_minibatches,
        n_epochs=p.n_epochs,
        gamma=p.gamma,
        gae_lambda=p.gae_lambda,
        learning_rate=p.learning_rate,
        clip_range=p.clip_range,
        ent_coef=p.ent_coef,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    model = PPO(env=env, **ppo_kwargs)

    config = {
        "args": asdict(args) | {"n_instances": n_instances},  # resolved: the values used
        "ppo": ppo_hyperparameters(ppo_kwargs),
        "godot": venv.model_metadata,  # read by export.py
        "git": git_info(Path(args.project or ".").resolve()),
        "versions": {pkg: importlib.metadata.version(pkg)
                     for pkg in ("godot-gymnasium", "stable-baselines3", "torch", "gymnasium")},
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    model.set_logger(configure(str(run_dir), ["stdout", "csv", "tensorboard"]))
    callbacks = []
    if args.checkpoint_every > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(args.checkpoint_every // n_envs, 1),  # counted in env steps, each is n_envs timesteps
            save_path=str(run_dir / "checkpoints"),
            name_prefix="step",
        ))
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    finally:
        model.save(run_dir / "final")
        env.close()
        export_onnx(
            run_dir / "final.zip",
            run_dir / "policy.onnx",
            godot_metadata=config["godot"],
            # Not the config: it holds absolute paths (godot_exe, project), and the .onnx ships in games.
            run_info={"run_name": run_dir.name, "model": "final.zip"},
        )
        print(f"run saved in {run_dir}")


if __name__ == "__main__":
    train(tyro.cli(Train))
