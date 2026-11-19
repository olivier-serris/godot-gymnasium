# godot-gymnasium

Train reinforcement-learning agents in a Godot (C#) game from Python.
Godot runs the envs on the CPU and talks to a Gymnasium `VectorEnv` over TCP
([godot_vector_env.py](python/godot_gymnasium/godot_vector_env.py) ↔ [GymBridge.cs](GymBridge.cs)).
Training uses Stable-Baselines3 PPO, and the trained policy is exported to ONNX so the game can run it
([NNController.cs](NNController.cs)).

This repository is the Godot addon, and it holds the Python package too (in `python/`, which Godot ignores).
Example: [godot-gymnasium-examples](https://github.com/olivier-serris/godot-gymnasium-examples), a dot that learns to reach a goal.

Inspired by [Godot RL Agents](https://github.com/edbeeching/godot_rl_agents), which is a real, finished library:
use it if you need something that just works. See [Differences with Godot RL Agents](#differences-with-godot-rl-agents).

## Install

The game must be a Godot **.NET** project (tested on Linux with Godot 4.7.2; Python ≥ 3.12).
From the game's git repository root, with the Godot project in `game/` (if `project.godot` is at the root, write `.`
instead of `game`):

```bash
git submodule add https://github.com/olivier-serris/godot-gymnasium game/addons/godot_gymnasium
uv add --editable ./game/addons/godot_gymnasium --extra sb3    # no pyproject.toml yet? first: uv init --bare
dotnet add game/MyGame.csproj package Microsoft.ML.OnnxRuntime --version 1.32.0
```

With pip instead of uv: `pip install -e "./game/addons/godot_gymnasium[sb3]"`.
To clone the game with the addon: `git clone --recursive`, or `git submodule update --init` after a plain clone.

Build the C# project before training, and again after every C# change (training runs the last build): **Build** in
the editor, or `$GODOT4 --headless --path game --build-solutions --quit` (`GODOT4`: your Godot .NET executable,
also used by training).

To update: `git submodule update --remote`, then build.

## Create your own

1. An agent: a C# class extending `ControllableAgent` (`using GodotGymnasium;`), with `ActionType`, `ActionSize`,
   `GetObservation()` and `ApplyAction()`.
2. An env: a C# class extending `GymEnv`, with `Reset()`, `GetReward()` and `IsTerminated()`.
3. A scene whose root is the env, with the agent inside it, set as the env's `Agent`.
4. Build the C# project (see [Install](#install)), then train:
   `uv run game/addons/godot_gymnasium/python/train.py --project game --env-scene res://my_task/my_task.tscn` (`--help` for all
   options).
   When training ends, the run folder `runs/my_task_<date>_<time>/` holds the trained policy `policy.onnx`.
5. Play: in your game scene, add an `NNController` with `ModelPath` set to the trained `policy.onnx`, and your agent in
   its `Agents`.

## Debug

See the envs while training: add `--no-headless --n-instances 1 --envs-per-instance 16` to `train.py`.

Run Godot from the editor (breakpoints, prints, remote scene tree):

1. Start Python first, without launching Godot: `uv run game/addons/godot_gymnasium/python/train.py --editor --total-timesteps 3000 --run-name debug`
2. In the editor, in **Debug → Customize Run Instances**, set the launch arguments
   `-- --env-scene=res://my_task/my_task.tscn --envs-per-instance=4`, then open
   `res://addons/godot_gymnasium/bridge.tscn` and press **F6**.

Physics runs in real time in this mode: keep `--total-timesteps` small.

## Differences with Godot RL Agents

- **Not a real library:** godot-gymnasium is a quick personal project, mostly written by babysitting Claude.
- **Training runs as fast as the CPU allows:** godot-gymnasium launches Godot with `--fixed-fps`, so the game no longer
  follows the real clock. Godot RL Agents speeds up the clock instead (`--speedup=8`), which keeps it tied to
  real time: 8× by default, and about 20× at most in headless Godot unless you change project settings
  (see their [issue #41](https://github.com/edbeeching/godot_rl_agents/issues/41)).
- **Batched inference in the game:** with several agents driven by the same model, godot-gymnasium's `NNController`
  runs the model once for all of them per decision. Godot RL Agents runs the model once per agent.
