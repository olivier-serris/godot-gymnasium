"""Stable-Baselines3 helpers (needs the sb3 extra: godot-gymnasium[sb3]).

SB3VecEnv:   a GodotVectorEnv seen as an SB3 VecEnv.
export_onnx: the actor of an SB3 PPO model to an .onnx file for NNController.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.vec_env import VecEnv

from .godot_vector_env import GodotVectorEnv

# ONNX metadata keys, keep in sync with OnnxPolicy.cs.
AGENT_PARAMETERS_KEY = "godot_gymnasium.agent_parameters"
PHYSICS_TICKS_KEY = "godot_gymnasium.physics_ticks_per_second"
RUN_KEY = "godot_gymnasium.run"


class SB3VecEnv(VecEnv):
    """Exposes a Gymnasium VectorEnv (same-step autoreset) with SB3's VecEnv API.

    SB3 has no built-in adapter for Gymnasium vector envs; this only reshapes infos.
    """

    def __init__(self, venv: GodotVectorEnv):
        self.venv = venv
        super().__init__(venv.num_envs, venv.single_observation_space, venv.single_action_space)

    def reset(self):
        seed = self._seeds[0] if self._seeds and self._seeds[0] is not None else None
        self._reset_seeds()
        obs, _ = self.venv.reset(seed=seed)
        return obs

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, rewards, terminated, truncated, info = self.venv.step(self._actions)
        dones = terminated | truncated
        infos = [{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(dones):
            infos[i]["terminal_observation"] = info["final_obs"][i]
            infos[i]["TimeLimit.truncated"] = bool(truncated[i] and not terminated[i])
        return obs, rewards, dones, infos

    def close(self):
        self.venv.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))


class OnnxActor(torch.nn.Module):
    """Deterministic actor of an SB3 ActorCriticPolicy (no critic, no sampling)."""

    def __init__(self, policy: ActorCriticPolicy):
        super().__init__()
        if policy.use_sde:
            raise NotImplementedError("gSDE policies are not supported")
        self.policy = policy
        self.is_box = isinstance(policy.action_space, spaces.Box)
        if self.is_box and not (np.all(policy.action_space.low == -1) and np.all(policy.action_space.high == 1)):
            raise ValueError(f"Box actions must be in [-1, 1], got {policy.action_space}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        p = self.policy
        obs = preprocess_obs(obs, p.observation_space, normalize_images=p.normalize_images)
        latent_pi = p.mlp_extractor.forward_actor(p.pi_features_extractor(obs))
        out = p.action_net(latent_pi)
        # The Gaussian mean is unbounded; SB3 clips actions before giving them to the env.
        return out.clamp(-1.0, 1.0) if self.is_box else out


def export_onnx(
    model_path: str | Path,
    onnx_path: str | Path,
    godot_metadata: dict[str, Any],
    run_info: dict[str, Any] | None = None,
    n_check: int = 1000,
) -> None:
    """Exports the actor of the SB3 PPO model saved at model_path to onnx_path, for NNController.

    Output: Box action (mean clipped to [-1, 1], as SB3 does) or Discrete logits. The .onnx is the
    only file the game needs. Its metadata holds:
      godot_gymnasium.agent_parameters          [AgentParameter] values, set on the agents by NNController
      godot_gymnasium.physics_ticks_per_second  physics rate at training, checked by NNController
      godot_gymnasium.run                       information only: model, timesteps, export date, + run_info
    godot_metadata: GodotVectorEnv.model_metadata of the training env.
    run_info: extra information, e.g. {"run_name": ...}; "run_name" and "model" are
      shown by NNController when it loads the model.
    n_check: random observations used to check that ONNX and SB3 give the same actions.
    """
    model_path, onnx_path = Path(model_path), Path(onnx_path)
    model = PPO.load(model_path, device="cpu")
    policy = model.policy.eval()
    obs_space, act_space = policy.observation_space, policy.action_space
    if not isinstance(obs_space, spaces.Box) or len(obs_space.shape) != 1:
        raise NotImplementedError(f"only flat Box observations are supported, got {obs_space}")
    obs_dim = obs_space.shape[0]

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(2, obs_dim)  # batch 2: a batch of 1 could be specialized as a constant
    torch.onnx.export(
        OnnxActor(policy).eval(),
        (example,),
        onnx_path,
        input_names=["obs"],
        output_names=["action"],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        external_data=False,  # one self-contained file (Godot loads it from bytes)
        dynamo=True,
    )
    onnx_model = onnx.load(onnx_path)
    for node in onnx_model.graph.node:
        del node.metadata_props[:]  # torch's stack traces: absolute paths of this machine, shipped in games
    info = {
        "model": model_path.name,
        "timesteps": model.num_timesteps,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    } | (run_info or {})
    onnx.helper.set_model_props(onnx_model, {
        AGENT_PARAMETERS_KEY: json.dumps(godot_metadata["agent_parameters"]),
        PHYSICS_TICKS_KEY: str(godot_metadata["physics_ticks_per_second"]),
        RUN_KEY: json.dumps(info),
    })
    onnx.save(onnx_model, onnx_path)

    # ONNX + NNController's clip/argmax must match SB3's deterministic predict.
    rng = np.random.default_rng(0)
    low = np.where(np.isfinite(obs_space.low), obs_space.low, -1.0)
    high = np.where(np.isfinite(obs_space.high), obs_space.high, 1.0)
    obs = rng.uniform(low, high, size=(n_check, obs_dim)).astype(np.float32)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(["action"], {"obs": obs})[0]
    expected, _ = model.predict(obs, deterministic=True)
    if isinstance(act_space, spaces.Box):
        max_err = float(np.abs(onnx_out - expected).max())
        if max_err > 1e-4:
            raise RuntimeError(f"ONNX and SB3 actions differ (max abs error {max_err:.2e})")
        print(f"check ok: max abs error {max_err:.2e} over {n_check} obs")
    else:
        n_diff = int((onnx_out.argmax(axis=1) != expected).sum())
        if n_diff:
            raise RuntimeError(f"ONNX and SB3 actions differ on {n_diff}/{n_check} obs")
        print(f"check ok: identical actions over {n_check} obs")
    print(f"exported {onnx_path}  obs_dim={obs_dim} -> action {session.get_outputs()[0].shape}")
