"""Gymnasium VectorEnv backed by Godot instances (see GymBridge.cs).

Total envs = n_instances x envs_per_instance. Each Godot instance hosts several env copies;
several instances run in parallel processes (one Godot instance mostly uses one core).
Autoreset mode is SAME_STEP: when an env ends, the returned obs is already the reset obs and
the last obs is in infos["final_obs"] (masked by infos["_final_obs"]).
"""

import json
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from . import protocol as P

BRIDGE_SCENE = "res://addons/godot_gymnasium/bridge.tscn"
DEFAULT_PORT = 11008
CONNECT_TIMEOUT = 60  # seconds, launch mode


def _project_physics_fps(project_path: str) -> int:
    """physics/common/physics_ticks_per_second from project.godot (Godot's default is 60)."""
    text = (Path(project_path) / "project.godot").read_text()
    m = re.search(r"^common/physics_ticks_per_second=(\d+)", text, re.MULTILINE)
    return int(m.group(1)) if m else 60


class GodotVectorEnv(VectorEnv):
    """Vectorized env over Godot.

    Args:
        godot_exe: Godot .NET executable to launch. Required unless editor.
        project_path: Godot project folder (the one with project.godot). Required unless editor.
        env_scene: res:// path of the env scene (root extends GymEnv). Required unless editor.
        n_instances: number of Godot processes (editor: number of editor runs to wait for).
        envs_per_instance: env copies per process.
        env_parameters: values for the env's [Export] properties, e.g. {"MaxEpisodeSteps": 500}.
        agent_parameters: values for the agent's [AgentParameter] properties, e.g. {"ActionRepeat": 4}.
        headless: run without a window. With a window, every env of each instance is shown in a grid.
        editor: launch nothing; wait for `n_instances` Godot instances to connect on DEFAULT_PORT
            (create this first, then run bridge.tscn from the editor). The env scene and number of envs
            come from the editor's launch arguments (Debug > Customize Run Instances): godot_exe, project_path, env_scene,
            envs_per_instance, env_parameters, agent_parameters and headless must be left unset.
            Physics runs in real time (no --fixed-fps).

    `model_metadata`: the agent's [AgentParameter] values and physics rate, to save with the model.
    """

    def __init__(
        self,
        godot_exe: str | None = None,
        project_path: str | None = None,
        env_scene: str | None = None,
        n_instances: int = 1,
        envs_per_instance: int | None = None,
        env_parameters: dict[str, Any] | None = None,
        agent_parameters: dict[str, Any] | None = None,
        headless: bool | None = None,
        editor: bool = False,
    ):
        self._procs: list[subprocess.Popen] = []
        self._socks: list[socket.socket] = []

        launch_only = {
            "godot_exe": godot_exe, "project_path": project_path, "env_scene": env_scene,
            "envs_per_instance": envs_per_instance, "env_parameters": env_parameters,
            "agent_parameters": agent_parameters, "headless": headless,
        }
        if editor:
            given = [name for name, value in launch_only.items() if value is not None]
            if given:
                raise ValueError(f"editor=True: Godot is not launched, {', '.join(given)} would be ignored")
        else:
            missing = [name for name in ("godot_exe", "project_path", "env_scene") if launch_only[name] is None]
            if missing:
                raise ValueError(f"{', '.join(missing)} required to launch Godot (or editor=True)")
            envs_per_instance = 1 if envs_per_instance is None else envs_per_instance
            headless = True if headless is None else headless
        # Launch mode: any free port (passed to Godot). Editor mode: the fixed DEFAULT_PORT.
        server = socket.create_server(("127.0.0.1", DEFAULT_PORT if editor else 0))
        port = server.getsockname()[1]
        if editor:
            print(f"Waiting for {n_instances} Godot instance(s) on port {port}: in the editor, set the launch arguments "
                  f"-- --env-scene=res://... --envs-per-instance=N in Debug > Customize Run Instances, then run {BRIDGE_SCENE} (F6)", flush=True)

        try:
            if not editor:
                # --fixed-fps decouples the simulation from wall-clock time (runs as fast as
                # possible); it must match the physics rate to get one physics step per frame.
                fixed_fps = _project_physics_fps(project_path)
                for _ in range(n_instances):
                    cmd = [godot_exe, "--path", project_path]
                    if headless:
                        cmd.append("--headless")
                    cmd += ["--fixed-fps", str(fixed_fps), BRIDGE_SCENE, "--"]
                    cmd += [
                        f"--port={port}",
                        f"--envs-per-instance={envs_per_instance}",
                        f"--env-scene={env_scene}",
                    ]
                    if env_parameters:
                        cmd.append(f"--env-parameters={json.dumps(env_parameters)}")
                    if agent_parameters:
                        cmd.append(f"--agent-parameters={json.dumps(agent_parameters)}")
                    self._procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL))

            # Godot connects once its envs are created. Stop waiting if an instance exits (setup error)
            # or none connects in time (e.g. the C# project is not built: Godot runs without GymBridge).
            server.settimeout(1)
            deadline = time.monotonic() + CONNECT_TIMEOUT
            while len(self._socks) < n_instances:
                try:
                    sock, _ = server.accept()
                except TimeoutError:
                    if any(proc.poll() is not None for proc in self._procs):
                        raise RuntimeError("Godot exited before connecting: see its errors above") from None
                    if not editor and time.monotonic() > deadline:
                        raise TimeoutError(
                            f"Godot did not connect within {CONNECT_TIMEOUT} s. If it printed 'Cannot instantiate "
                            f"C# script', build the C# project: {godot_exe} --headless --path {project_path} "
                            "--build-solutions --quit") from None
                    continue
                self._socks.append(sock)  # so close() cleans it up if setup fails
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                hello = json.loads(P.recv_msg(sock, P.HELLO))
                if hello.get("protocol_version") != P.PROTOCOL_VERSION:
                    raise ConnectionError(
                        f"Godot addon protocol version {hello.get('protocol_version')}, Python expects "
                        f"{P.PROTOCOL_VERSION}: build the game's C# project again (Godot runs the last build)")

            # All instances run the same scene and config: any HELLO describes them all,
            self._n_per = hello["n_envs"]  # envs per instance
            self._slices = [slice(i * self._n_per, (i + 1) * self._n_per) for i in range(n_instances)]
            self.single_observation_space = spaces.Box(-np.inf, np.inf, shape=(hello["obs_size"],), dtype=np.float32)
            n = hello["action_size"]
            self.single_action_space = (spaces.Discrete(n) if hello["action_type"] == "discrete"
                                        else spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32))
            self.model_metadata = {
                "agent_parameters": hello["agent_parameters"],
                "physics_ticks_per_second": hello["physics_ticks_per_second"],
            }
        except BaseException:
            for proc in self._procs:
                proc.kill()  # setup failed: don't wait 5 s for each of the other instances
            self.close()
            raise
        finally:
            server.close()

        self.num_envs = self._n_per * n_instances
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

        self._obs_dim = self.single_observation_space.shape[0]
        self._discrete = isinstance(self.single_action_space, spaces.Discrete)
        self._act_dtype = np.int32 if self._discrete else np.float32
        # Fixed part of one instance's STEP_RESULT (the final obs of ended envs follow it).
        self._step_dtype = np.dtype([
            ("obs", np.float32, (self._n_per, self._obs_dim)),
            ("reward", np.float32, self._n_per),
            ("terminated", np.bool_, self._n_per),
            ("truncated", np.bool_, self._n_per),
        ])

    # ---------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        # Env i gets seed + i; -1 = no seed.
        seeds = np.full(self.num_envs, -1, np.int64) if seed is None else seed + np.arange(self.num_envs, dtype=np.int64)

        for sock, sl in zip(self._socks, self._slices):
            P.send_msg(sock, P.RESET, seeds[sl].tobytes())
        obs = np.empty((self.num_envs, self._obs_dim), dtype=np.float32)
        for sock, sl in zip(self._socks, self._slices):
            obs[sl] = np.frombuffer(P.recv_msg(sock, P.RESET_RESULT), dtype=np.float32).reshape(-1, self._obs_dim)
        return obs, {}

    def step(self, actions):
        actions = np.asarray(actions, dtype=self._act_dtype)
        # Send to all instances first so they simulate in parallel, then collect.
        for sock, sl in zip(self._socks, self._slices):
            P.send_msg(sock, P.STEP, actions[sl].tobytes())

        n_total = self.num_envs
        obs = np.empty((n_total, self._obs_dim), dtype=np.float32)
        rewards = np.empty(n_total, dtype=np.float32)
        terminated = np.empty(n_total, dtype=np.bool_)
        truncated = np.empty(n_total, dtype=np.bool_)
        final_obs = np.full(n_total, None, dtype=object)

        for sock, sl in zip(self._socks, self._slices):
            buf = P.recv_msg(sock, P.STEP_RESULT)
            msg = np.frombuffer(buf, self._step_dtype, count=1)[0]
            obs[sl] = msg["obs"]
            rewards[sl] = msg["reward"]
            terminated[sl] = msg["terminated"]
            truncated[sl] = msg["truncated"]
            done_idx = np.flatnonzero(msg["terminated"] | msg["truncated"])
            if len(done_idx):
                finals = np.frombuffer(buf, np.float32, offset=self._step_dtype.itemsize).reshape(-1, self._obs_dim)
                for row, i in zip(finals, done_idx):
                    final_obs[sl.start + i] = row.copy()

        infos: dict[str, Any] = {}
        done = terminated | truncated
        if done.any():
            infos = {"final_obs": final_obs, "_final_obs": done, "final_info": {}, "_final_info": done}
        return obs, rewards, terminated, truncated, infos

    def close_extras(self, **kwargs):
        for sock in self._socks:
            try:
                P.send_msg(sock, P.CLOSE)
            except OSError:
                pass
            sock.close()
        for proc in self._procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
