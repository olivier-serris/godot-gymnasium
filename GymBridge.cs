#nullable enable
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text.Json;
using Godot;

namespace GodotGymnasium;

/// <summary>
/// Runs N copies of a training scene (GymEnv) and lets Python (godot_gymnasium.GodotVectorEnv) step
/// them over TCP.
/// For debug, from the editor: start Python first (GodotVectorEnv(editor=True), e.g. train.py --editor),
/// then run bridge.tscn (F6) with the launch arguments below set in Debug > Customize Run Instances
/// (don't edit bridge.tscn: it is replaced when the addon is updated).
///
/// When Python launches Godot, it appends these arguments after "--" on Godot's command line
/// (Godot passes everything after "--" through to OS.GetCmdlineUserArgs()):
///   --port=PORT          TCP port Python listens on (default 11008)
///   --envs-per-instance=N  number of env copies to run (default 1)
///   --env-scene=res://..   the GymEnv scene to instantiate (required, also in editor runs)
///   --env-parameters=JSON    {"Property": value} set on each GymEnv's [Export] properties
///   --agent-parameters=JSON  same, on each env's ControllableAgent [AgentParameter] properties
/// Example: godot --path game --fixed-fps 60 res://addons/godot_gymnasium/bridge.tscn -- --port=11008 --envs-per-instance=64 --env-scene=res://env.tscn
///
/// Wire format (little-endian): [u8 type][u32 payload length][payload]
///   HELLO        G->P  JSON {protocol_version, n_envs, obs_size, action_type ("box" or "discrete"), action_size,
///                      agent_parameters, physics_ticks_per_second}
///   RESET        P->G  i64[n] seeds (-1 = no seed)
///   RESET_RESULT G->P  f32[n*obs]
///   STEP         P->G  f32[n*act] (Box) or i32[n] (Discrete)
///   STEP_RESULT  G->P  f32[n*obs] obs, f32[n] reward, u8[n] terminated, u8[n] truncated,
///                      f32[k*obs] final obs of the k envs that ended (in env order)
///   CLOSE        P->G  empty
/// Envs that end are reset immediately (gymnasium "same-step" autoreset).
/// Keep in sync with python/godot_gymnasium/protocol.py.
/// </summary>
public partial class GymBridge : Node
{
    const int ProtocolVersion = 2; // bump when the messages change, = PROTOCOL_VERSION in protocol.py
    const byte Hello = 1, ResetCmd = 2, ResetResult = 3, StepCmd = 4, StepResult = 5, Close = 6;
    const int Header = 5, F32 = sizeof(float), U8 = 1; // sizes in bytes

    PackedScene _envScene = null!; // --env-scene
    int _envsPerInstance = 1;      // --envs-per-instance

    int _port = 11008; // editor-run port, must match DEFAULT_PORT in python/godot_gymnasium/godot_vector_env.py
    string _envParameters = "{}", _agentParameters = "{}"; // JSON objects
    readonly List<GymEnv> _envs = new();
    readonly List<ControllableAgent> _agents = new(); // _agents[i] == _envs[i].Agent
    int _obsSize, _actSize, _actionRepeat;
    bool _isDiscrete;

    TcpClient _client = null!;
    NetworkStream _stream = null!;
    byte[] _recvBuf = Array.Empty<byte>();
    byte[] _sendBuf = Array.Empty<byte>();
    readonly byte[] _header = new byte[5];

    int _framesLeft = 0; // > 0 while an agent step is being simulated
    bool _closed = false;

    public override void _Ready()
    {
        // Run before the envs' own _PhysicsProcess so actions are applied in the same frame.
        ProcessPhysicsPriority = int.MinValue;
        try
        {
            ParseArgs();
            CreateEnvs();
            Connect();
            SendHello();
        }
        catch (Exception e)
        {
            GD.PushError($"[GymBridge] Setup failed: {e}"); // {e}: with the stack trace (file and line)
            Shutdown(1);
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        if (_closed) return;
        try
        {
            if (_framesLeft > 0)
            {
                _framesLeft--;
                if (_framesLeft > 0) return;
                SendStepResult();
            }
            HandleCommands();
        }
        catch (Exception e)
        {
            GD.PushError($"[GymBridge] {e}");
            Shutdown(1);
        }
    }

    // ---- Setup -----------------------------------------------------------------

    void ParseArgs()
    {
        foreach (var arg in OS.GetCmdlineUserArgs())
        {
            var parts = arg.TrimStart('-').Split('=', 2);
            if (parts.Length != 2) continue;
            var (key, val) = (parts[0], parts[1]);
            switch (key)
            {
                case "port": _port = int.Parse(val); break;
                case "envs-per-instance": _envsPerInstance = int.Parse(val); break;
                case "env-scene":
                    if (!ResourceLoader.Exists(val)) throw new InvalidOperationException($"--env-scene={val}: file not found");
                    _envScene = GD.Load<PackedScene>(val);
                    break;
                case "env-parameters": _envParameters = val; break;
                case "agent-parameters": _agentParameters = val; break;
            }
        }
        if (_envScene == null) throw new InvalidOperationException("No env scene: pass --env-scene=res://... (editor: Debug > Customize Run Instances)");
        if (_envsPerInstance < 1) throw new InvalidOperationException("--envs-per-instance must be >= 1");
    }

    void CreateEnvs()
    {
        bool headless = DisplayServer.GetName() == "headless";
        // With a window, every env is shown as a tile of a grid.
        GridContainer? grid = null;
        if (!headless)
        {
            grid = new GridContainer { Columns = (int)Math.Ceiling(Math.Sqrt(_envsPerInstance)) };
            grid.SetAnchorsPreset(Control.LayoutPreset.FullRect);
            AddChild(grid);
        }
        for (int i = 0; i < _envsPerInstance; i++)
        {
            // SubViewport: env copies never collide.
            var viewport = new SubViewport
            {
                Name = $"Env{i}",
                OwnWorld3D = true,
                RenderTargetUpdateMode = headless ? SubViewport.UpdateMode.Disabled : SubViewport.UpdateMode.Always,
                // 2D: the env sees the project's window size (by default a SubViewport is 512x512, or a grid tile).
                Size2DOverride = GetTree().Root.ContentScaleSize,
                Size2DOverrideStretch = true,
            };
            if (_envScene.Instantiate() is not GymEnv env)
                throw new InvalidOperationException($"Root of {_envScene.ResourcePath} must extend GymEnv");
            var agent = env.Agent ?? throw new InvalidOperationException($"{_envScene.ResourcePath}: Agent is not set");
            // Before AddChild, so _Ready sees the overrides.
            env.SetEnvParameters(_envParameters, "--env-parameters");
            agent.SetAgentParameters(_agentParameters, "--agent-parameters", requireAll: false);
            viewport.AddChild(env);

            if (grid == null)
            {
                AddChild(viewport);
            }
            else
            {
                // Stretch: the viewport renders at tile size, so small tiles stay cheap.
                var container = new SubViewportContainer
                {
                    Stretch = true,
                    SizeFlagsHorizontal = Control.SizeFlags.ExpandFill,
                    SizeFlagsVertical = Control.SizeFlags.ExpandFill,
                };
                container.AddChild(viewport);
                grid.AddChild(container);
            }
            _envs.Add(env);
            _agents.Add(agent);
        }

        var a0 = _agents[0];
        _obsSize = a0.GetObservation().Length; // envs are ready: their agents can observe
        _isDiscrete = a0.ActionType == ActionType.Discrete;
        _actSize = _isDiscrete ? 1 : a0.ActionSize;
        _actionRepeat = a0.ActionRepeat;

        // Largest outgoing message: STEP_RESULT with every env ending.
        int perEnv = _obsSize * F32   // obs
                   + F32              // reward
                   + U8 + U8          // terminated, truncated
                   + _obsSize * F32;  // final obs (worst case: every env ended)
        _sendBuf = new byte[Header + _envsPerInstance * perEnv];
    }

    void Connect()
    {
        _client = new TcpClient("127.0.0.1", _port) { NoDelay = true };
        _stream = _client.GetStream();
    }

    void SendHello()
    {
        var a0 = _agents[0];
        var hello = new
        {
            protocol_version = ProtocolVersion,
            n_envs = _envsPerInstance,
            obs_size = _obsSize,
            action_type = _isDiscrete ? "discrete" : "box",
            action_size = a0.ActionSize,
            // Read after _Ready: the values training really uses.
            agent_parameters = a0.GetAgentParameters(),
            physics_ticks_per_second = Engine.PhysicsTicksPerSecond,
        };
        var json = JsonSerializer.SerializeToUtf8Bytes(hello);
        // Sent once: written directly, _sendBuf is for the per-step messages.
        var header = new byte[5];
        header[0] = Hello;
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(1), (uint)json.Length);
        _stream.Write(header);
        _stream.Write(json);
    }

    // ---- Command loop ------------------------------------------------------------

    /// Blocks until a STEP (then returns so physics can run) or CLOSE.
    void HandleCommands()
    {
        while (true)
        {
            var (type, len) = ReadMessage();
            switch (type)
            {
                case ResetCmd:
                    {
                        var seeds = MemoryMarshal.Cast<byte, long>(_recvBuf.AsSpan(0, len));
                        var obs = MemoryMarshal.Cast<byte, float>(_sendBuf.AsSpan(5, _envs.Count * _obsSize * 4));
                        for (int i = 0; i < _envs.Count; i++)
                        {
                            _envs[i].ResetInternal(seeds[i]);
                            CopyObservation(_agents[i], obs.Slice(i * _obsSize, _obsSize));
                        }
                        Send(ResetResult, obs.Length * 4);
                        break;
                    }
                case StepCmd:
                    {
                        var payload = _recvBuf.AsSpan(0, len);
                        if (_isDiscrete)
                        {
                            var acts = MemoryMarshal.Cast<byte, int>(payload);
                            for (int i = 0; i < _agents.Count; i++) _agents[i].ApplyDiscreteAction(acts[i]);
                        }
                        else
                        {
                            var acts = MemoryMarshal.Cast<byte, float>(payload);
                            for (int i = 0; i < _agents.Count; i++)
                                _agents[i].ApplyAction(acts.Slice(i * _actSize, _actSize));
                        }
                        _framesLeft = _actionRepeat;
                        return;
                    }
                case Close:
                    Shutdown(0);
                    return;
                default:
                    throw new InvalidOperationException($"Unknown message type {type}");
            }
        }
    }

    void CopyObservation(ControllableAgent agent, Span<float> row)
    {
        var obs = agent.GetObservation();
        if (obs.Length != row.Length)
            throw new InvalidOperationException($"{agent.Name} observation has {obs.Length} values, expected {row.Length}");
        obs.CopyTo(row);
    }

    void SendStepResult()
    {
        int nEnvs = _envs.Count, obsDim = _obsSize;
        // Byte offsets in _sendBuf: each part starts where the previous one ends.
        int obsAt = Header;                        // f32[nEnvs * obsDim] obs
        int rewAt = obsAt + nEnvs * obsDim * F32;  // f32[nEnvs] reward
        int termAt = rewAt + nEnvs * F32;          // u8[nEnvs] terminated
        int truncAt = termAt + nEnvs * U8;         // u8[nEnvs] truncated
        int finalAt = truncAt + nEnvs * U8;        // f32[nEnded * obsDim] final obs of the envs that ended
        var obs = MemoryMarshal.Cast<byte, float>(_sendBuf.AsSpan(obsAt, nEnvs * obsDim * F32));
        var rew = MemoryMarshal.Cast<byte, float>(_sendBuf.AsSpan(rewAt, nEnvs * F32));
        var term = _sendBuf.AsSpan(termAt, nEnvs * U8);
        var trunc = _sendBuf.AsSpan(truncAt, nEnvs * U8);
        var finals = MemoryMarshal.Cast<byte, float>(_sendBuf.AsSpan(finalAt));
        int nEnded = 0;

        for (int i = 0; i < nEnvs; i++)
        {
            var env = _envs[i];
            var agent = _agents[i];
            env.EpisodeStep++;
            var row = obs.Slice(i * obsDim, obsDim);
            CopyObservation(agent, row);
            rew[i] = env.GetReward();
            bool terminated = env.IsTerminated();
            bool truncated = !terminated && env.MaxEpisodeSteps > 0 && env.EpisodeStep >= env.MaxEpisodeSteps;
            term[i] = (byte)(terminated ? 1 : 0);
            trunc[i] = (byte)(truncated ? 1 : 0);
            if (terminated || truncated)
            {
                row.CopyTo(finals.Slice(nEnded * obsDim, obsDim));
                nEnded++;
                env.ResetInternal(-1);
                CopyObservation(agent, row);
            }
        }
        int end = finalAt + nEnded * obsDim * F32;
        Send(StepResult, end - Header);
    }

    // ---- IO ----------------------------------------------------------------------

    (byte type, int len) ReadMessage()
    {
        _stream.ReadExactly(_header);
        byte type = _header[0];
        int len = (int)BinaryPrimitives.ReadUInt32LittleEndian(_header.AsSpan(1));
        if (_recvBuf.Length < len) _recvBuf = new byte[len];
        _stream.ReadExactly(_recvBuf, 0, len);
        return (type, len);
    }

    /// Payload must already be at _sendBuf[5..5+len].
    void Send(byte type, int len)
    {
        _sendBuf[0] = type;
        BinaryPrimitives.WriteUInt32LittleEndian(_sendBuf.AsSpan(1), (uint)len);
        _stream.Write(_sendBuf, 0, 5 + len);
    }

    void Shutdown(int exitCode)
    {
        _closed = true;
        GetTree().Quit(exitCode);
    }
}
