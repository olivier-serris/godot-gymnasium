#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace GodotGymnasium;

/// <summary>
/// Controls agents with a trained neural network. Every ActionRepeat frames: reads all agents'
/// observations, runs the network once for all of them, applies each agent's action.
/// Agents: list them in Agents, or call Add(agent).
/// </summary>
public partial class NNController : Node
{

    /// res:// path, or a file path such as a run folder's policy.onnx (editor and debug runs only).
    /// Replaced by --model=PATH on the command line (after "--", applies to every NNController), e.g.
    ///   godot --path game res://main.tscn -- --model=/abs/path/runs/<run>/policy.onnx
    [Export(PropertyHint.File, "*.onnx")] public string ModelPath { get; set; } = "";
    /// Agents placed in the scene. Spawned ones: Add.
    [Export] public Godot.Collections.Array<ControllableAgent> Agents { get; set; } = new();

    OnnxPolicy? _policy;
    readonly List<ControllableAgent> _agents = new();   // driven now
    int _actionRepeat = 1;
    int _framesLeft = 0;

    public override void _EnterTree()
    {
        if (_policy != null) return; // already loaded (node moved in the tree)
        // Run before the agents' own _PhysicsProcess so the actions are used in the same frame.
        ProcessPhysicsPriority = int.MinValue;
        foreach (var arg in OS.GetCmdlineUserArgs())
            if (arg.StartsWith("--model=")) ModelPath = arg["--model=".Length..];
        _policy = new OnnxPolicy(ModelPath);
        GD.Print($"[NNController] {Name}: {ModelPath} ({DescribeRun(_policy.RunInfoJson)}), " +
                 $"agent parameters {_policy.AgentParametersJson}");
        if (_policy.PhysicsTicksPerSecond != Engine.PhysicsTicksPerSecond)
            GD.PushWarning($"[NNController] {Name}: model trained at {_policy.PhysicsTicksPerSecond} physics ticks/s, the game runs at " +
                           $"{Engine.PhysicsTicksPerSecond}: decisions are not at the trained rate");
        foreach (var agent in Agents) Add(agent);
    }

    static string DescribeRun(string runInfoJson)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(runInfoJson);
        var run = doc.RootElement;
        if (!run.TryGetProperty("run_name", out var name)) return "no run info";
        return $"run {name.GetString()}, {run.GetProperty("model").GetString()}, {run.GetProperty("timesteps")} timesteps";
    }

    /// Starts controlling agent and sets its [AgentParameter] values from the model.
    public void Add(ControllableAgent agent)
    {
        if (_policy == null)
            throw new InvalidOperationException($"{Name}: a model must be loaded before adding agents");
        if (_agents.Contains(agent))
            throw new InvalidOperationException($"{Name}: {agent.Name} is already added");

        if (agent.ActionSize != _policy.ActionSize)
            throw new InvalidOperationException(
                $"{Name}: {agent.Name} ActionSize {agent.ActionSize}, model {ModelPath} outputs {_policy.ActionSize}");
        agent.SetAgentParameters(_policy.AgentParametersJson, ModelPath, requireAll: true);
        _actionRepeat = agent.ActionRepeat; // just set from the model: the same for every agent
        _agents.Add(agent);
    }

    /// Stops driving agent; its last action stays applied. Call it before taking an agent
    /// out of the tree.
    public void Remove(ControllableAgent agent) => _agents.Remove(agent);

    public override void _PhysicsProcess(double delta)
    {
        if (_framesLeft > 0) _framesLeft--;
        bool isDecisionFrame = _framesLeft == 0;
        if (!isDecisionFrame) return;
        _framesLeft = _actionRepeat;

        _agents.RemoveAll(agent => !IsInstanceValid(agent)); // freed
        if (_agents.Count == 0) return;

        var observations = _agents.Select(agent => agent.GetObservation()).ToArray();
        var actions = _policy!.Act(observations);
        for (int i = 0; i < _agents.Count; i++)
            ApplyOutput(_agents[i], actions[i]);
    }

    /// output: Box action or Discrete logits.
    static void ApplyOutput(ControllableAgent agent, float[] output)
    {
        if (agent.ActionType == ActionType.Discrete)
        {
            int bestChoice = 0; // argmax of the logits
            for (int i = 1; i < output.Length; i++)
                if (output[i] > output[bestChoice]) bestChoice = i;
            agent.ApplyDiscreteAction(bestChoice);
        }
        else
        {
            agent.ApplyAction(output);
        }
    }

    public override void _Notification(int what)
    {
        // Freed, not just moved: release the model.
        if (what == NotificationPredelete) _policy?.Dispose();
    }
}
