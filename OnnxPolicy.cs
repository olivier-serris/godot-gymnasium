#nullable enable
using System;
using System.Collections.Generic;
using Godot;
using Microsoft.ML.OnnxRuntime;

namespace GodotGymnasium;

/// <summary>
/// Runs a trained neural network (.onnx file made by godot_gymnasium.sb3.export_onnx): observations of a batch
/// of agents in, one action per agent out.
/// </summary>
public sealed class OnnxPolicy : IDisposable
{
    // Keys of the .onnx metadata written by godot_gymnasium.sb3.export_onnx (python/godot_gymnasium/sb3.py).
    const string AgentParametersKey = "godot_gymnasium.agent_parameters";
    const string PhysicsTicksKey = "godot_gymnasium.physics_ticks_per_second";
    const string RunKey = "godot_gymnasium.run";

    /// The [AgentParameter] values saved at training, written into the .onnx by export_onnx.
    public string AgentParametersJson { get; }
    /// The physics rate during training, written into the .onnx by export_onnx.
    public int PhysicsTicksPerSecond { get; }
    /// Information only, JSON: run_name, model, timesteps, exported_at.
    /// "{}" for a model exported without it.
    public string RunInfoJson { get; }
    /// Size of one agent's output: Box action length or number of Discrete choices.
    public int ActionSize => _session.OutputMetadata["action"].Dimensions[1];

    readonly InferenceSession _session;
    readonly RunOptions _runOptions = new();

    /// path: res:// or user:// (works from an exported .pck), or a file path (editor and debug runs).
    public OnnxPolicy(string path)
    {
        byte[] bytes = Godot.FileAccess.GetFileAsBytes(path);
        if (bytes.Length == 0)
            throw new ArgumentException($"Cannot read ONNX model '{path}': {Godot.FileAccess.GetOpenError()}");

        // One thread: the network runs on Godot's main thread only, never on other cores.
        using var options = new SessionOptions { IntraOpNumThreads = 1 };
        _session = new InferenceSession(bytes, options);

        var metadata = _session.ModelMetadata.CustomMetadataMap;
        AgentParametersJson = metadata[AgentParametersKey];
        PhysicsTicksPerSecond = int.Parse(metadata[PhysicsTicksKey]);
        RunInfoJson = metadata.TryGetValue(RunKey, out var runInfo) ? runInfo : "{}";
    }

    /// One observation per agent in, one action per agent out (same order).
    public float[][] Act(float[][] observations)
    {
        int batchSize = observations.Length, obsSize = observations[0].Length;
        // The network takes one flat array: the observations end to end.
        var flatObs = new float[batchSize * obsSize];
        for (int i = 0; i < batchSize; i++)
        {
            if (observations[i].Length != obsSize)
                throw new ArgumentException($"observation {i} has {observations[i].Length} values, observation 0 has {obsSize}");
            observations[i].CopyTo(flatObs, i * obsSize);
        }

        // ONNX Runtime throws if obsSize doesn't match the network's input.
        using var input = OrtValue.CreateTensorValueFromMemory(flatObs, new long[] { batchSize, obsSize });
        // "obs" and "action": the tensor names set by export_onnx.
        using var outputs = _session.Run(_runOptions, new[] { "obs" }, new[] { input }, new[] { "action" });
        var flatActions = outputs[0].GetTensorDataAsSpan<float>();

        int actSize = flatActions.Length / batchSize;
        var actions = new float[batchSize][];
        for (int i = 0; i < batchSize; i++)
            actions[i] = flatActions.Slice(i * actSize, actSize).ToArray();
        return actions;
    }

    public void Dispose()
    {
        _runOptions.Dispose();
        _session.Dispose();
    }
}
