#nullable enable
using System;
using System.Collections.Generic;
using Godot;

namespace GodotGymnasium;

public enum ActionType { Box, Discrete }

/// <summary>
/// it provides observations and applies actions
/// </summary>
public abstract partial class ControllableAgent : Node
{
    // ---- Spaces (override) -------------------------------------------------
    public abstract ActionType ActionType { get; }
    /// Box: number of floats. Discrete: number of choices (n).
    public abstract int ActionSize { get; }

    /// Physics frames an action is held before the next decision.
    [Export, AgentParameter] public int ActionRepeat { get; set; } = 1;

    // ---- Agent logic (override) --------------------------------------------
    /// Same length on every call.
    public abstract float[] GetObservation();

    /// Box action: length ActionSize, values in [-1, 1].
    public virtual void ApplyAction(ReadOnlySpan<float> action) =>
        throw new NotImplementedException("Override ApplyAction for Box actions.");

    /// Discrete action, in [0, ActionSize).
    public virtual void ApplyDiscreteAction(int action) =>
        throw new NotImplementedException("Override ApplyDiscreteAction for Discrete actions.");

    // ---- Agent parameters ---------------------------------------------------
    /// Current value of each [AgentParameter] property (e.g. {"Speed": 200, "ActionRepeat": 4}).
    public Dictionary<string, object?> GetAgentParameters()
    {
        var values = new Dictionary<string, object?>();
        foreach (var p in Parameters.Find(this, typeof(AgentParameterAttribute)).Values)
        {
            // The values are written as JSON into the model file and set back by NNController;
            // only bool, number and string types come back unchanged.
            bool isJsonRoundTripSafe = p.PropertyType.IsPrimitive || p.PropertyType == typeof(string);
            if (!isJsonRoundTripSafe || !p.CanWrite)
                throw new InvalidOperationException(
                    $"[AgentParameter] {GetType().Name}.{p.Name}: must be a bool, a number or a string, with a public setter");
            values[p.Name] = p.GetValue(this);
        }
        return values;
    }

    /// Sets each [AgentParameter] property named in json. requireAll: json must name them all.
    /// source: where json comes from, shown in the error message.
    public void SetAgentParameters(string json, string source, bool requireAll) =>
        Parameters.Set(this, typeof(AgentParameterAttribute), json, source, requireAll);
}
