using System;

namespace GodotGymnasium;

/// <summary>
/// Marks an agent property whose training value must be reused in the game:
/// settable from Python (agent_parameters / --agent-parameters), saved in the .onnx file,
/// set back by NNController.
/// NNController may set it after _Ready: read the property where you use it, don't cache
/// values computed from it.
/// </summary>
[AttributeUsage(AttributeTargets.Property)]
public sealed class AgentParameterAttribute : Attribute { }
