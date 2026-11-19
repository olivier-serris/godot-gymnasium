#nullable enable
using Godot;

namespace GodotGymnasium;

/// <summary>
/// Training scene for one agent: sets up each episode, gives rewards, decides when it ends.
/// Not used in the game.
/// </summary>
public abstract partial class GymEnv : Node
{
    /// The agent being trained.
    [Export] public ControllableAgent Agent { get; set; } = null!;

    // ---- Task parameters ----------------------------------------------------

    /// Truncate after this many agent steps (0 = never). Counted by the bridge.
    [Export] public int MaxEpisodeSteps { get; set; } = 0;

    /// Agent steps since the last reset.
    public int EpisodeStep { get; internal set; }

    /// Seeded by the bridge on reset when Python passes a seed.
    protected RandomNumberGenerator Rng { get; } = new();

    /// Sets each [Export] property named in json (e.g. {"MaxEpisodeSteps": 500});
    /// the others keep their value. source: where json comes from, shown in the error message.
    public void SetEnvParameters(string json, string source) =>
        Parameters.Set(this, typeof(ExportAttribute), json, source, requireAll: false);

    // ---- Task logic (override) ---------------------------------------------
    /// Put the env in an initial state (Rng already seeded).
    public abstract void Reset();

    /// Reward for the last agent step.
    public abstract float GetReward();

    public abstract bool IsTerminated();

    // ---- Called by the bridge ----------------------------------------------
    internal void ResetInternal(long seed)
    {
        if (seed >= 0) Rng.Seed = (ulong)seed;
        EpisodeStep = 0;
        Reset();
    }
}
