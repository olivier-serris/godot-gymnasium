#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.Json;

namespace GodotGymnasium;

/// <summary>
/// Finds the public properties of an object marked with an attribute, and sets them from JSON.
/// Used by ControllableAgent ([AgentParameter]) and GymEnv ([Export]).
/// </summary>
internal static class Parameters
{
    public static Dictionary<string, PropertyInfo> Find(object target, Type attribute) =>
        target.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(p => p.IsDefined(attribute, inherit: true))
            .ToDictionary(p => p.Name);

    /// Sets each property marked with attribute and named in json (e.g. {"Speed": 200}).
    /// An unknown name throws. requireAll: every marked property must be in json.
    /// source: where json comes from, shown in the error message.
    public static void Set(object target, Type attribute, string json, string source, bool requireAll)
    {
        var props = Find(target, attribute);
        using var doc = JsonDocument.Parse(json);
        var values = doc.RootElement.EnumerateObject().ToDictionary(e => e.Name, e => e.Value);

        string expected = $"{target.GetType().Name} has [{string.Join(", ", props.Keys)}]";
        foreach (var name in values.Keys)
            if (!props.ContainsKey(name))
                throw new InvalidOperationException($"{source}: unknown parameter '{name}', {expected}");
        if (requireAll && values.Count != props.Count)
            throw new InvalidOperationException($"{source} has [{string.Join(", ", values.Keys)}], {expected}");

        foreach (var (name, value) in values)
        {
            var p = props[name];
            object? v;
            try { v = value.Deserialize(p.PropertyType); }
            catch (Exception e) when (e is JsonException or NotSupportedException)
            {
                throw new InvalidOperationException($"{source}: {target.GetType().Name}.{name} = {value} is not a {p.PropertyType.Name}");
            }
            p.SetValue(target, v);
        }
    }
}
