namespace ArisenEngine.Vegetation.GenericRenderPipeline;

internal enum VegetationRenderValidationMode
{
    Full,
    OpaqueOnly,
    Disabled
}

internal static class VegetationRenderValidationPolicy
{
    public const string EnvironmentVariableName =
        "ARISEN_VEGETATION_RENDER_VALIDATION_MODE";

    /// <summary>
    /// Wind time shared by every process of a vegetation visual comparison. The compared runs are
    /// independent processes, so a live clock would give each of them a different gust phase and
    /// the frame-by-frame comparison would be meaningless. The value is deliberately non-zero so
    /// the compared frames still show wind-displaced vegetation instead of the rest pose.
    /// </summary>
    public const float ComparableWindTimeSeconds = 6.0f;

    public static VegetationRenderValidationSelection ResolveFromEnvironment()
    {
        string? value = Environment.GetEnvironmentVariable(EnvironmentVariableName);
        if (string.IsNullOrWhiteSpace(value))
        {
            return new VegetationRenderValidationSelection(
                VegetationRenderValidationMode.Full,
                pinsWindClock: false);
        }

        VegetationRenderValidationMode mode;
        if (string.Equals(value, "full", StringComparison.OrdinalIgnoreCase))
        {
            mode = VegetationRenderValidationMode.Full;
        }
        else if (string.Equals(value, "opaque-only", StringComparison.OrdinalIgnoreCase))
        {
            mode = VegetationRenderValidationMode.OpaqueOnly;
        }
        else if (string.Equals(value, "disabled", StringComparison.OrdinalIgnoreCase))
        {
            mode = VegetationRenderValidationMode.Disabled;
        }
        else
        {
            throw new InvalidOperationException(
                $"{EnvironmentVariableName} accepts only 'full', 'opaque-only', or 'disabled'; " +
                $"received '{value}'.");
        }

        // Every process launched by a comparison harness declares its mode, including the plain
        // 'full' run whose summary is the comparison baseline.
        return new VegetationRenderValidationSelection(mode, pinsWindClock: true);
    }
}
