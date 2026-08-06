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

    public static VegetationRenderValidationMode ResolveFromEnvironment()
    {
        string? value = Environment.GetEnvironmentVariable(EnvironmentVariableName);
        if (string.IsNullOrWhiteSpace(value) ||
            string.Equals(value, "full", StringComparison.OrdinalIgnoreCase))
        {
            return VegetationRenderValidationMode.Full;
        }
        if (string.Equals(value, "opaque-only", StringComparison.OrdinalIgnoreCase))
        {
            return VegetationRenderValidationMode.OpaqueOnly;
        }
        if (string.Equals(value, "disabled", StringComparison.OrdinalIgnoreCase))
        {
            return VegetationRenderValidationMode.Disabled;
        }

        throw new InvalidOperationException(
            $"{EnvironmentVariableName} accepts only 'full', 'opaque-only', or 'disabled'; " +
            $"received '{value}'.");
    }
}
