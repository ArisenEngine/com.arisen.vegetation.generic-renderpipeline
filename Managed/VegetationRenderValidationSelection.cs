namespace ArisenEngine.Vegetation.GenericRenderPipeline;

/// <summary>
/// Resolved vegetation render-validation request: which passes a frame is allowed to run, and
/// whether the run participates in a cross-process visual comparison.
/// </summary>
internal readonly struct VegetationRenderValidationSelection
{
    public VegetationRenderValidationSelection(
        VegetationRenderValidationMode mode,
        bool pinsWindClock)
    {
        Mode = mode;
        PinsWindClock = pinsWindClock;
    }

    public VegetationRenderValidationMode Mode { get; }

    /// <summary>
    /// True for every process launched by a vegetation visual comparison harness. Those processes
    /// are compared pixel by pixel after they have exited, so each of them pins the wind clock to
    /// <see cref="VegetationRenderValidationPolicy.ComparableWindTimeSeconds"/>: the wind is the
    /// only time-driven renderer input, and separate processes would otherwise observe different
    /// wall-clock gust phases and render different vegetation.
    /// </summary>
    public bool PinsWindClock { get; }
}
