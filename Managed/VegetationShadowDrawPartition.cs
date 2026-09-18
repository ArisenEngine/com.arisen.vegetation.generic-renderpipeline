using ArisenEngine.Rendering;

namespace ArisenEngine.Vegetation.GenericRenderPipeline;

internal readonly record struct VegetationShadowDrawRange(int CascadeIndex, int Start, int Count)
{
    public int End => checked(Start + Count);
    public bool IsEmpty => Count == 0;
}

internal static class VegetationShadowDrawWorkPartition
{
    public const int MaximumDrawsPerWorkItem = 256;

    public static int GetWorkItemCount(in DirectionalShadowCascadeDrawRangeSet drawRanges)
    {
        int workItemCount = 0;
        for (int cascadeIndex = 0; cascadeIndex < drawRanges.Count; cascadeIndex++)
        {
            DirectionalShadowCascadeDrawRange range = drawRanges.GetRange(cascadeIndex);
            if (range.Count <= 0)
            {
                continue;
            }

            workItemCount = checked(
                workItemCount +
                ((range.Count + MaximumDrawsPerWorkItem - 1) / MaximumDrawsPerWorkItem));
        }

        return workItemCount;
    }

    public static bool TryGetRange(
        in DirectionalShadowCascadeDrawRangeSet drawRanges,
        int workItemIndex,
        out VegetationShadowDrawRange range)
    {
        range = default;
        if (workItemIndex < 0)
        {
            return false;
        }

        int remaining = workItemIndex;
        for (int cascadeIndex = 0; cascadeIndex < drawRanges.Count; cascadeIndex++)
        {
            DirectionalShadowCascadeDrawRange cascadeRange = drawRanges.GetRange(cascadeIndex);
            if (cascadeRange.Count <= 0)
            {
                continue;
            }

            int cascadeWorkItemCount =
                (cascadeRange.Count + MaximumDrawsPerWorkItem - 1) / MaximumDrawsPerWorkItem;
            if (remaining >= cascadeWorkItemCount)
            {
                remaining -= cascadeWorkItemCount;
                continue;
            }

            int start = checked(
                cascadeRange.Start + (remaining * MaximumDrawsPerWorkItem));
            int count = Math.Min(MaximumDrawsPerWorkItem, cascadeRange.End - start);
            range = new VegetationShadowDrawRange(cascadeIndex, start, count);
            return true;
        }

        return false;
    }
}
