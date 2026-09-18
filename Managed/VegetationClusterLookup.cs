using ArisenEngine.Vegetation;

namespace ArisenEngine.Vegetation.GenericRenderPipeline;

internal static class VegetationClusterLookup
{
    public static bool TryFindResidentCluster(
        ReadOnlySpan<VegetationResidentClusterData> orderedClusters,
        Guid clusterGuid,
        out VegetationResidentClusterData cluster)
    {
        int low = 0;
        int high = orderedClusters.Length - 1;
        while (low <= high)
        {
            int middle = low + ((high - low) >> 1);
            VegetationResidentClusterData candidate = orderedClusters[middle];
            int comparison = candidate.Guid.CompareTo(clusterGuid);
            if (comparison < 0)
            {
                low = middle + 1;
                continue;
            }

            if (comparison > 0)
            {
                high = middle - 1;
                continue;
            }

            cluster = candidate;
            return true;
        }

        cluster = null!;
        return false;
    }

    public static bool TryFindSelection(
        ReadOnlySpan<VegetationCullingSelection> orderedSelections,
        Guid clusterGuid,
        out VegetationCullingSelection selection)
    {
        int low = 0;
        int high = orderedSelections.Length - 1;
        while (low <= high)
        {
            int middle = low + ((high - low) >> 1);
            int comparison = orderedSelections[middle].ClusterGuid.CompareTo(clusterGuid);
            if (comparison < 0)
            {
                low = middle + 1;
                continue;
            }

            if (comparison > 0)
            {
                high = middle - 1;
                continue;
            }

            while (middle > 0 && orderedSelections[middle - 1].ClusterGuid == clusterGuid)
            {
                middle--;
            }

            selection = orderedSelections[middle];
            return true;
        }

        selection = default;
        return false;
    }
}
