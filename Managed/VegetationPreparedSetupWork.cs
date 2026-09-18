using ArisenEngine.Resources.Serialization;
using ArisenEngine.Threading;
using ArisenEngine.Vegetation;

namespace ArisenEngine.Vegetation.GenericRenderPipeline;

/// <summary>
/// Resolves the prepared rendering view owned by a resident cluster generation.
/// </summary>
internal interface IVegetationPreparedClusterSource
{
    bool TryGetCluster(
        Guid clusterGuid,
        ulong generation,
        out VegetationPreparedClusterView cluster);
}

/// <summary>
/// Bounded setup ranges shared by the gather and prepared-draw setup phases.
/// </summary>
internal static class VegetationSetupWorkPartition
{
    public const int MaximumInputsPerWorkItem = 256;

    public static int GetWorkItemCount(int itemCount) =>
        itemCount <= 0
            ? 0
            : checked((itemCount + MaximumInputsPerWorkItem - 1) /
                MaximumInputsPerWorkItem);

    public static bool TryGetRange(
        int itemCount,
        int workItemIndex,
        out int start,
        out int count)
    {
        start = 0;
        count = 0;
        if (itemCount <= 0 || workItemIndex < 0)
        {
            return false;
        }

        int candidateStart = checked(workItemIndex * MaximumInputsPerWorkItem);
        if (candidateStart >= itemCount)
        {
            return false;
        }

        start = candidateStart;
        count = Math.Min(MaximumInputsPerWorkItem, itemCount - candidateStart);
        return true;
    }
}

/// <summary>
/// Preallocated per-shard output regions plus the ordered merge that reconstitutes
/// the deterministic serial output order.
/// </summary>
internal sealed class VegetationSetupShardBuffer<T>
{
    private T[] m_Regions = Array.Empty<T>();
    private int[] m_Counts = Array.Empty<int>();
    private int m_RegionCount;
    private int m_CapacityPerRegion;

    public int RegionCount => m_RegionCount;

    public void EnsureRegions(int regionCount, int capacityPerRegion)
    {
        if (regionCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(regionCount));
        }

        if (capacityPerRegion <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(capacityPerRegion));
        }

        int required = checked(regionCount * capacityPerRegion);
        if (m_Regions.Length < required)
        {
            int capacity = Math.Max(4, m_Regions.Length);
            while (capacity < required)
            {
                capacity = checked(capacity * 2);
            }

            Array.Resize(ref m_Regions, capacity);
        }

        if (m_Counts.Length < regionCount)
        {
            Array.Resize(ref m_Counts, regionCount);
        }

        m_RegionCount = regionCount;
        m_CapacityPerRegion = capacityPerRegion;
    }

    public Span<T> GetRegion(int regionIndex)
    {
        if ((uint)regionIndex >= (uint)m_RegionCount)
        {
            throw new ArgumentOutOfRangeException(nameof(regionIndex));
        }

        return new Span<T>(
            m_Regions,
            regionIndex * m_CapacityPerRegion,
            m_CapacityPerRegion);
    }

    public int GetCount(int regionIndex) => m_Counts[regionIndex];

    public void SetCount(int regionIndex, int count)
    {
        if (count < 0 || count > m_CapacityPerRegion)
        {
            throw new InvalidOperationException(
                $"Vegetation setup shard '{regionIndex}' produced {count} entries " +
                $"for a region capacity of {m_CapacityPerRegion}.");
        }

        m_Counts[regionIndex] = count;
    }

    public int MergeInto(Span<T> destination)
    {
        int total = 0;
        for (int regionIndex = 0; regionIndex < m_RegionCount; regionIndex++)
        {
            int count = m_Counts[regionIndex];
            if (count == 0)
            {
                continue;
            }

            if (total + count > destination.Length)
            {
                throw new InvalidOperationException(
                    "Vegetation setup shard output exceeds the prepared destination.");
            }

            new ReadOnlySpan<T>(
                m_Regions,
                regionIndex * m_CapacityPerRegion,
                count).CopyTo(destination[total..]);
            total += count;
        }

        return total;
    }
}

internal readonly struct VegetationPreparedClusterFrame
{
    public VegetationPreparedClusterFrame(
        in VegetationClusterComponent component,
        in VegetationPreparedClusterView prepared,
        in VegetationCullingSelection selection)
    {
        Component = component;
        Prepared = prepared;
        Selection = selection;
    }

    public VegetationClusterComponent Component { get; }

    public VegetationPreparedClusterView Prepared { get; }

    public VegetationCullingSelection Selection { get; }
}

/// <summary>
/// Runs setup work items on the shared task graph, reusing one task per work item across
/// frames. A missing task graph, a single work item, or an empty range runs the same work
/// item bodies inline so results never depend on the scheduling policy.
/// </summary>
internal sealed class VegetationSetupWorkDispatcher
{
    private readonly ITaskGraph? m_TaskSystem;
    private ActionTask[] m_Tasks = Array.Empty<ActionTask>();
    private Action<int>? m_WorkItem;
    private int m_ItemCount;

    public VegetationSetupWorkDispatcher(ITaskGraph? taskSystem)
    {
        m_TaskSystem = taskSystem;
    }

    public int Dispatch(int itemCount, Action<int> workItem)
    {
        ArgumentNullException.ThrowIfNull(workItem);
        if (m_WorkItem != null)
        {
            throw new InvalidOperationException(
                "Vegetation setup work cannot be dispatched while a dispatch is running.");
        }

        int workItemCount = VegetationSetupWorkPartition.GetWorkItemCount(itemCount);
        if (workItemCount == 0)
        {
            return 0;
        }

        m_ItemCount = itemCount;
        m_WorkItem = workItem;
        try
        {
            if (m_TaskSystem == null || workItemCount == 1)
            {
                for (int workItemIndex = 0;
                     workItemIndex < workItemCount;
                     workItemIndex++)
                {
                    RunWorkItem(workItemIndex);
                }

                return workItemCount;
            }

            EnsureTasks(workItemCount);
            for (int workItemIndex = 0; workItemIndex < workItemCount; workItemIndex++)
            {
                m_TaskSystem.AddTask(m_Tasks[workItemIndex]);
            }

            m_TaskSystem.Execute();
            return workItemCount;
        }
        finally
        {
            m_WorkItem = null;
        }
    }

    private void EnsureTasks(int workItemCount)
    {
        if (m_Tasks.Length >= workItemCount)
        {
            return;
        }

        int capacity = Math.Max(4, m_Tasks.Length);
        while (capacity < workItemCount)
        {
            capacity = checked(capacity * 2);
        }

        var tasks = new ActionTask[capacity];
        Array.Copy(m_Tasks, tasks, m_Tasks.Length);
        for (int index = m_Tasks.Length; index < capacity; index++)
        {
            int capturedWorkItemIndex = index;
            tasks[index] = new ActionTask(
                () => RunWorkItem(capturedWorkItemIndex),
                $"Vegetation.Setup[{capturedWorkItemIndex}]");
        }

        m_Tasks = tasks;
    }

    private void RunWorkItem(int workItemIndex)
    {
        Action<int> workItem = m_WorkItem
            ?? throw new InvalidOperationException(
                "Vegetation setup work item ran outside an active dispatch.");
        if (!VegetationSetupWorkPartition.TryGetRange(
                m_ItemCount,
                workItemIndex,
                out _,
                out _))
        {
            throw new InvalidOperationException(
                $"Vegetation setup work item '{workItemIndex}' is outside the " +
                $"dispatched range of {m_ItemCount} items.");
        }

        workItem(workItemIndex);
    }
}

/// <summary>
/// Pure per-range setup kernels. Each kernel is a function of its range only, so any
/// partition of the input produces the same ordered concatenation.
/// </summary>
internal static class VegetationPreparedSetup
{
    public static int GatherCullingInputs(
        ReadOnlySpan<VegetationClusterComponent> extractedClusters,
        int start,
        int count,
        ReadOnlySpan<VegetationResidentClusterData> residentClusters,
        IVegetationPreparedClusterSource preparedClusters,
        Span<VegetationClusterCullingInput> destination)
    {
        int written = 0;
        int end = checked(start + count);
        for (int componentIndex = start; componentIndex < end; componentIndex++)
        {
            ref readonly VegetationClusterComponent component =
                ref extractedClusters[componentIndex];
            if (!VegetationClusterLookup.TryFindResidentCluster(
                    residentClusters,
                    component.ClusterGuid,
                    out VegetationResidentClusterData resident) ||
                !MatchesComponent(component, resident) ||
                !preparedClusters.TryGetCluster(
                    component.ClusterGuid,
                    resident.Generation,
                    out VegetationPreparedClusterView prepared))
            {
                continue;
            }

            destination[written++] = new VegetationClusterCullingInput(
                component,
                resident,
                prepared);
        }

        return written;
    }

    public static int BuildPreparedFrames(
        ReadOnlySpan<VegetationClusterCullingInput> inputs,
        int start,
        int count,
        ReadOnlySpan<VegetationCullingSelection> selections,
        Span<VegetationPreparedClusterFrame> destination)
    {
        int written = 0;
        int end = checked(start + count);
        for (int inputIndex = start; inputIndex < end; inputIndex++)
        {
            ref readonly VegetationClusterCullingInput input = ref inputs[inputIndex];
            if (!VegetationClusterLookup.TryFindSelection(
                    selections,
                    input.Resident.Guid,
                    out VegetationCullingSelection selection) ||
                !selection.Accepted)
            {
                continue;
            }

            destination[written++] = new VegetationPreparedClusterFrame(
                input.Component,
                input.Prepared,
                selection);
        }

        return written;
    }

    public static bool MatchesComponent(
        in VegetationClusterComponent component,
        VegetationResidentClusterData resident) =>
        resident.BiomeGuid == component.BiomeGuid &&
        resident.ContainsSpecies(component.SpeciesGuid) &&
        resident.Origin == new WorldPosition(
            component.OriginX,
            component.OriginY,
            component.OriginZ) &&
        resident.PageCount == component.PageCount &&
        resident.InstanceCount == component.InstanceCount;
}
