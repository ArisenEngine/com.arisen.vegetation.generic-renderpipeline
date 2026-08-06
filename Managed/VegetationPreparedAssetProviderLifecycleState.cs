using ArisenEngine.Resources.Serialization;

namespace ArisenEngine.Vegetation.GenericRenderPipeline;

internal sealed class VegetationPreparedAssetProviderLifecycleState
{
    private readonly object m_Gate = new();
    private readonly HashSet<RuntimeAssetResidencyKey> m_ReleaseTombstones = new();
    private RuntimePreparedAssetProviderMetrics m_PhysicalMetrics;
    private MetricsPublication m_PublishedMetrics = new(default);

    internal object Gate => m_Gate;

    internal bool RequestReleaseLocked(in RuntimeAssetResidencyKey key)
    {
        EnsureGateHeld();
        bool added = m_ReleaseTombstones.Add(key);
        if (added)
        {
            PublishMetricsLocked();
        }

        return added;
    }

    internal bool IsReleasePendingLocked(in RuntimeAssetResidencyKey key)
    {
        EnsureGateHeld();
        return m_ReleaseTombstones.Contains(key);
    }

    internal bool IsReleasePending(in RuntimeAssetResidencyKey key)
    {
        lock (m_Gate)
        {
            return m_ReleaseTombstones.Contains(key);
        }
    }

    internal void CompleteReleaseLocked(
        in RuntimeAssetResidencyKey key,
        in RuntimePreparedAssetProviderMetrics physicalMetrics)
    {
        EnsureGateHeld();
        if (!m_ReleaseTombstones.Remove(key))
        {
            throw new InvalidOperationException(
                $"Vegetation prepared-asset release was not pending for '{key}'.");
        }

        m_PhysicalMetrics = physicalMetrics;
        PublishMetricsLocked();
    }

    internal void PublishPhysicalMetrics(
        in RuntimePreparedAssetProviderMetrics physicalMetrics)
    {
        lock (m_Gate)
        {
            m_PhysicalMetrics = physicalMetrics;
            PublishMetricsLocked();
        }
    }

    internal void PublishPhysicalMetricsLocked(
        in RuntimePreparedAssetProviderMetrics physicalMetrics)
    {
        EnsureGateHeld();
        m_PhysicalMetrics = physicalMetrics;
        PublishMetricsLocked();
    }

    internal RuntimePreparedAssetProviderMetrics ReadMetrics() =>
        Volatile.Read(ref m_PublishedMetrics).Metrics;

    private void PublishMetricsLocked()
    {
        int pendingDisposalCount = checked(
            m_PhysicalMetrics.PendingDisposalCount + m_ReleaseTombstones.Count);
        var next = new RuntimePreparedAssetProviderMetrics(
            m_PhysicalMetrics.PreparedResourceCount,
            m_PhysicalMetrics.EstimatedGpuBytes,
            pendingDisposalCount,
            m_PhysicalMetrics.DescriptorCount);
        MetricsPublication current = Volatile.Read(ref m_PublishedMetrics);
        if (current.Metrics == next)
        {
            return;
        }

        Volatile.Write(ref m_PublishedMetrics, new MetricsPublication(next));
    }

    private void EnsureGateHeld()
    {
        if (!Monitor.IsEntered(m_Gate))
        {
            throw new InvalidOperationException(
                "Vegetation prepared-asset lifecycle gate must be held for this operation.");
        }
    }

    private sealed record MetricsPublication(
        RuntimePreparedAssetProviderMetrics Metrics);
}
