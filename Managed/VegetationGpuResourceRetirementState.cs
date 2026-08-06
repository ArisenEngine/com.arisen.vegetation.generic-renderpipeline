namespace ArisenEngine.Vegetation.GenericRenderPipeline;

internal sealed class VegetationGpuResourceRetirementState
{
    private readonly object m_Gate = new();
    private readonly Queue<IVegetationClusterGpuResource> m_Pending = new();
    private readonly HashSet<IVegetationClusterGpuResource> m_Owned =
        new(ReferenceEqualityComparer.Instance);
    private int m_PendingCount;

    internal int PendingCount => Volatile.Read(ref m_PendingCount);

    internal bool RequestRelease(IVegetationClusterGpuResource resource)
    {
        ArgumentNullException.ThrowIfNull(resource);
        lock (m_Gate)
        {
            if (!m_Owned.Add(resource))
            {
                return false;
            }

            m_Pending.Enqueue(resource);
            Volatile.Write(ref m_PendingCount, m_Pending.Count);
            return true;
        }
    }

    internal void Drain(Action<IVegetationClusterGpuResource> retire)
    {
        ArgumentNullException.ThrowIfNull(retire);
        if (Volatile.Read(ref m_PendingCount) == 0)
        {
            return;
        }

        while (TryPeek(out IVegetationClusterGpuResource? resource))
        {
            retire(resource);
            Complete(resource);
        }
    }

    private bool TryPeek(out IVegetationClusterGpuResource resource)
    {
        lock (m_Gate)
        {
            if (m_Pending.Count == 0)
            {
                resource = null!;
                return false;
            }

            resource = m_Pending.Peek();
            return true;
        }
    }

    private void Complete(IVegetationClusterGpuResource resource)
    {
        lock (m_Gate)
        {
            if (m_Pending.Count == 0 ||
                !ReferenceEquals(m_Pending.Peek(), resource) ||
                !m_Owned.Remove(resource))
            {
                throw new InvalidOperationException(
                    "Vegetation GPU retirement ordering was lost.");
            }

            m_Pending.Dequeue();
            Volatile.Write(ref m_PendingCount, m_Pending.Count);
        }
    }
}
