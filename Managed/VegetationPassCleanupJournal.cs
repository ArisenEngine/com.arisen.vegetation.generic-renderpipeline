namespace ArisenEngine.Vegetation.GenericRenderPipeline;

internal struct VegetationPassCleanupJournal
{
    private ulong m_CompletedLegs;

    internal void BeginOwnership() => m_CompletedLegs = 0;

    internal void Release(int legIndex, Action release)
    {
        ArgumentNullException.ThrowIfNull(release);
        if ((uint)legIndex >= 64u)
        {
            throw new ArgumentOutOfRangeException(nameof(legIndex));
        }

        ulong leg = 1UL << legIndex;
        if ((m_CompletedLegs & leg) != 0)
        {
            return;
        }

        release();
        m_CompletedLegs |= leg;
    }
}
