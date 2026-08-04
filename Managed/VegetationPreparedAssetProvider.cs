using ArisenEngine.Core.Assets;
using ArisenEngine.Resources.Serialization;
using ArisenEngine.Threading;
using ArisenEngine.Vegetation.Assets;

namespace ArisenEngine.Vegetation.GenericRenderPipeline;

internal sealed class VegetationPreparedAssetProvider :
    IRuntimePreparedAssetProvider,
    IDisposable
{
    public const string Id =
        "com.arisen.vegetation.generic-renderpipeline.prepared-assets";

    private readonly object m_Gate = new();
    private readonly object m_DisposeGate = new();
    private readonly IAssetDatabase m_AssetDatabase;
    private readonly IBackgroundTaskScheduler m_BackgroundTasks;
    private readonly IVegetationRuntimeDataStore m_RuntimeData;
    private readonly IRuntimeAssetResidencyService m_ResidencyService;
    private readonly Dictionary<RuntimeAssetResidencyKey, PreparedResource> m_Prepared = new();
    private readonly Dictionary<RuntimeAssetResidencyKey, PendingPreparation> m_Pending = new();
    private readonly HashSet<BackgroundTask<WorkerPreparedResource>> m_Outstanding = new();
    private int m_PreparedResourceCount;
    private bool m_Disposed;

    public VegetationPreparedAssetProvider(
        IAssetDatabase assetDatabase,
        IBackgroundTaskScheduler backgroundTasks,
        IVegetationRuntimeDataStore runtimeData,
        IRuntimeAssetResidencyService residencyService)
    {
        m_AssetDatabase = assetDatabase ?? throw new ArgumentNullException(nameof(assetDatabase));
        m_BackgroundTasks = backgroundTasks
            ?? throw new ArgumentNullException(nameof(backgroundTasks));
        m_RuntimeData = runtimeData ?? throw new ArgumentNullException(nameof(runtimeData));
        m_ResidencyService = residencyService
            ?? throw new ArgumentNullException(nameof(residencyService));
    }

    public string ProviderId => Id;

    public bool Supports(string assetType) =>
        assetType is VegetationAssetTypes.Cluster or
            VegetationAssetTypes.InstancePage or
            VegetationAssetTypes.Species or
            VegetationAssetTypes.Biome;

    public RuntimePreparedAssetResult Prepare(RuntimeAssetResidencyKey key)
    {
        if (!Supports(key.AssetType))
        {
            return RuntimePreparedAssetResult.Failed(
                $"Vegetation GenericRP does not prepare asset type '{key.AssetType}'.");
        }

        string expectedVariant = GetExpectedVariant(key.AssetType);
        if (!string.Equals(key.Variant, expectedVariant, StringComparison.Ordinal))
        {
            return RuntimePreparedAssetResult.Failed(
                $"Vegetation asset '{key}' uses unsupported variant '{key.Variant}'; " +
                $"expected '{expectedVariant}'.");
        }

        try
        {
            lock (m_Gate)
            {
                if (m_Disposed)
                {
                    return RuntimePreparedAssetResult.Failed(
                        "Vegetation prepared assets have been disposed.");
                }

                if (m_Prepared.ContainsKey(key))
                {
                    return RuntimePreparedAssetResult.Ready(estimatedGpuBytes: 0);
                }
            }

            if (!TryCaptureResidencyClaim(
                    key,
                    out RuntimeAssetPreparationClaim claim,
                    out string claimDiagnostic))
            {
                return RuntimePreparedAssetResult.Waiting(claimDiagnostic);
            }

            lock (m_Gate)
            {
                if (m_Disposed)
                {
                    return RuntimePreparedAssetResult.Failed(
                        "Vegetation prepared assets have been disposed.");
                }

                PruneCompletedOrphansLocked();
                if (m_Prepared.ContainsKey(key))
                {
                    return RuntimePreparedAssetResult.Ready(estimatedGpuBytes: 0);
                }

                if (m_Pending.TryGetValue(key, out PendingPreparation? pending))
                {
                    if (pending.Claim != claim)
                    {
                        m_Pending.Remove(key);
                        pending.Task.Cancel();
                        return StartPreparationLocked(key, claim);
                    }

                    return ClaimPreparationLocked(key, pending);
                }

                return StartPreparationLocked(key, claim);
            }
        }
        catch (Exception ex) when (
            ex is InvalidOperationException or InvalidDataException or ArgumentException)
        {
            return RuntimePreparedAssetResult.Failed(
                $"Vegetation prepared-resource setup failed for '{key}': {ex.Message}");
        }
    }

    public void Release(RuntimeAssetResidencyKey key)
    {
        BackgroundTask<WorkerPreparedResource>? task = null;
        lock (m_Gate)
        {
            if (m_Pending.Remove(key, out PendingPreparation? pending))
            {
                task = pending.Task;
            }

            PruneCompletedOrphansLocked();
        }

        task?.Cancel();
        ReleasePreparedResource(key);
    }

    public RuntimePreparedAssetProviderMetrics GetMetrics() => new(
        Volatile.Read(ref m_PreparedResourceCount),
        EstimatedGpuBytes: 0,
        PendingDisposalCount: 0);

    public void Dispose()
    {
        lock (m_DisposeGate)
        {
            BackgroundTask<WorkerPreparedResource>[] tasks;
            lock (m_Gate)
            {
                m_Disposed = true;
                tasks = m_Outstanding
                    .Concat(m_Pending.Values.Select(pending => pending.Task))
                    .Distinct()
                    .ToArray();
                m_Pending.Clear();
                m_Outstanding.Clear();
            }

            var cleanupFailures = new List<Exception>();
            for (int index = 0; index < tasks.Length; index++)
            {
                try
                {
                    tasks[index].Cancel();
                }
                catch (Exception ex)
                {
                    cleanupFailures.Add(new InvalidOperationException(
                        "Vegetation background preparation cancellation failed during disposal.",
                        ex));
                }
            }

            for (int index = 0; index < tasks.Length; index++)
            {
                try
                {
                    tasks[index].Wait();
                }
                catch (Exception ex)
                {
                    cleanupFailures.Add(new InvalidOperationException(
                        "Vegetation background preparation join failed during disposal.",
                        ex));
                }
            }

            RuntimeAssetResidencyKey[] preparedKeys;
            lock (m_Gate)
            {
                preparedKeys = m_Prepared.Keys.Order().ToArray();
            }

            for (int index = 0; index < preparedKeys.Length; index++)
            {
                try
                {
                    ReleasePreparedResource(preparedKeys[index]);
                }
                catch (Exception ex)
                {
                    cleanupFailures.Add(new InvalidOperationException(
                        $"Vegetation resident publication '{preparedKeys[index]}' failed to " +
                        "release during disposal.",
                        ex));
                }
            }

            if (cleanupFailures.Count != 0)
            {
                throw new AggregateException(
                    "Vegetation prepared-resource disposal failed.",
                    cleanupFailures);
            }
        }
    }

    internal void WaitForWorkerIdle()
    {
        BackgroundTask<WorkerPreparedResource>[] tasks;
        lock (m_Gate)
        {
            tasks = m_Outstanding
                .Concat(m_Pending.Values.Select(pending => pending.Task))
                .Where(task => !task.IsCompleted)
                .Distinct()
                .ToArray();
        }

        for (int index = 0; index < tasks.Length; index++)
        {
            tasks[index].Wait();
        }
    }

    internal bool HasPendingPreparations
    {
        get
        {
            lock (m_Gate)
            {
                return m_Pending.Count != 0;
            }
        }
    }

    private WorkerPreparedResource PrepareOnWorker(
        RuntimeAssetResidencyKey key,
        RuntimeAssetPreparationClaim claim,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        WorkerPreparedResource prepared = key.AssetType switch
        {
            VegetationAssetTypes.Cluster => LoadCluster(claim),
            VegetationAssetTypes.InstancePage => LoadPage(claim),
            VegetationAssetTypes.Species => LoadSpecies(claim),
            VegetationAssetTypes.Biome => LoadBiome(claim),
            _ => throw new InvalidOperationException(
                $"Unsupported vegetation asset type '{key.AssetType}'.")
        };
        cancellationToken.ThrowIfCancellationRequested();
        if (!IsPreparationClosureCurrent(claim, prepared.RequiredClaims))
        {
            throw new StaleVegetationPreparationException(
                $"Vegetation residency ownership changed while '{key}' was being prepared.");
        }

        return prepared;
    }

    private WorkerPreparedResource LoadCluster(in RuntimeAssetPreparationClaim claim)
    {
        if (!VegetationResidentAssetLoader.TryLoadCluster(
                m_AssetDatabase,
                claim,
                out CookedVegetationCluster cluster,
                out string diagnostic))
        {
            ThrowIfPreparationStale(claim, Array.Empty<RuntimeAssetPreparationClaim>(), diagnostic);
            throw new InvalidDataException(diagnostic);
        }

        RuntimeAssetResidencyKey[] requiredKeys =
            VegetationResidentAssetLoader.GetRequiredResidencyKeys(cluster);
        BindPreparationDependenciesOrThrow(
            claim,
            requiredKeys,
            "Vegetation cluster biome/species/page closure");
        RuntimeAssetPreparationClaim[] requiredClaims = CaptureRequiredClaimsOrThrow(
            claim,
            requiredKeys,
            "vegetation cluster biome/species/page closure");
        if (!VegetationResidentAssetLoader.TryValidateClusterClosure(
                m_AssetDatabase,
                cluster,
                requiredClaims,
                out diagnostic))
        {
            ThrowIfPreparationStale(claim, requiredClaims, diagnostic);
            throw new InvalidDataException(diagnostic);
        }

        return WorkerPreparedResource.ForCluster(
            m_RuntimeData.PrepareCluster(cluster),
            requiredKeys,
            requiredClaims);
    }

    private WorkerPreparedResource LoadPage(in RuntimeAssetPreparationClaim claim)
    {
        if (!VegetationResidentAssetLoader.TryLoadPage(
                m_AssetDatabase,
                claim,
                out CookedVegetationInstancePage page,
                out VegetationInstancePagePayloadIdentity payloadIdentity,
                out string diagnostic))
        {
            ThrowIfPreparationStale(claim, Array.Empty<RuntimeAssetPreparationClaim>(), diagnostic);
            throw new InvalidDataException(diagnostic);
        }

        RuntimeAssetResidencyKey[] requiredKeys =
            VegetationResidentAssetLoader.GetRequiredResidencyKeys(page);
        BindPreparationDependenciesOrThrow(
            claim,
            requiredKeys,
            "Vegetation instance-page parent/species closure");
        RuntimeAssetPreparationClaim[] requiredClaims = CaptureRequiredClaimsOrThrow(
            claim,
            requiredKeys,
            "vegetation instance-page parent/species closure");
        if (!VegetationResidentAssetLoader.TryValidatePageClosure(
                m_AssetDatabase,
                page,
                payloadIdentity,
                requiredClaims,
                out diagnostic))
        {
            ThrowIfPreparationStale(claim, requiredClaims, diagnostic);
            throw new InvalidDataException(diagnostic);
        }

        return WorkerPreparedResource.ForPage(
            m_RuntimeData.PreparePage(page, payloadIdentity),
            requiredKeys,
            requiredClaims);
    }

    private WorkerPreparedResource LoadSpecies(in RuntimeAssetPreparationClaim claim)
    {
        if (!VegetationResidentAssetLoader.TryLoadSpecies(
                m_AssetDatabase,
                claim,
                out CookedVegetationSpecies species,
                out string diagnostic))
        {
            ThrowIfPreparationStale(claim, Array.Empty<RuntimeAssetPreparationClaim>(), diagnostic);
            throw new InvalidDataException(diagnostic);
        }

        RuntimeAssetResidencyKey[] requiredKeys =
            VegetationResidentAssetLoader.GetRequiredResidencyKeys(species);
        BindPreparationDependenciesOrThrow(
            claim,
            requiredKeys,
            "Vegetation species mesh/material closure");
        RuntimeAssetPreparationClaim[] requiredClaims = CaptureRequiredClaimsOrThrow(
            claim,
            requiredKeys,
            "vegetation species mesh/material closure");
        if (!VegetationResidentAssetLoader.TryValidateSpeciesClosure(
                species,
                requiredClaims,
                out diagnostic))
        {
            ThrowIfPreparationStale(claim, requiredClaims, diagnostic);
            throw new InvalidDataException(diagnostic);
        }

        return WorkerPreparedResource.ForDependencies(requiredKeys, requiredClaims);
    }

    private WorkerPreparedResource LoadBiome(in RuntimeAssetPreparationClaim claim)
    {
        if (!VegetationResidentAssetLoader.TryLoadBiome(
                m_AssetDatabase,
                claim,
                out CookedVegetationBiome biome,
                out string diagnostic))
        {
            ThrowIfPreparationStale(claim, Array.Empty<RuntimeAssetPreparationClaim>(), diagnostic);
            throw new InvalidDataException(diagnostic);
        }

        RuntimeAssetResidencyKey[] requiredKeys =
            VegetationResidentAssetLoader.GetRequiredResidencyKeys(biome);
        BindPreparationDependenciesOrThrow(
            claim,
            requiredKeys,
            "Vegetation biome species closure");
        RuntimeAssetPreparationClaim[] requiredClaims = CaptureRequiredClaimsOrThrow(
            claim,
            requiredKeys,
            "vegetation biome species closure");
        if (!VegetationResidentAssetLoader.TryValidateBiomeClosure(
                m_AssetDatabase,
                biome,
                requiredClaims,
                out diagnostic))
        {
            ThrowIfPreparationStale(claim, requiredClaims, diagnostic);
            throw new InvalidDataException(diagnostic);
        }

        return WorkerPreparedResource.ForDependencies(requiredKeys, requiredClaims);
    }

    private RuntimePreparedAssetResult ClaimPreparationLocked(
        RuntimeAssetResidencyKey key,
        PendingPreparation pending)
    {
        BackgroundTask<WorkerPreparedResource> task = pending.Task;
        if (!task.IsCompleted)
        {
            return RuntimePreparedAssetResult.Waiting(
                $"Vegetation asset '{key}' is being prepared on a background worker.");
        }

        m_Pending.Remove(key);
        m_Outstanding.Remove(task);
        if (!task.TryGetResult(out WorkerPreparedResource prepared))
        {
            if (task.Failure is StaleVegetationPreparationException stale)
            {
                return RuntimePreparedAssetResult.Waiting(stale.Message);
            }

            if (!IsResidencyClaimCurrent(key, pending.Claim))
            {
                return RuntimePreparedAssetResult.Waiting(
                    $"Vegetation asset '{key}' failed after its residency owner plan changed.");
            }

            string diagnostic = task.Status == BackgroundTaskStatus.Cancelled
                ? "Background preparation was cancelled before publication."
                : task.Failure?.Message ?? "Background preparation failed without a diagnostic.";
            return RuntimePreparedAssetResult.Failed(
                $"Vegetation prepared-resource setup failed for '{key}': {diagnostic}");
        }

        VegetationResidentResourceHandle handle = default;
        bool preparedEntryPublished = false;
        try
        {
            if (!m_ResidencyService.TryCommitPreparedPublication(
                    pending.Claim,
                    prepared.RequiredClaims,
                    prepared.RequiredKeys,
                    estimatedGpuBytes: 0,
                    () =>
                    {
                        handle = prepared.Publish(m_RuntimeData);
                        m_Prepared.Add(key, new PreparedResource(handle));
                        preparedEntryPublished = true;
                        Volatile.Write(ref m_PreparedResourceCount, m_Prepared.Count);
                    },
                    out string diagnostic))
            {
                return IsPreparationClosureCurrent(pending.Claim, prepared.RequiredClaims)
                    ? RuntimePreparedAssetResult.Failed(
                        $"Vegetation exact dependency commit failed before publication: " +
                        diagnostic)
                    : RuntimePreparedAssetResult.Waiting(
                        $"Vegetation asset '{key}' dependency closure became stale before " +
                        "publication.");
            }

            return RuntimePreparedAssetResult.Ready(estimatedGpuBytes: 0);
        }
        catch (Exception publicationFailure)
        {
            Exception? cleanupFailure = null;
            bool removePreparedEntry = !handle.IsValid;
            if (handle.IsValid)
            {
                try
                {
                    if (!m_RuntimeData.Remove(handle))
                    {
                        throw new InvalidOperationException(
                            $"Vegetation publication '{key}' was not owned by its rollback handle.");
                    }

                    removePreparedEntry = true;
                }
                catch (Exception ex)
                {
                    cleanupFailure = ex;
                }
            }

            if (preparedEntryPublished && removePreparedEntry)
            {
                m_Prepared.Remove(key);
                Volatile.Write(ref m_PreparedResourceCount, m_Prepared.Count);
            }

            if (cleanupFailure != null)
            {
                throw new AggregateException(
                    $"Vegetation publication and rollback both failed for '{key}'.",
                    publicationFailure,
                    cleanupFailure);
            }

            if (publicationFailure is RuntimePreparedPublicationInvalidatedException)
            {
                return RuntimePreparedAssetResult.Waiting(
                    $"Vegetation asset '{key}' dependency closure changed during publication.");
            }

            throw;
        }
    }

    private RuntimePreparedAssetResult StartPreparationLocked(
        RuntimeAssetResidencyKey key,
        RuntimeAssetPreparationClaim claim)
    {
        BackgroundTask<WorkerPreparedResource> task = m_BackgroundTasks.Schedule(
            $"Vegetation.Prepare.{key.AssetType}.{key.Guid:D}",
            cancellationToken => PrepareOnWorker(key, claim, cancellationToken));
        m_Pending.Add(key, new PendingPreparation(task, claim));
        m_Outstanding.Add(task);
        return RuntimePreparedAssetResult.Waiting(
            $"Vegetation asset '{key}' is being prepared on a background worker.");
    }

    private bool TryCaptureResidencyClaim(
        RuntimeAssetResidencyKey key,
        out RuntimeAssetPreparationClaim claim,
        out string diagnostic)
    {
        if (m_ResidencyService.TryGetPreparationClaim(key, out claim))
        {
            diagnostic = string.Empty;
            return true;
        }

        claim = default;
        diagnostic =
            $"Vegetation asset '{key}' has no active CPU residency owner for preparation.";
        return false;
    }

    private bool IsResidencyClaimCurrent(
        RuntimeAssetResidencyKey key,
        RuntimeAssetPreparationClaim claim) =>
        TryCaptureResidencyClaim(key, out RuntimeAssetPreparationClaim current, out _) &&
        current == claim;

    private bool IsPreparationClosureCurrent(
        in RuntimeAssetPreparationClaim claim,
        IReadOnlyList<RuntimeAssetPreparationClaim> requiredClaims)
    {
        if (!IsResidencyClaimCurrent(claim.Key, claim))
        {
            return false;
        }

        for (int index = 0; index < requiredClaims.Count; index++)
        {
            RuntimeAssetPreparationClaim requiredClaim = requiredClaims[index];
            if (!IsResidencyClaimCurrent(requiredClaim.Key, requiredClaim))
            {
                return false;
            }
        }

        return true;
    }

    private void BindPreparationDependenciesOrThrow(
        in RuntimeAssetPreparationClaim claim,
        IReadOnlyList<RuntimeAssetResidencyKey> requiredKeys,
        string relationship)
    {
        if (m_ResidencyService.TryBindPreparationDependencies(
                claim,
                requiredKeys,
                out string diagnostic))
        {
            return;
        }

        if (!IsResidencyClaimCurrent(claim.Key, claim))
        {
            throw new StaleVegetationPreparationException(
                $"{relationship} binding became stale: {diagnostic}");
        }

        throw new InvalidDataException($"{relationship} binding failed: {diagnostic}");
    }

    private RuntimeAssetPreparationClaim[] CaptureRequiredClaimsOrThrow(
        in RuntimeAssetPreparationClaim claim,
        IReadOnlyList<RuntimeAssetResidencyKey> requiredKeys,
        string relationship)
    {
        var requiredClaims = new RuntimeAssetPreparationClaim[requiredKeys.Count];
        for (int index = 0; index < requiredKeys.Count; index++)
        {
            RuntimeAssetResidencyKey requiredKey = requiredKeys[index];
            if (!TryCaptureResidencyClaim(
                    requiredKey,
                    out requiredClaims[index],
                    out string diagnostic))
            {
                throw new StaleVegetationPreparationException(
                    $"{relationship} lost exact residency-held dependency '{requiredKey}': " +
                    diagnostic);
            }
        }

        if (!IsResidencyClaimCurrent(claim.Key, claim))
        {
            throw new StaleVegetationPreparationException(
                $"{relationship} root claim became stale while dependency claims were captured.");
        }

        return requiredClaims;
    }

    private void ThrowIfPreparationStale(
        in RuntimeAssetPreparationClaim claim,
        IReadOnlyList<RuntimeAssetPreparationClaim> requiredClaims,
        string diagnostic)
    {
        if (!IsPreparationClosureCurrent(claim, requiredClaims))
        {
            throw new StaleVegetationPreparationException(
                $"Vegetation exact held-handle closure became stale: {diagnostic}");
        }
    }

    private void ReleasePreparedResource(RuntimeAssetResidencyKey key)
    {
        PreparedResource? prepared;
        lock (m_Gate)
        {
            while (true)
            {
                if (!m_Prepared.TryGetValue(key, out prepared))
                {
                    return;
                }

                if (!prepared.ReleaseInProgress)
                {
                    prepared.ReleaseInProgress = true;
                    break;
                }

                Monitor.Wait(m_Gate);
            }
        }

        try
        {
            if (prepared.ResidentHandle.IsValid)
            {
                if (!m_RuntimeData.Remove(prepared.ResidentHandle))
                {
                    throw new InvalidOperationException(
                        $"Vegetation publication '{key}' was not owned by its resident handle.");
                }
            }
        }
        catch
        {
            lock (m_Gate)
            {
                prepared.ReleaseInProgress = false;
                Monitor.PulseAll(m_Gate);
            }

            throw;
        }

        lock (m_Gate)
        {
            if (m_Prepared.TryGetValue(key, out PreparedResource? current) &&
                ReferenceEquals(current, prepared))
            {
                m_Prepared.Remove(key);
                Volatile.Write(ref m_PreparedResourceCount, m_Prepared.Count);
            }

            prepared.ReleaseInProgress = false;
            Monitor.PulseAll(m_Gate);
        }
    }

    private void PruneCompletedOrphansLocked()
    {
        m_Outstanding.RemoveWhere(task => task.IsCompleted);
    }

    private static string GetExpectedVariant(string assetType) => assetType switch
    {
        VegetationAssetTypes.Cluster => VegetationClusterAssetCooker.RuntimeVariant,
        VegetationAssetTypes.InstancePage => VegetationInstancePageAssetCooker.RuntimeVariant,
        VegetationAssetTypes.Species => VegetationSpeciesAssetCooker.RuntimeVariant,
        VegetationAssetTypes.Biome => VegetationBiomeAssetCooker.RuntimeVariant,
        _ => string.Empty
    };

    private sealed class PreparedResource
    {
        public PreparedResource(VegetationResidentResourceHandle residentHandle)
        {
            ResidentHandle = residentHandle;
        }

        public VegetationResidentResourceHandle ResidentHandle { get; }

        public bool ReleaseInProgress { get; set; }
    }

    private sealed class PendingPreparation
    {
        public PendingPreparation(
            BackgroundTask<WorkerPreparedResource> task,
            RuntimeAssetPreparationClaim claim)
        {
            Task = task;
            Claim = claim;
        }

        public BackgroundTask<WorkerPreparedResource> Task { get; }

        public RuntimeAssetPreparationClaim Claim { get; }
    }

    private sealed class StaleVegetationPreparationException : InvalidOperationException
    {
        public StaleVegetationPreparationException(string message)
            : base(message)
        {
        }
    }

    private readonly record struct WorkerPreparedResource(
        VegetationPreparedClusterData? Cluster,
        VegetationPreparedPageData? Page,
        RuntimeAssetResidencyKey[] RequiredKeys,
        RuntimeAssetPreparationClaim[] RequiredClaims)
    {
        public static WorkerPreparedResource ForCluster(
            VegetationPreparedClusterData cluster,
            RuntimeAssetResidencyKey[] requiredKeys,
            RuntimeAssetPreparationClaim[] requiredClaims) =>
            new(cluster, null, requiredKeys, requiredClaims);

        public static WorkerPreparedResource ForPage(
            VegetationPreparedPageData page,
            RuntimeAssetResidencyKey[] requiredKeys,
            RuntimeAssetPreparationClaim[] requiredClaims) =>
            new(null, page, requiredKeys, requiredClaims);

        public static WorkerPreparedResource ForDependencies(
            RuntimeAssetResidencyKey[] requiredKeys,
            RuntimeAssetPreparationClaim[] requiredClaims) =>
            new(null, null, requiredKeys, requiredClaims);

        public VegetationResidentResourceHandle Publish(IVegetationRuntimeDataStore store)
        {
            if (Cluster != null)
            {
                return store.PublishCluster(Cluster);
            }

            return Page != null
                ? store.PublishPage(Page)
                : default;
        }
    }
}
