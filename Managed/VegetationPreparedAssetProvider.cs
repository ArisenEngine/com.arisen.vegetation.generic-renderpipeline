using ArisenEngine.Core.Assets;
using ArisenEngine.Core.RHI;
using ArisenEngine.Resources.Serialization;
using ArisenEngine.Threading;
using ArisenEngine.Vegetation.Assets;

namespace ArisenEngine.Vegetation.GenericRenderPipeline;

internal enum VegetationPreparedPublicationStage
{
    PreparedEntryAdded = 0,
    ClusterMappingAdded = 1,
    GpuBytesCharged = 2,
    PreparedCountUpdated = 3
}

internal sealed class VegetationPreparedAssetProvider :
    IRuntimePreparedAssetProvider,
    IVegetationPreparedClusterSource,
    IDisposable
{
    public const string Id =
        "com.arisen.vegetation.generic-renderpipeline.prepared-assets";

    private readonly object m_DisposeGate = new();
    private readonly IAssetDatabase m_AssetDatabase;
    private readonly IBackgroundTaskScheduler m_BackgroundTasks;
    private readonly IVegetationRuntimeDataStore m_RuntimeData;
    private readonly IRuntimeAssetResidencyService m_ResidencyService;
    private readonly IVegetationClusterGpuResourceFactory? m_GpuResourceFactory;
    private readonly Action<RuntimeAssetResidencyKey, VegetationPreparedPublicationStage>?
        m_PublicationObserver;
    private readonly VegetationPreparedAssetProviderLifecycleState m_LifecycleState = new();
    private readonly Dictionary<RuntimeAssetResidencyKey, PreparedResource> m_Prepared = new();
    private readonly Dictionary<Guid, PreparedResource> m_ClustersByGuid = new();
    private readonly Dictionary<RuntimeAssetResidencyKey, PendingPreparation> m_Pending = new();
    private readonly HashSet<BackgroundTask<WorkerPreparedResource>> m_Outstanding = new();
    private int m_PreparedResourceCount;
    private int m_ResidencyRegistrationOwned;
    private long m_EstimatedGpuBytes;
    private PreparedClusterPublication[] m_ClusterPublications =
        Array.Empty<PreparedClusterPublication>();
    private bool m_Disposed;

    public VegetationPreparedAssetProvider(
        IAssetDatabase assetDatabase,
        IBackgroundTaskScheduler backgroundTasks,
        IVegetationRuntimeDataStore runtimeData,
        IRuntimeAssetResidencyService residencyService,
        IVegetationClusterGpuResourceFactory? gpuResourceFactory = null,
        Action<RuntimeAssetResidencyKey, VegetationPreparedPublicationStage>?
            publicationObserver = null)
    {
        m_AssetDatabase = assetDatabase ?? throw new ArgumentNullException(nameof(assetDatabase));
        m_BackgroundTasks = backgroundTasks
            ?? throw new ArgumentNullException(nameof(backgroundTasks));
        m_RuntimeData = runtimeData ?? throw new ArgumentNullException(nameof(runtimeData));
        m_ResidencyService = residencyService
            ?? throw new ArgumentNullException(nameof(residencyService));
        m_GpuResourceFactory = gpuResourceFactory;
        m_PublicationObserver = publicationObserver;
        PublishMetricsSnapshot();
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
            lock (m_LifecycleState.Gate)
            {
                if (m_Disposed)
                {
                    return RuntimePreparedAssetResult.Failed(
                        "Vegetation prepared assets have been disposed.");
                }

                if (m_LifecycleState.IsReleasePendingLocked(key))
                {
                    return RuntimePreparedAssetResult.Waiting(
                        $"Vegetation is retiring the previous exact publication for '{key}'.");
                }

                if (m_Prepared.TryGetValue(key, out PreparedResource? existing))
                {
                    return RuntimePreparedAssetResult.Ready(existing.EstimatedGpuBytes);
                }
            }

            if (!TryCaptureResidencyClaim(
                    key,
                    out RuntimeAssetPreparationClaim claim,
                    out string claimDiagnostic))
            {
                return RuntimePreparedAssetResult.Waiting(claimDiagnostic);
            }

            lock (m_LifecycleState.Gate)
            {
                if (m_Disposed)
                {
                    return RuntimePreparedAssetResult.Failed(
                        "Vegetation prepared assets have been disposed.");
                }

                if (m_LifecycleState.IsReleasePendingLocked(key))
                {
                    return RuntimePreparedAssetResult.Waiting(
                        $"Vegetation is retiring the previous exact publication for '{key}'.");
                }

                PruneCompletedOrphansLocked();
                if (m_Prepared.TryGetValue(key, out PreparedResource? existing))
                {
                    return RuntimePreparedAssetResult.Ready(existing.EstimatedGpuBytes);
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
        finally
        {
            PublishMetricsSnapshot();
        }
    }

    public void Release(RuntimeAssetResidencyKey key)
    {
        BackgroundTask<WorkerPreparedResource>? task = null;
        lock (m_LifecycleState.Gate)
        {
            if (m_LifecycleState.RequestReleaseLocked(key))
            {
                PublishClusterPublicationsLocked();
            }
            if (m_Pending.Remove(key, out PendingPreparation? pending))
            {
                task = pending.Task;
            }

            PruneCompletedOrphansLocked();
        }

        task?.Cancel();
        ReleasePreparedResource(key);
    }

    public RuntimePreparedAssetProviderMetrics GetMetrics() =>
        m_LifecycleState.ReadMetrics();

    internal void UpdateFrameContext(RHIDevice device, ulong deviceGeneration)
    {
        m_GpuResourceFactory?.UpdateFrameContext(device, deviceGeneration);
        PublishMetricsSnapshot();
    }

    internal bool InvalidateStaleDependencies()
    {
        bool staleDependencies = false;
        PreparedClusterPublication[] publications =
            Volatile.Read(ref m_ClusterPublications);
        for (int index = 0; index < publications.Length; index++)
        {
            if (!publications[index].Resource.DependenciesCurrent)
            {
                staleDependencies = true;
                break;
            }
        }

        if (!staleDependencies)
        {
            return false;
        }

        if (Volatile.Read(ref m_ResidencyRegistrationOwned) == 0)
        {
            return true;
        }

        if (!m_ResidencyService.IsPreparedProviderRegistered(this))
        {
            SetResidencyRegistrationOwned(false);
            return true;
        }

        m_ResidencyService.InvalidatePreparedProvider(
            ProviderId,
            "Vegetation prepared resources depend on stale Generic RP publications.");
        return true;
    }

    public bool TryGetCluster(
        Guid clusterGuid,
        ulong generation,
        out VegetationPreparedClusterView cluster)
    {
        if (generation != 0)
        {
            PreparedClusterPublication[] publications =
                Volatile.Read(ref m_ClusterPublications);
            int low = 0;
            int high = publications.Length - 1;
            while (low <= high)
            {
                int middle = low + ((high - low) >> 1);
                ref readonly PreparedClusterPublication publication =
                    ref publications[middle];
                int comparison = publication.ClusterGuid.CompareTo(clusterGuid);
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

                if (publication.ResidentGeneration == generation)
                {
                    cluster = publication.Resource.CreateView(generation);
                    return cluster.IsValid;
                }

                break;
            }
        }

        cluster = default;
        return false;
    }

    internal void UpdateSubmittedTicket(ulong submittedTicket)
    {
        m_GpuResourceFactory?.UpdateSubmittedTicket(submittedTicket);
        PublishMetricsSnapshot();
    }

    internal void SetResidencyRegistrationOwned(bool owned)
    {
        Volatile.Write(ref m_ResidencyRegistrationOwned, owned ? 1 : 0);
    }

    internal void ReleaseAllDeviceResources()
    {
        if (Volatile.Read(ref m_ResidencyRegistrationOwned) != 0)
        {
            if (m_ResidencyService.IsPreparedProviderRegistered(this))
            {
                m_ResidencyService.InvalidatePreparedProvider(
                    ProviderId,
                    "Vegetation prepared resources are waiting for the active RHI device.");
            }
            else
            {
                SetResidencyRegistrationOwned(false);
            }
        }

        RuntimeAssetResidencyKey[] preparedKeys;
        lock (m_LifecycleState.Gate)
        {
            preparedKeys = m_Prepared.Keys.Order().ToArray();
        }
        for (int index = 0; index < preparedKeys.Length; index++)
        {
            Release(preparedKeys[index]);
        }

        m_GpuResourceFactory?.ReleaseAllDeviceResources();
        PublishMetricsSnapshot();
    }

    public void Dispose()
    {
        lock (m_DisposeGate)
        {
            BackgroundTask<WorkerPreparedResource>[] tasks;
            lock (m_LifecycleState.Gate)
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
            lock (m_LifecycleState.Gate)
            {
                preparedKeys = m_Prepared.Keys.Order().ToArray();
            }

            for (int index = 0; index < preparedKeys.Length; index++)
            {
                try
                {
                    Release(preparedKeys[index]);
                }
                catch (Exception ex)
                {
                    cleanupFailures.Add(new InvalidOperationException(
                        $"Vegetation resident publication '{preparedKeys[index]}' failed to " +
                        "release during disposal.",
                        ex));
                }
            }

            bool hasPreparedResources;
            lock (m_LifecycleState.Gate)
            {
                hasPreparedResources = m_Prepared.Count != 0;
            }

            if (!hasPreparedResources)
            {
                try
                {
                    m_GpuResourceFactory?.ReleaseAllDeviceResources();
                }
                catch (Exception ex)
                {
                    cleanupFailures.Add(new InvalidOperationException(
                        "Vegetation GPU factory failed to release during disposal.",
                        ex));
                }
            }

            PublishMetricsSnapshot();

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
        lock (m_LifecycleState.Gate)
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
            lock (m_LifecycleState.Gate)
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

        var species = new CookedVegetationSpecies[cluster.Species.Count];
        for (int index = 0; index < species.Length; index++)
        {
            CookedVegetationSpeciesReference reference = cluster.Species[index];
            RuntimeAssetPreparationClaim speciesClaim = FindRequiredClaim(
                requiredClaims,
                reference.Guid,
                reference.PackageId,
                VegetationAssetTypes.Species,
                VegetationSpeciesAssetCooker.RuntimeVariant);
            if (!VegetationResidentAssetLoader.TryLoadSpecies(
                    m_AssetDatabase,
                    speciesClaim,
                    out species[index],
                    out diagnostic))
            {
                ThrowIfPreparationStale(claim, requiredClaims, diagnostic);
                throw new InvalidDataException(diagnostic);
            }
        }

        var pages = new CookedVegetationInstancePage[cluster.Pages.Count];
        for (int index = 0; index < pages.Length; index++)
        {
            CookedVegetationInstancePageReference reference = cluster.Pages[index];
            RuntimeAssetPreparationClaim pageClaim = FindRequiredClaim(
                requiredClaims,
                reference.Guid,
                reference.PackageId,
                VegetationAssetTypes.InstancePage,
                VegetationInstancePageAssetCooker.RuntimeVariant);
            if (!VegetationResidentAssetLoader.TryLoadPage(
                    m_AssetDatabase,
                    pageClaim,
                    out pages[index],
                    out _,
                    out diagnostic))
            {
                ThrowIfPreparationStale(claim, requiredClaims, diagnostic);
                throw new InvalidDataException(diagnostic);
            }
        }

        return WorkerPreparedResource.ForCluster(
            m_RuntimeData.PrepareCluster(cluster),
            cluster,
            species,
            pages,
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

        if (!task.TryGetResult(out WorkerPreparedResource prepared))
        {
            m_Pending.Remove(key);
            m_Outstanding.Remove(task);
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

        IVegetationClusterGpuResource? gpuResource = null;
        long estimatedGpuBytes = 0;
        if (prepared.ClusterSource != null && m_GpuResourceFactory != null)
        {
            VegetationGpuResourceBuildResult gpuResult = m_GpuResourceFactory.TryCreate(
                prepared.ClusterSource,
                prepared.Species!,
                prepared.Pages!);
            if (gpuResult.Status == VegetationGpuResourceBuildStatus.Waiting)
            {
                return RuntimePreparedAssetResult.Waiting(gpuResult.Diagnostic);
            }
            if (gpuResult.Status != VegetationGpuResourceBuildStatus.Ready ||
                gpuResult.Resource == null)
            {
                m_Pending.Remove(key);
                m_Outstanding.Remove(task);
                return RuntimePreparedAssetResult.Failed(
                    $"Vegetation cluster GPU setup failed for '{key}': " +
                    gpuResult.Diagnostic);
            }

            gpuResource = gpuResult.Resource;
            estimatedGpuBytes = gpuResource.EstimatedGpuBytes;
        }

        m_Pending.Remove(key);
        m_Outstanding.Remove(task);

        VegetationResidentResourceHandle handle = default;
        bool preparedEntryPublished = false;
        PreparedResource? publishedResource = null;
        try
        {
            if (!m_ResidencyService.TryCommitPreparedPublication(
                    pending.Claim,
                    prepared.RequiredClaims,
                    prepared.RequiredKeys,
                    estimatedGpuBytes,
                    () =>
                    {
                        handle = prepared.Publish(m_RuntimeData);
                        publishedResource = new PreparedResource(
                            key,
                            handle,
                            gpuResource,
                            estimatedGpuBytes);
                        m_Prepared.Add(key, publishedResource);
                        preparedEntryPublished = true;
                        ObservePublication(key, VegetationPreparedPublicationStage.PreparedEntryAdded);
                        if (gpuResource != null)
                        {
                            m_ClustersByGuid.Add(key.Guid, publishedResource);
                            publishedResource.ClusterMappingPublished = true;
                            ObservePublication(
                                key,
                                VegetationPreparedPublicationStage.ClusterMappingAdded);
                            Interlocked.Add(ref m_EstimatedGpuBytes, estimatedGpuBytes);
                            publishedResource.GpuBytesCharged = true;
                            ObservePublication(
                                key,
                                VegetationPreparedPublicationStage.GpuBytesCharged);
                        }
                        Volatile.Write(ref m_PreparedResourceCount, m_Prepared.Count);
                        ObservePublication(
                            key,
                            VegetationPreparedPublicationStage.PreparedCountUpdated);
                        m_LifecycleState.PublishPhysicalMetricsLocked(
                            CapturePhysicalMetrics());
                    },
                    out string diagnostic))
            {
                ReleaseGpuResource(gpuResource);
                gpuResource = null;
                return IsPreparationClosureCurrent(pending.Claim, prepared.RequiredClaims)
                    ? RuntimePreparedAssetResult.Failed(
                        $"Vegetation exact dependency commit failed before publication: " +
                        diagnostic)
                    : RuntimePreparedAssetResult.Waiting(
                        $"Vegetation asset '{key}' dependency closure became stale before " +
                        "publication.");
            }

            PublishClusterPublicationsLocked();
            return RuntimePreparedAssetResult.Ready(estimatedGpuBytes);
        }
        catch (Exception publicationFailure)
        {
            var cleanupFailures = new List<Exception>();
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

                    if (publishedResource != null)
                    {
                        publishedResource.RuntimeDataPublished = false;
                    }
                    removePreparedEntry = true;
                }
                catch (Exception ex)
                {
                    cleanupFailures.Add(ex);
                }
            }

            if (preparedEntryPublished &&
                publishedResource != null &&
                removePreparedEntry)
            {
                try
                {
                    ReleaseGpuResource(publishedResource.GpuResource);
                    publishedResource.GpuReleaseComplete = true;
                    RollbackPreparedPublicationLocked(key, publishedResource);
                    gpuResource = null;
                }
                catch (Exception ex)
                {
                    cleanupFailures.Add(ex);
                }
            }
            else if (!preparedEntryPublished && removePreparedEntry)
            {
                try
                {
                    ReleaseGpuResource(gpuResource);
                    gpuResource = null;
                }
                catch (Exception ex)
                {
                    cleanupFailures.Add(ex);
                }
            }

            Volatile.Write(ref m_PreparedResourceCount, m_Prepared.Count);
            if (cleanupFailures.Count != 0)
            {
                cleanupFailures.Insert(0, publicationFailure);
                throw new AggregateException(
                    $"Vegetation publication and rollback both failed for '{key}'.",
                    cleanupFailures);
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

    private static RuntimeAssetPreparationClaim FindRequiredClaim(
        IReadOnlyList<RuntimeAssetPreparationClaim> claims,
        Guid guid,
        string packageId,
        string assetType,
        string variant)
    {
        for (int index = 0; index < claims.Count; index++)
        {
            RuntimeAssetPreparationClaim claim = claims[index];
            if (claim.Key.Guid == guid &&
                string.Equals(claim.Key.PackageId, packageId, StringComparison.Ordinal) &&
                string.Equals(claim.Key.AssetType, assetType, StringComparison.Ordinal) &&
                string.Equals(claim.Key.Variant, variant, StringComparison.Ordinal))
            {
                return claim;
            }
        }

        throw new InvalidDataException(
            $"Vegetation exact closure is missing '{packageId}:{assetType}:{guid:D}:{variant}'.");
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
        lock (m_LifecycleState.Gate)
        {
            while (true)
            {
                if (!m_Prepared.TryGetValue(key, out prepared))
                {
                    if (m_LifecycleState.IsReleasePendingLocked(key))
                    {
                        m_LifecycleState.CompleteReleaseLocked(
                            key,
                            CapturePhysicalMetrics());
                    }

                    return;
                }

                if (!prepared.ReleaseInProgress)
                {
                    ValidatePreparedPublicationMappingsLocked(key, prepared);
                    prepared.ReleaseInProgress = true;
                    break;
                }

                Monitor.Wait(m_LifecycleState.Gate);
            }
        }

        try
        {
            if (prepared.RuntimeDataPublished)
            {
                if (!m_RuntimeData.Remove(prepared.ResidentHandle))
                {
                    throw new InvalidOperationException(
                        $"Vegetation publication '{key}' was not owned by its resident handle.");
                }

                prepared.RuntimeDataPublished = false;
            }

            if (!prepared.GpuReleaseComplete)
            {
                ReleaseGpuResource(prepared.GpuResource);
                prepared.GpuReleaseComplete = true;
            }
        }
        catch
        {
            lock (m_LifecycleState.Gate)
            {
                prepared.ReleaseInProgress = false;
                m_LifecycleState.PublishPhysicalMetricsLocked(
                    CapturePhysicalMetrics());
                Monitor.PulseAll(m_LifecycleState.Gate);
            }

            throw;
        }

        lock (m_LifecycleState.Gate)
        {
            if (!m_Prepared.TryGetValue(key, out PreparedResource? current) ||
                !ReferenceEquals(current, prepared))
            {
                throw new InvalidOperationException(
                    $"Vegetation publication '{key}' changed during release.");
            }

            m_Prepared.Remove(key);
            if (prepared.ClusterMappingPublished)
            {
                m_ClustersByGuid.Remove(key.Guid);
                prepared.ClusterMappingPublished = false;
            }
            if (prepared.GpuBytesCharged)
            {
                Interlocked.Add(ref m_EstimatedGpuBytes, -prepared.EstimatedGpuBytes);
                prepared.GpuBytesCharged = false;
            }
            Volatile.Write(ref m_PreparedResourceCount, m_Prepared.Count);
            PublishClusterPublicationsLocked();

            prepared.ReleaseInProgress = false;
            m_LifecycleState.CompleteReleaseLocked(
                key,
                CapturePhysicalMetrics());
            Monitor.PulseAll(m_LifecycleState.Gate);
        }

    }

    private void RollbackPreparedPublicationLocked(
        RuntimeAssetResidencyKey key,
        PreparedResource resource)
    {
        if (!m_Prepared.TryGetValue(key, out PreparedResource? current) ||
            !ReferenceEquals(current, resource))
        {
            throw new InvalidOperationException(
                $"Vegetation publication '{key}' changed before rollback.");
        }

        ValidatePreparedPublicationMappingsLocked(key, resource);
        if (resource.RuntimeDataPublished || !resource.GpuReleaseComplete)
        {
            throw new InvalidOperationException(
                $"Vegetation publication '{key}' rollback has not released external ownership.");
        }
        if (resource.ClusterMappingPublished)
        {
            m_ClustersByGuid.Remove(key.Guid);
            resource.ClusterMappingPublished = false;
        }
        if (resource.GpuBytesCharged)
        {
            Interlocked.Add(ref m_EstimatedGpuBytes, -resource.EstimatedGpuBytes);
            resource.GpuBytesCharged = false;
        }

        m_Prepared.Remove(key);
    }

    private void ValidatePreparedPublicationMappingsLocked(
        RuntimeAssetResidencyKey key,
        PreparedResource resource)
    {
        if (resource.ClusterMappingPublished &&
            (!m_ClustersByGuid.TryGetValue(key.Guid, out PreparedResource? cluster) ||
             !ReferenceEquals(cluster, resource)))
        {
            throw new InvalidOperationException(
                $"Vegetation cluster mapping '{key.Guid:D}' is owned by another publication.");
        }
    }

    private void PublishClusterPublicationsLocked()
    {
        if (!Monitor.IsEntered(m_LifecycleState.Gate))
        {
            throw new InvalidOperationException(
                "Vegetation cluster publications require the lifecycle gate.");
        }

        var publications = new PreparedClusterPublication[m_ClustersByGuid.Count];
        int count = 0;
        foreach (PreparedResource resource in m_ClustersByGuid.Values)
        {
            if (resource.GpuResource == null ||
                m_LifecycleState.IsReleasePendingLocked(resource.Key))
            {
                continue;
            }

            publications[count++] = new PreparedClusterPublication(
                resource.Key.Guid,
                resource.ResidentHandle.Generation,
                resource.GpuResource);
        }

        if (count != publications.Length)
        {
            Array.Resize(ref publications, count);
        }
        Array.Sort(publications);
        Volatile.Write(ref m_ClusterPublications, publications);
    }

    private void ObservePublication(
        RuntimeAssetResidencyKey key,
        VegetationPreparedPublicationStage stage)
    {
        m_PublicationObserver?.Invoke(key, stage);
    }

    private void ReleaseGpuResource(IVegetationClusterGpuResource? resource)
    {
        if (resource == null)
        {
            return;
        }

        if (m_GpuResourceFactory != null)
        {
            m_GpuResourceFactory.RequestRelease(resource);
        }
        else
        {
            resource.Dispose();
        }
    }

    private void PruneCompletedOrphansLocked()
    {
        m_Outstanding.RemoveWhere(task => task.IsCompleted);
    }

    private RuntimePreparedAssetProviderMetrics CapturePhysicalMetrics() => new(
        Volatile.Read(ref m_PreparedResourceCount),
        Interlocked.Read(ref m_EstimatedGpuBytes),
        m_GpuResourceFactory?.PendingDisposalCount ?? 0);

    private void PublishMetricsSnapshot()
    {
        lock (m_LifecycleState.Gate)
        {
            m_LifecycleState.PublishPhysicalMetricsLocked(CapturePhysicalMetrics());
        }
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
        public PreparedResource(
            RuntimeAssetResidencyKey key,
            VegetationResidentResourceHandle residentHandle,
            IVegetationClusterGpuResource? gpuResource,
            long estimatedGpuBytes)
        {
            Key = key;
            ResidentHandle = residentHandle;
            GpuResource = gpuResource;
            EstimatedGpuBytes = gpuResource == null ? 0 : estimatedGpuBytes;
            RuntimeDataPublished = residentHandle.IsValid;
            GpuReleaseComplete = gpuResource == null;
        }

        public RuntimeAssetResidencyKey Key { get; }

        public VegetationResidentResourceHandle ResidentHandle { get; }

        public IVegetationClusterGpuResource? GpuResource { get; }

        public long EstimatedGpuBytes { get; }

        public bool RuntimeDataPublished { get; set; }

        public bool GpuReleaseComplete { get; set; }

        public bool ClusterMappingPublished { get; set; }

        public bool GpuBytesCharged { get; set; }

        public bool ReleaseInProgress { get; set; }
    }

    private readonly record struct PreparedClusterPublication(
        Guid ClusterGuid,
        ulong ResidentGeneration,
        IVegetationClusterGpuResource Resource) :
        IComparable<PreparedClusterPublication>
    {
        public int CompareTo(PreparedClusterPublication other) =>
            ClusterGuid.CompareTo(other.ClusterGuid);
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
        CookedVegetationCluster? ClusterSource,
        CookedVegetationSpecies[]? Species,
        CookedVegetationInstancePage[]? Pages,
        RuntimeAssetResidencyKey[] RequiredKeys,
        RuntimeAssetPreparationClaim[] RequiredClaims)
    {
        public static WorkerPreparedResource ForCluster(
            VegetationPreparedClusterData cluster,
            CookedVegetationCluster clusterSource,
            CookedVegetationSpecies[] species,
            CookedVegetationInstancePage[] pages,
            RuntimeAssetResidencyKey[] requiredKeys,
            RuntimeAssetPreparationClaim[] requiredClaims) =>
            new(cluster, null, clusterSource, species, pages, requiredKeys, requiredClaims);

        public static WorkerPreparedResource ForPage(
            VegetationPreparedPageData page,
            RuntimeAssetResidencyKey[] requiredKeys,
            RuntimeAssetPreparationClaim[] requiredClaims) =>
            new(null, page, null, null, null, requiredKeys, requiredClaims);

        public static WorkerPreparedResource ForDependencies(
            RuntimeAssetResidencyKey[] requiredKeys,
            RuntimeAssetPreparationClaim[] requiredClaims) =>
            new(null, null, null, null, null, requiredKeys, requiredClaims);

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
