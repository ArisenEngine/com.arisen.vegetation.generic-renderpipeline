using Arisen.Native.RHI;
using ArisenEngine.Core.RHI;
using ArisenEngine.Rendering;
using ArisenEngine.Rendering.Resources;
using ArisenEngine.Resources.Serialization;
using ArisenEngine.Vegetation.Assets;
using System.Numerics;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;

namespace ArisenEngine.Vegetation.GenericRenderPipeline;

internal sealed class VegetationGpuResourceFactory : IVegetationClusterGpuResourceFactory
{
    private const uint InvalidBindlessIndex = uint.MaxValue;

    private readonly IGenericRenderPipelinePreparedAssetSource m_PreparedAssets;
    private readonly DeferredRenderResourceDisposalQueue m_DisposalQueue = new();
    private readonly VegetationGpuResourceRetirementState m_RetirementState = new();
    private readonly Action<IVegetationClusterGpuResource> m_RetirePendingResource;
    private readonly object m_PendingBufferGate = new();
    private readonly List<PendingBufferRelease> m_PendingBufferReleases = new();
    private RHIDevice m_Device;
    private ulong m_DeviceGeneration;
    private ulong m_LastSubmittedTicket;
    private int m_DeferredDisposalCount;
    private int m_SetupThreadId;

    public VegetationGpuResourceFactory(
        IGenericRenderPipelinePreparedAssetSource preparedAssets)
    {
        m_PreparedAssets = preparedAssets
            ?? throw new ArgumentNullException(nameof(preparedAssets));
        m_RetirePendingResource = RetirePendingResource;
    }

    public int PendingDisposalCount
    {
        get
        {
            int pendingBufferCount;
            lock (m_PendingBufferGate)
            {
                pendingBufferCount = m_PendingBufferReleases.Count;
            }

            return checked(
                m_RetirementState.PendingCount +
                Volatile.Read(ref m_DeferredDisposalCount) +
                pendingBufferCount);
        }
    }

    public void UpdateFrameContext(RHIDevice device, ulong deviceGeneration)
    {
        BindSetupThread();
        if (!device.IsValid)
        {
            throw new ArgumentException(
                "Vegetation GPU resources require a valid RHI device.",
                nameof(device));
        }
        if (deviceGeneration == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(deviceGeneration));
        }
        if (m_Device.IsValid &&
            (m_Device.Handle != device.Handle || m_DeviceGeneration != deviceGeneration))
        {
            throw new InvalidOperationException(
                $"Vegetation GPU resources are still bound to graphics generation " +
                $"{m_DeviceGeneration} and cannot bind generation {deviceGeneration}.");
        }

        m_DisposalQueue.BindDevice(device, deviceGeneration);
        m_Device = device;
        m_DeviceGeneration = deviceGeneration;
        DrainPendingResourceReleases();
    }

    public VegetationGpuResourceBuildResult TryCreate(
        CookedVegetationCluster cluster,
        IReadOnlyList<CookedVegetationSpecies> species,
        IReadOnlyList<CookedVegetationInstancePage> pages)
    {
        EnsureSetupThread();
        ArgumentNullException.ThrowIfNull(cluster);
        ArgumentNullException.ThrowIfNull(species);
        ArgumentNullException.ThrowIfNull(pages);
        if (!m_Device.IsValid || m_DeviceGeneration == 0)
        {
            return VegetationGpuResourceBuildResult.Waiting(
                "Vegetation cluster GPU setup is waiting for a valid RHI frame context.");
        }

        PreparedDependencyLeaseSet? dependencyLeases = null;
        try
        {
            ValidateClosure(cluster, species, pages);
            if (!TryAcquireDependencies(
                    species,
                    out dependencyLeases,
                    out string dependencyDiagnostic))
            {
                return VegetationGpuResourceBuildResult.Waiting(dependencyDiagnostic);
            }

            InstanceBuildRecord[] records = BuildRecords(cluster, species, pages);
            Array.Sort(records, InstanceBuildRecordComparer.Instance);

            var instances = new VegetationGpuInstance[records.Length];
            for (int index = 0; index < records.Length; index++)
            {
                ref readonly InstanceBuildRecord record = ref records[index];
                instances[index] = VegetationGpuInstancePacking.Pack(
                    record.Instance,
                    record.PageOrigin,
                    pages[0].Origin);
            }

            if (!TryBuildPendingBatches(
                    records,
                    dependencyLeases,
                    out PendingBatch[] pending,
                    out string diagnostic))
            {
                return VegetationGpuResourceBuildResult.Waiting(diagnostic);
            }

            if (!dependencyLeases.DependenciesCurrent)
            {
                return VegetationGpuResourceBuildResult.Waiting(
                    $"Vegetation cluster '{cluster.Guid:D}' prepared mesh or material " +
                    "dependencies changed before instance upload.");
            }

            VegetationGpuBuffer? instanceBuffer = null;
            try
            {
                instanceBuffer = VegetationGpuBuffer.Create(
                    m_Device,
                    instances,
                    $"Vegetation.{cluster.Guid:N}.Instances",
                    RetainPendingBufferRelease);
                var batches = new VegetationPreparedBatch[pending.Length];
                for (int index = 0; index < pending.Length; index++)
                {
                    ref readonly PendingBatch batch = ref pending[index];
                    batches[index] = batch.Create(instanceBuffer.BindlessIndex);
                }

                var resource = new VegetationClusterGpuResource(
                    m_DeviceGeneration,
                    cluster.Guid,
                    pages[0].Origin,
                    instances.Length,
                    batches,
                    instanceBuffer,
                    dependencyLeases);
                instanceBuffer = null;
                dependencyLeases = null;
                return VegetationGpuResourceBuildResult.Ready(resource);
            }
            finally
            {
                instanceBuffer?.Dispose();
            }
        }
        catch (Exception ex) when (
            ex is InvalidDataException or
            InvalidOperationException or
            ArgumentException or
            OverflowException or
            NotSupportedException)
        {
            return VegetationGpuResourceBuildResult.Failed(ex.Message);
        }
        finally
        {
            dependencyLeases?.Dispose();
        }
    }

    public void RequestRelease(IVegetationClusterGpuResource resource)
    {
        ArgumentNullException.ThrowIfNull(resource);
        if (resource is not VegetationClusterGpuResource)
        {
            throw new InvalidOperationException(
                "Vegetation GPU resource was not created by this factory type.");
        }

        m_RetirementState.RequestRelease(resource);
    }

    public void UpdateSubmittedTicket(ulong submittedTicket)
    {
        EnsureSetupThread();
        m_LastSubmittedTicket = Math.Max(m_LastSubmittedTicket, submittedTicket);
        DrainPendingResourceReleases();
        try
        {
            if (m_Device.IsValid)
            {
                m_DisposalQueue.ReleaseCompleted(m_Device, m_DeviceGeneration);
            }
        }
        finally
        {
            PublishDeferredDisposalCount();
        }
    }

    public void ReleaseAllDeviceResources()
    {
        EnsureSetupThread();
        DrainPendingResourceReleases();
        ReleasePendingBufferReleases();
        try
        {
            if (m_Device.IsValid)
            {
                m_DisposalQueue.ReleaseDevice(
                    m_Device,
                    m_DeviceGeneration,
                    m_LastSubmittedTicket);
            }
            else if (m_DisposalQueue.PendingCount != 0)
            {
                throw new InvalidOperationException(
                    $"Vegetation cannot release {m_DisposalQueue.PendingCount} deferred " +
                    "resources without a valid RHI device.");
            }
        }
        finally
        {
            PublishDeferredDisposalCount();
        }

        ReleasePendingBufferReleases();
        m_Device = default;
        m_DeviceGeneration = 0;
        m_LastSubmittedTicket = 0;
    }

    private void DrainPendingResourceReleases()
    {
        EnsureSetupThread();
        try
        {
            m_RetirementState.Drain(m_RetirePendingResource);
        }
        finally
        {
            PublishDeferredDisposalCount();
        }
    }

    private void PublishDeferredDisposalCount() =>
        Volatile.Write(ref m_DeferredDisposalCount, m_DisposalQueue.PendingCount);

    private void RetirePendingResource(IVegetationClusterGpuResource resource)
    {
        if (!m_Device.IsValid ||
            resource is not VegetationClusterGpuResource owned ||
            owned.DeviceGeneration != m_DeviceGeneration)
        {
            throw new InvalidOperationException(
                "Vegetation GPU resource does not belong to the active graphics generation.");
        }

        m_DisposalQueue.Enqueue(resource, m_LastSubmittedTicket);
    }

    private void BindSetupThread()
    {
        int currentThreadId = Environment.CurrentManagedThreadId;
        int ownerThreadId = Volatile.Read(ref m_SetupThreadId);
        if (ownerThreadId == 0)
        {
            ownerThreadId = Interlocked.CompareExchange(
                ref m_SetupThreadId,
                currentThreadId,
                comparand: 0);
            if (ownerThreadId == 0)
            {
                ownerThreadId = currentThreadId;
            }
        }

        if (ownerThreadId != currentThreadId)
        {
            throw new InvalidOperationException(
                $"Vegetation GPU resources are setup-thread affine to thread " +
                $"{ownerThreadId}; thread {currentThreadId} attempted access.");
        }
    }

    private void EnsureSetupThread()
    {
        int ownerThreadId = Volatile.Read(ref m_SetupThreadId);
        if (ownerThreadId != 0 && ownerThreadId != Environment.CurrentManagedThreadId)
        {
            throw new InvalidOperationException(
                $"Vegetation GPU resources are setup-thread affine to thread " +
                $"{ownerThreadId}; thread {Environment.CurrentManagedThreadId} attempted access.");
        }
    }

    private bool TryAcquireDependencies(
        IReadOnlyList<CookedVegetationSpecies> species,
        out PreparedDependencyLeaseSet dependencies,
        out string diagnostic)
    {
        var meshLeases = new Dictionary<
            RuntimeAssetResidencyKey,
            IGenericRenderPipelinePreparedMeshLease>();
        var materialLeases = new Dictionary<
            RuntimeAssetResidencyKey,
            IGenericRenderPipelinePreparedMaterialLease>();
        var acquired = new List<IGenericRenderPipelinePreparedAssetLease>();
        bool ownershipTransferred = false;
        try
        {
            for (int speciesIndex = 0; speciesIndex < species.Count; speciesIndex++)
            {
                CookedVegetationSpecies item = species[speciesIndex];
                CookedVegetationSpeciesLod lod = item.Lods[0];
                RuntimeAssetResidencyKey meshKey = CreateMeshKey(lod.Mesh);
                if (!meshLeases.ContainsKey(meshKey))
                {
                    if (!m_PreparedAssets.TryAcquirePreparedMesh(
                            meshKey,
                            out IGenericRenderPipelinePreparedMeshLease meshLease))
                    {
                        dependencies = null!;
                        diagnostic =
                            $"Vegetation species '{item.PackageId}:{item.Guid:D}' is waiting " +
                            $"for prepared mesh '{meshKey}'.";
                        return false;
                    }

                    acquired.Add(meshLease);
                    if (meshLease.Key != meshKey ||
                        meshLease.DeviceGeneration != m_DeviceGeneration ||
                        !meshLease.IsCurrent ||
                        !meshLease.Resource.IsValid)
                    {
                        dependencies = null!;
                        diagnostic =
                            $"Vegetation species '{item.PackageId}:{item.Guid:D}' acquired " +
                            $"a stale or mismatched mesh publication for '{meshKey}'.";
                        return false;
                    }
                    meshLeases.Add(meshKey, meshLease);
                }

                RuntimeAssetResidencyKey materialKey = CreateMaterialKey(lod.Material);
                if (!materialLeases.ContainsKey(materialKey))
                {
                    if (!m_PreparedAssets.TryAcquirePreparedMaterial(
                            materialKey,
                            out IGenericRenderPipelinePreparedMaterialLease materialLease))
                    {
                        dependencies = null!;
                        diagnostic =
                            $"Vegetation species '{item.PackageId}:{item.Guid:D}' is waiting " +
                            $"for prepared material '{materialKey}'.";
                        return false;
                    }

                    acquired.Add(materialLease);
                    if (materialLease.Key != materialKey ||
                        materialLease.DeviceGeneration != m_DeviceGeneration ||
                        !materialLease.IsCurrent ||
                        !materialLease.Resource.IsValid)
                    {
                        dependencies = null!;
                        diagnostic =
                            $"Vegetation species '{item.PackageId}:{item.Guid:D}' acquired " +
                            $"a stale or mismatched material publication for '{materialKey}'.";
                        return false;
                    }
                    materialLeases.Add(materialKey, materialLease);
                }
            }

            dependencies = new PreparedDependencyLeaseSet(
                meshLeases,
                materialLeases,
                acquired.ToArray());
            diagnostic = string.Empty;
            ownershipTransferred = true;
            return true;
        }
        finally
        {
            if (!ownershipTransferred)
            {
                DisposeAcquiredDependencies(acquired);
            }
        }
    }

    private static RuntimeAssetResidencyKey CreateMeshKey(
        in CookedVegetationMeshReference mesh) => new(
        mesh.Guid,
        mesh.PackageId,
        "Mesh",
        RuntimeAssetVariantPolicy.StaticMesh);

    private static RuntimeAssetResidencyKey CreateMaterialKey(
        in CookedVegetationMaterialReference material) => new(
        material.Guid,
        material.PackageId,
        "Material",
        RuntimeAssetVariantPolicy.Material);

    private static void DisposeAcquiredDependencies(
        List<IGenericRenderPipelinePreparedAssetLease> acquired)
    {
        List<Exception>? failures = null;
        for (int index = acquired.Count - 1; index >= 0; index--)
        {
            try
            {
                acquired[index].Dispose();
            }
            catch (Exception error)
            {
                (failures ??= new List<Exception>()).Add(error);
            }
        }

        if (failures != null)
        {
            throw new AggregateException(
                "Vegetation prepared dependency acquisition rollback failed.",
                failures);
        }
    }

    private void RetainPendingBufferRelease(
        RHIFactory factory,
        RHIBufferHandle buffer,
        string name)
    {
        if (!factory.IsValid || !buffer.IsValid)
        {
            throw new InvalidOperationException(
                "Vegetation cannot retain invalid pending buffer ownership.");
        }

        lock (m_PendingBufferGate)
        {
            m_PendingBufferReleases.Add(new PendingBufferRelease(factory, buffer, name));
        }
    }

    private void ReleasePendingBufferReleases()
    {
        List<Exception>? failures = null;
        lock (m_PendingBufferGate)
        {
            for (int index = m_PendingBufferReleases.Count - 1; index >= 0; index--)
            {
                PendingBufferRelease pending = m_PendingBufferReleases[index];
                try
                {
                    pending.Factory.ReleaseBuffer(pending.Buffer);
                    m_PendingBufferReleases.RemoveAt(index);
                }
                catch (Exception error)
                {
                    (failures ??= new List<Exception>()).Add(
                        new InvalidOperationException(
                            $"Vegetation pending buffer '{pending.Name}' failed to release.",
                            error));
                }
            }
        }

        if (failures != null)
        {
            throw new AggregateException(
                "Vegetation pending GPU buffer cleanup failed.",
                failures);
        }
    }

    private bool TryBuildPendingBatches(
        InstanceBuildRecord[] records,
        PreparedDependencyLeaseSet dependencyLeases,
        out PendingBatch[] batches,
        out string diagnostic)
    {
        var pending = new List<PendingBatch>();
        int groupStart = 0;
        while (groupStart < records.Length)
        {
            ref readonly InstanceBuildRecord first = ref records[groupStart];
            int groupEnd = groupStart + 1;
            while (groupEnd < records.Length && first.HasSameBatchKey(records[groupEnd]))
            {
                groupEnd++;
            }

            CookedVegetationSpeciesLod lod = first.Species.Lods[0];
            RuntimeAssetResidencyKey meshKey = CreateMeshKey(lod.Mesh);
            if (!dependencyLeases.TryGetMesh(
                    meshKey,
                    out IGenericRenderPipelinePreparedMeshLease meshLease) ||
                !meshLease.IsCurrent ||
                !meshLease.Resource.IsValid)
            {
                batches = Array.Empty<PendingBatch>();
                diagnostic =
                    $"Vegetation species '{first.Species.Guid:D}' is waiting for prepared mesh " +
                    $"'{meshKey}'.";
                return false;
            }
            RHIStaticMeshResource mesh = meshLease.Resource;
            if (mesh.VertexStride != MeshAssetCooker.StaticMeshVertexStride)
            {
                throw new NotSupportedException(
                    $"Vegetation mesh '{lod.Mesh.Guid:D}' uses unsupported vertex stride " +
                    $"{mesh.VertexStride}; expected {MeshAssetCooker.StaticMeshVertexStride}.");
            }
            RuntimeAssetResidencyKey materialKey = CreateMaterialKey(lod.Material);
            if (!dependencyLeases.TryGetMaterial(
                    materialKey,
                    out IGenericRenderPipelinePreparedMaterialLease materialLease) ||
                !materialLease.IsCurrent ||
                !materialLease.Resource.IsValid)
            {
                batches = Array.Empty<PendingBatch>();
                diagnostic =
                    $"Vegetation species '{first.Species.Guid:D}' is waiting for prepared " +
                    $"material '{materialKey}'.";
                return false;
            }
            RHIMaterialResource material = materialLease.Resource;

            VegetationPreparedMaterialData materialData = PrepareMaterial(material, lod.Material.Guid);
            ReadOnlySpan<MeshSubmesh> submeshes = mesh.Submeshes;
            for (int submeshIndex = 0; submeshIndex < submeshes.Length; submeshIndex++)
            {
                MeshSubmesh submesh = submeshes[submeshIndex];
                pending.Add(new PendingBatch(
                    first.Species.Guid,
                    lod.Mesh.Guid,
                    lod.Material.Guid,
                    mesh.VertexBuffer,
                    mesh.IndexBuffer,
                    mesh.IndexType,
                    submesh.IndexCount,
                    submesh.FirstIndex,
                    submesh.VertexOffset,
                    checked((uint)groupStart),
                    checked((uint)(groupEnd - groupStart)),
                    first.Species.ShadowPolicy,
                    materialData));
            }

            groupStart = groupEnd;
        }

        batches = pending.ToArray();
        diagnostic = string.Empty;
        return batches.Length > 0;
    }

    private static VegetationPreparedMaterialData PrepareMaterial(
        RHIMaterialResource material,
        Guid materialGuid)
    {
        MaterialRenderState state = material.RenderState;
        if (state.BlendEnabled ||
            state.CullMode != ECullModeFlagBits.CULL_MODE_NONE ||
            state.FrontFace != EFrontFace.FRONT_FACE_COUNTER_CLOCKWISE)
        {
            throw new NotSupportedException(
                $"Vegetation material '{materialGuid:D}' must use the Item 5 opaque, " +
                "two-sided, counter-clockwise render-state contract.");
        }
        if (!material.TryGetTexture2DConstants(
                MaterialTextureSlots.BaseColor,
                out MaterialTexture2DBindlessConstants baseColor))
        {
            throw new NotSupportedException(
                $"Vegetation material '{materialGuid:D}' must provide a BaseColor texture.");
        }

        Vector4 baseColorFactor = material.GetVector4PropertyOrDefault(
            MaterialPropertySlots.BaseColorFactor,
            Vector4.One);
        float metallic = material.GetScalarPropertyOrDefault(
            MaterialPropertySlots.MetallicFactor,
            0.0f);
        float roughness = material.GetScalarPropertyOrDefault(
            MaterialPropertySlots.RoughnessFactor,
            1.0f);
        if (!IsFinite(baseColorFactor) ||
            !float.IsFinite(metallic) ||
            !float.IsFinite(roughness))
        {
            throw new InvalidDataException(
                $"Vegetation material '{materialGuid:D}' contains non-finite PBR values.");
        }

        return new VegetationPreparedMaterialData(
            baseColorFactor,
            Math.Clamp(metallic, 0.0f, 1.0f),
            Math.Clamp(roughness, 0.04f, 1.0f),
            baseColor.ImageIndex,
            baseColor.SamplerIndex);
    }

    private static InstanceBuildRecord[] BuildRecords(
        CookedVegetationCluster cluster,
        IReadOnlyList<CookedVegetationSpecies> species,
        IReadOnlyList<CookedVegetationInstancePage> pages)
    {
        var records = new InstanceBuildRecord[cluster.InstanceCount];
        int recordCount = 0;
        for (int pageIndex = 0; pageIndex < pages.Count; pageIndex++)
        {
            CookedVegetationInstancePage page = pages[pageIndex];
            for (int instanceIndex = 0; instanceIndex < page.Instances.Count; instanceIndex++)
            {
                CookedVegetationInstance instance = page.Instances[instanceIndex];
                if (instance.SpeciesIndex >= page.Species.Count)
                {
                    throw new InvalidDataException(
                        $"Vegetation page '{page.Guid:D}' instance species index " +
                        $"{instance.SpeciesIndex} is outside its species table.");
                }

                CookedVegetationSpeciesReference reference =
                    page.Species[checked((int)instance.SpeciesIndex)];
                CookedVegetationSpecies resolved = FindSpecies(species, reference);
                records[recordCount++] = new InstanceBuildRecord(
                    resolved,
                    page.Origin,
                    instance);
            }
        }

        if (recordCount != records.Length)
        {
            throw new InvalidDataException(
                $"Vegetation cluster '{cluster.Guid:D}' instance count changed during GPU packing.");
        }
        return records;
    }

    private static CookedVegetationSpecies FindSpecies(
        IReadOnlyList<CookedVegetationSpecies> species,
        in CookedVegetationSpeciesReference reference)
    {
        for (int index = 0; index < species.Count; index++)
        {
            CookedVegetationSpecies candidate = species[index];
            if (candidate.Guid == reference.Guid &&
                string.Equals(candidate.PackageId, reference.PackageId, StringComparison.Ordinal))
            {
                return candidate;
            }
        }

        throw new InvalidDataException(
            $"Vegetation species '{reference.PackageId}:{reference.Guid:D}' is absent from " +
            "the exact cluster closure.");
    }

    private static void ValidateClosure(
        CookedVegetationCluster cluster,
        IReadOnlyList<CookedVegetationSpecies> species,
        IReadOnlyList<CookedVegetationInstancePage> pages)
    {
        if (cluster.Guid == Guid.Empty ||
            cluster.InstanceCount <= 0 ||
            species.Count != cluster.Species.Count ||
            pages.Count != cluster.Pages.Count ||
            pages.Count == 0)
        {
            throw new InvalidDataException("Vegetation GPU cluster closure is incomplete.");
        }

        WorldPosition origin = pages[0].Origin;
        int instanceCount = 0;
        for (int index = 0; index < pages.Count; index++)
        {
            CookedVegetationInstancePage page = pages[index];
            if (page.ClusterGuid != cluster.Guid || page.Origin != origin)
            {
                throw new InvalidDataException(
                    $"Vegetation cluster '{cluster.Guid:D}' pages do not share one cluster origin.");
            }
            instanceCount = checked(instanceCount + page.Instances.Count);
        }
        if (instanceCount != cluster.InstanceCount)
        {
            throw new InvalidDataException(
                $"Vegetation cluster '{cluster.Guid:D}' GPU instance count is inconsistent.");
        }

        for (int index = 0; index < species.Count; index++)
        {
            CookedVegetationSpecies item = species[index];
            if (item.Lods is not { Count: > 0 })
            {
                throw new InvalidDataException(
                    $"Vegetation species '{item.Guid:D}' has no renderable LOD.");
            }
        }
    }

    private static bool IsFinite(Vector4 value) =>
        float.IsFinite(value.X) &&
        float.IsFinite(value.Y) &&
        float.IsFinite(value.Z) &&
        float.IsFinite(value.W);

    private readonly struct InstanceBuildRecord
    {
        public InstanceBuildRecord(
            CookedVegetationSpecies species,
            WorldPosition pageOrigin,
            in CookedVegetationInstance instance)
        {
            Species = species;
            PageOrigin = pageOrigin;
            Instance = instance;
        }

        public CookedVegetationSpecies Species { get; }
        public WorldPosition PageOrigin { get; }
        public CookedVegetationInstance Instance { get; }

        public bool HasSameBatchKey(in InstanceBuildRecord other)
        {
            CookedVegetationSpeciesLod left = Species.Lods[0];
            CookedVegetationSpeciesLod right = other.Species.Lods[0];
            return Species.Guid == other.Species.Guid &&
                string.Equals(
                    Species.PackageId,
                    other.Species.PackageId,
                    StringComparison.Ordinal) &&
                left.Mesh == right.Mesh &&
                left.Material == right.Material &&
                Species.ShadowPolicy == other.Species.ShadowPolicy;
        }
    }

    private sealed class InstanceBuildRecordComparer : IComparer<InstanceBuildRecord>
    {
        public static InstanceBuildRecordComparer Instance { get; } = new();

        public int Compare(InstanceBuildRecord left, InstanceBuildRecord right)
        {
            CookedVegetationSpeciesLod leftLod = left.Species.Lods[0];
            CookedVegetationSpeciesLod rightLod = right.Species.Lods[0];
            int result = CreateMeshKey(leftLod.Mesh).CompareTo(
                CreateMeshKey(rightLod.Mesh));
            if (result != 0) return result;
            result = CreateMaterialKey(leftLod.Material).CompareTo(
                CreateMaterialKey(rightLod.Material));
            if (result != 0) return result;
            result = left.Species.ShadowPolicy.CompareTo(right.Species.ShadowPolicy);
            if (result != 0) return result;
            result = left.Species.Guid.CompareTo(right.Species.Guid);
            if (result != 0) return result;
            result = StringComparer.Ordinal.Compare(
                left.Species.PackageId,
                right.Species.PackageId);
            return result != 0
                ? result
                : left.Instance.StableKey.CompareTo(right.Instance.StableKey);
        }
    }

    private readonly struct PendingBatch
    {
        public PendingBatch(
            Guid speciesGuid,
            Guid meshGuid,
            Guid materialGuid,
            RHIBufferHandle vertexBuffer,
            RHIBufferHandle indexBuffer,
            EIndexType indexType,
            uint indexCount,
            uint firstIndex,
            int vertexOffset,
            uint firstInstance,
            uint instanceCount,
            VegetationShadowPolicy shadowPolicy,
            in VegetationPreparedMaterialData material)
        {
            SpeciesGuid = speciesGuid;
            MeshGuid = meshGuid;
            MaterialGuid = materialGuid;
            VertexBuffer = vertexBuffer;
            IndexBuffer = indexBuffer;
            IndexType = indexType;
            IndexCount = indexCount;
            FirstIndex = firstIndex;
            VertexOffset = vertexOffset;
            FirstInstance = firstInstance;
            InstanceCount = instanceCount;
            ShadowPolicy = shadowPolicy;
            Material = material;
        }

        public Guid SpeciesGuid { get; }
        public Guid MeshGuid { get; }
        public Guid MaterialGuid { get; }
        public RHIBufferHandle VertexBuffer { get; }
        public RHIBufferHandle IndexBuffer { get; }
        public EIndexType IndexType { get; }
        public uint IndexCount { get; }
        public uint FirstIndex { get; }
        public int VertexOffset { get; }
        public uint FirstInstance { get; }
        public uint InstanceCount { get; }
        public VegetationShadowPolicy ShadowPolicy { get; }
        public VegetationPreparedMaterialData Material { get; }

        public VegetationPreparedBatch Create(uint instanceBufferIndex) => new(
            SpeciesGuid,
            MeshGuid,
            MaterialGuid,
            VertexBuffer,
            IndexBuffer,
            IndexType,
            IndexCount,
            FirstIndex,
            VertexOffset,
            instanceBufferIndex,
            FirstInstance,
            InstanceCount,
            ShadowPolicy,
            Material);
    }

    private sealed class PreparedDependencyLeaseSet : IDisposable
    {
        private readonly object m_DisposeGate = new();
        private readonly Dictionary<
            RuntimeAssetResidencyKey,
            IGenericRenderPipelinePreparedMeshLease> m_Meshes;
        private readonly Dictionary<
            RuntimeAssetResidencyKey,
            IGenericRenderPipelinePreparedMaterialLease> m_Materials;
        private readonly IGenericRenderPipelinePreparedAssetLease?[] m_OwnedLeases;

        public PreparedDependencyLeaseSet(
            Dictionary<
                RuntimeAssetResidencyKey,
                IGenericRenderPipelinePreparedMeshLease> meshes,
            Dictionary<
                RuntimeAssetResidencyKey,
                IGenericRenderPipelinePreparedMaterialLease> materials,
            IGenericRenderPipelinePreparedAssetLease[] ownedLeases)
        {
            m_Meshes = meshes ?? throw new ArgumentNullException(nameof(meshes));
            m_Materials = materials ?? throw new ArgumentNullException(nameof(materials));
            m_OwnedLeases = ownedLeases
                ?? throw new ArgumentNullException(nameof(ownedLeases));
        }

        public bool DependenciesCurrent
        {
            get
            {
                if (m_OwnedLeases.Length == 0)
                {
                    return false;
                }

                for (int index = 0; index < m_OwnedLeases.Length; index++)
                {
                    IGenericRenderPipelinePreparedAssetLease? lease =
                        Volatile.Read(ref m_OwnedLeases[index]);
                    if (lease == null || !lease.IsCurrent)
                    {
                        return false;
                    }
                }

                return true;
            }
        }

        public bool TryGetMesh(
            in RuntimeAssetResidencyKey key,
            out IGenericRenderPipelinePreparedMeshLease lease) =>
            m_Meshes.TryGetValue(key, out lease!);

        public bool TryGetMaterial(
            in RuntimeAssetResidencyKey key,
            out IGenericRenderPipelinePreparedMaterialLease lease) =>
            m_Materials.TryGetValue(key, out lease!);

        public void Dispose()
        {
            lock (m_DisposeGate)
            {
                List<Exception>? failures = null;
                for (int index = m_OwnedLeases.Length - 1; index >= 0; index--)
                {
                    IGenericRenderPipelinePreparedAssetLease? lease =
                        m_OwnedLeases[index];
                    if (lease == null)
                    {
                        continue;
                    }

                    try
                    {
                        lease.Dispose();
                        m_OwnedLeases[index] = null;
                    }
                    catch (Exception error)
                    {
                        (failures ??= new List<Exception>()).Add(error);
                    }
                }

                if (failures != null)
                {
                    throw new AggregateException(
                        "Vegetation prepared dependency lease release failed.",
                        failures);
                }
            }
        }
    }

    private sealed class VegetationClusterGpuResource : IVegetationClusterGpuResource
    {
        private readonly Guid m_ClusterGuid;
        private readonly WorldPosition m_Origin;
        private readonly int m_InstanceCount;
        private readonly VegetationPreparedBatch[] m_Batches;
        private VegetationGpuBuffer? m_InstanceBuffer;
        private PreparedDependencyLeaseSet? m_DependencyLeases;

        public VegetationClusterGpuResource(
            ulong deviceGeneration,
            Guid clusterGuid,
            WorldPosition origin,
            int instanceCount,
            VegetationPreparedBatch[] batches,
            VegetationGpuBuffer instanceBuffer,
            PreparedDependencyLeaseSet dependencyLeases)
        {
            DeviceGeneration = deviceGeneration;
            m_ClusterGuid = clusterGuid;
            m_Origin = origin;
            m_InstanceCount = instanceCount;
            m_Batches = batches;
            m_InstanceBuffer = instanceBuffer;
            m_DependencyLeases = dependencyLeases;
        }

        public ulong DeviceGeneration { get; }
        public long EstimatedGpuBytes => m_InstanceBuffer?.SizeInBytes ?? 0;
        public bool DependenciesCurrent =>
            Volatile.Read(ref m_DependencyLeases)?.DependenciesCurrent == true;

        public VegetationPreparedClusterView CreateView(ulong generation)
        {
            if (generation == 0 ||
                Volatile.Read(ref m_InstanceBuffer) == null ||
                !DependenciesCurrent)
            {
                return default;
            }
            return new VegetationPreparedClusterView(
                m_ClusterGuid,
                generation,
                m_Origin,
                m_Batches,
                m_InstanceCount);
        }

        public void Dispose()
        {
            VegetationGpuBuffer? instanceBuffer = Volatile.Read(ref m_InstanceBuffer);
            if (instanceBuffer != null)
            {
                instanceBuffer.Dispose();
                Interlocked.CompareExchange(ref m_InstanceBuffer, null, instanceBuffer);
            }

            PreparedDependencyLeaseSet? dependencyLeases =
                Volatile.Read(ref m_DependencyLeases);
            if (dependencyLeases == null)
            {
                return;
            }

            dependencyLeases.Dispose();
            Interlocked.CompareExchange(
                ref m_DependencyLeases,
                null,
                dependencyLeases);
        }
    }

    private sealed class VegetationGpuBuffer : IDisposable
    {
        private readonly object m_DisposeGate = new();
        private RHIFactory m_Factory;

        private VegetationGpuBuffer(
            RHIFactory factory,
            RHIBufferHandle buffer,
            uint bindlessIndex,
            long sizeInBytes)
        {
            m_Factory = factory;
            Buffer = buffer;
            BindlessIndex = bindlessIndex;
            SizeInBytes = sizeInBytes;
        }

        public RHIBufferHandle Buffer { get; private set; }
        public uint BindlessIndex { get; private set; }
        public long SizeInBytes { get; }

        public static unsafe VegetationGpuBuffer Create(
            RHIDevice device,
            ReadOnlySpan<VegetationGpuInstance> data,
            string name,
            Action<RHIFactory, RHIBufferHandle, string> retainPendingBuffer)
        {
            ArgumentNullException.ThrowIfNull(retainPendingBuffer);
            RHIBufferHandle buffer = Upload(
                device,
                data,
                name,
                retainPendingBuffer);
            RHIFactory factory = device.GetFactory();
            uint bindlessIndex;
            try
            {
                bindlessIndex = factory.RegisterBindlessResourceBuffer(buffer);
            }
            catch (Exception registrationFailure)
            {
                var cleanupFailures = new List<Exception>(1);
                TryReleaseConstructionBuffer(
                    factory,
                    ref buffer,
                    name,
                    retainPendingBuffer,
                    cleanupFailures);
                if (cleanupFailures.Count != 0)
                {
                    throw new AggregateException(
                        $"Vegetation instance buffer '{name}' registration and rollback failed.",
                        new[] { registrationFailure }.Concat(cleanupFailures));
                }

                throw;
            }

            if (bindlessIndex == InvalidBindlessIndex)
            {
                var registrationFailure = new InvalidOperationException(
                    $"[Vegetation.GenericRP] Failed to register instance buffer '{name}'.");
                var cleanupFailures = new List<Exception>(1);
                TryReleaseConstructionBuffer(
                    factory,
                    ref buffer,
                    name,
                    retainPendingBuffer,
                    cleanupFailures);
                if (cleanupFailures.Count != 0)
                {
                    throw new AggregateException(
                        $"Vegetation instance buffer '{name}' registration and rollback failed.",
                        new[] { registrationFailure }.Concat(cleanupFailures));
                }

                throw registrationFailure;
            }

            return new VegetationGpuBuffer(
                factory,
                buffer,
                bindlessIndex,
                checked((long)data.Length * sizeof(VegetationGpuInstance)));
        }

        public void Dispose()
        {
            lock (m_DisposeGate)
            {
                if (!m_Factory.IsValid)
                {
                    if (!Buffer.IsValid && BindlessIndex == InvalidBindlessIndex)
                    {
                        return;
                    }

                    throw new InvalidOperationException(
                        "Vegetation GPU buffer ownership cannot be released without a valid RHI factory.");
                }

                List<Exception>? failures = null;
                if (BindlessIndex != InvalidBindlessIndex)
                {
                    try
                    {
                        m_Factory.UnregisterBindlessResourceBuffer(BindlessIndex);
                        BindlessIndex = InvalidBindlessIndex;
                    }
                    catch (Exception error)
                    {
                        failures = new List<Exception>(2)
                        {
                            new InvalidOperationException(
                                "Vegetation instance buffer bindless registration failed to release.",
                                error)
                        };
                    }
                }

                if (BindlessIndex == InvalidBindlessIndex && Buffer.IsValid)
                {
                    try
                    {
                        m_Factory.ReleaseBuffer(Buffer);
                        Buffer = RHIBufferHandle.Invalid;
                    }
                    catch (Exception error)
                    {
                        (failures ??= new List<Exception>(1)).Add(
                            new InvalidOperationException(
                                "Vegetation instance buffer failed to release.",
                                error));
                    }
                }

                if (!Buffer.IsValid && BindlessIndex == InvalidBindlessIndex)
                {
                    m_Factory = default;
                }

                if (failures != null)
                {
                    throw new AggregateException(
                        "Vegetation GPU buffer disposal failed.",
                        failures);
                }
            }
        }

        private static unsafe RHIBufferHandle Upload(
            RHIDevice device,
            ReadOnlySpan<VegetationGpuInstance> data,
            string name,
            Action<RHIFactory, RHIBufferHandle, string> retainPendingBuffer)
        {
            if (!device.IsValid || data.IsEmpty)
            {
                throw new ArgumentException(
                    "Vegetation instance upload requires a valid device and non-empty data.");
            }

            ulong byteSize = checked(
                (ulong)data.Length * (ulong)sizeof(VegetationGpuInstance));
            RHIFactory factory = device.GetFactory();
            RHIBufferHandle staging = factory.CreateBuffer(
                byteSize,
                (uint)EBufferUsageFlagBits.BUFFER_USAGE_TRANSFER_SRC_BIT,
                ESharingMode.SHARING_MODE_EXCLUSIVE,
                ERHIMemoryUsage.Upload,
                name + ".Upload");
            RHIBufferHandle destination = RHIBufferHandle.Invalid;
            Exception? uploadFailure = null;
            try
            {
                if (!staging.IsValid)
                {
                    throw new InvalidOperationException(
                        $"[Vegetation.GenericRP] Failed to create staging buffer '{name}'.");
                }

                IntPtr mapped = factory.MapBuffer(staging);
                if (mapped == IntPtr.Zero)
                {
                    throw new InvalidOperationException(
                        $"[Vegetation.GenericRP] Failed to map staging buffer '{name}'.");
                }
                try
                {
                    MemoryMarshal.AsBytes(data).CopyTo(
                        new Span<byte>((void*)mapped, checked((int)byteSize)));
                }
                finally
                {
                    factory.UnmapBuffer(staging);
                }

                destination = factory.CreateBuffer(
                    byteSize,
                    (uint)EBufferUsageFlagBits.BUFFER_USAGE_TRANSFER_DST_BIT |
                    (uint)EBufferUsageFlagBits.BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    ESharingMode.SHARING_MODE_EXCLUSIVE,
                    ERHIMemoryUsage.GpuOnly,
                    name);
                if (!destination.IsValid)
                {
                    throw new InvalidOperationException(
                        $"[Vegetation.GenericRP] Failed to create instance buffer '{name}'.");
                }

                RHICommandBufferPool commandPool =
                    factory.CreateCommandBufferPool(RHIQueueType.Graphics);
                RHICommandBuffer commandBuffer = default;
                try
                {
                    commandBuffer = commandPool.GetCommandBuffer(0);
                    commandBuffer.Begin();
                    commandBuffer.CopyBuffer(staging, 0, destination, 0, byteSize);
                    Span<RHIBufferMemoryBarrier> barriers = stackalloc RHIBufferMemoryBarrier[1];
                    barriers[0] = new RHIBufferMemoryBarrier
                    {
                        SrcAccessMask = EAccessFlag.ACCESS_TRANSFER_WRITE_BIT,
                        DstAccessMask = EAccessFlag.ACCESS_SHADER_READ_BIT,
                        SrcQueueFamilyIndex = RHIQueueFamily.Ignored,
                        DstQueueFamilyIndex = RHIQueueFamily.Ignored,
                        Buffer = destination,
                        SrcStageMask = EPipelineStageFlagBits.PIPELINE_STAGE_TRANSFER_BIT,
                        DstStageMask = EPipelineStageFlagBits.PIPELINE_STAGE_VERTEX_SHADER_BIT
                    };
                    commandBuffer.PipelineBarrier(
                        EPipelineStageFlagBits.PIPELINE_STAGE_TRANSFER_BIT,
                        EPipelineStageFlagBits.PIPELINE_STAGE_VERTEX_SHADER_BIT,
                        barriers);
                    commandBuffer.End();
                    ulong ticket = device.Submit(commandBuffer);
                    device.WaitQueueTicket(ticket);
                }
                finally
                {
                    if (commandBuffer.IsValid)
                    {
                        commandPool.ReleaseCommandBuffer(0, commandBuffer.RHIHandle);
                    }
                    if (commandPool.IsValid)
                    {
                        factory.ReleaseCommandBufferPool(commandPool.RHIHandle);
                    }
                }
            }
            catch (Exception error)
            {
                uploadFailure = error;
            }

            var cleanupFailures = new List<Exception>(2);
            TryReleaseConstructionBuffer(
                factory,
                ref staging,
                name + ".Upload",
                retainPendingBuffer,
                cleanupFailures);
            if (uploadFailure != null || cleanupFailures.Count != 0)
            {
                TryReleaseConstructionBuffer(
                    factory,
                    ref destination,
                    name,
                    retainPendingBuffer,
                    cleanupFailures);
                if (cleanupFailures.Count != 0)
                {
                    IEnumerable<Exception> failures = uploadFailure == null
                        ? cleanupFailures
                        : new[] { uploadFailure }.Concat(cleanupFailures);
                    throw new AggregateException(
                        $"Vegetation instance buffer '{name}' construction cleanup failed.",
                        failures);
                }

                ExceptionDispatchInfo.Capture(uploadFailure!).Throw();
            }

            return destination;
        }

        private static void TryReleaseConstructionBuffer(
            RHIFactory factory,
            ref RHIBufferHandle buffer,
            string name,
            Action<RHIFactory, RHIBufferHandle, string> retainPendingBuffer,
            ICollection<Exception> failures)
        {
            if (!buffer.IsValid)
            {
                return;
            }

            RHIBufferHandle owned = buffer;
            try
            {
                factory.ReleaseBuffer(owned);
                buffer = RHIBufferHandle.Invalid;
            }
            catch (Exception releaseFailure)
            {
                retainPendingBuffer(factory, owned, name);
                buffer = RHIBufferHandle.Invalid;
                failures.Add(new InvalidOperationException(
                    $"Vegetation construction buffer '{name}' was retained after release failed.",
                    releaseFailure));
            }
        }
    }

    private readonly record struct PendingBufferRelease(
        RHIFactory Factory,
        RHIBufferHandle Buffer,
        string Name);
}
