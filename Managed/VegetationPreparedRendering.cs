using Arisen.Native.RHI;
using ArisenEngine.Core.RHI;
using ArisenEngine.Resources.Serialization;
using ArisenEngine.Vegetation.Assets;
using System.Numerics;
using System.Runtime.InteropServices;

namespace ArisenEngine.Vegetation.GenericRenderPipeline;

[StructLayout(LayoutKind.Sequential)]
internal readonly struct VegetationGpuInstance
{
    public const int Stride = 48;

    public readonly Vector4 OriginRelativePositionScale;
    public readonly Quaternion Orientation;
    public readonly uint StableVariation;
    public readonly float WindPhase;
    public readonly float ColorVariation;
    public readonly uint Flags;

    public VegetationGpuInstance(
        Vector4 originRelativePositionScale,
        Quaternion orientation,
        uint stableVariation,
        float windPhase,
        float colorVariation,
        uint flags)
    {
        OriginRelativePositionScale = originRelativePositionScale;
        Orientation = orientation;
        StableVariation = stableVariation;
        WindPhase = windPhase;
        ColorVariation = colorVariation;
        Flags = flags;
    }
}

internal static class VegetationGpuInstancePacking
{
    private const double MaximumExactFloatInteger = 16_777_216.0;
    private const float InverseUInt24 = 1.0f / 16_777_215.0f;

    public static VegetationGpuInstance Pack(
        in CookedVegetationInstance instance,
        WorldPosition pageOrigin,
        WorldPosition clusterOrigin,
        uint flags = 0)
    {
        Vector3 position = ToOriginRelativePosition(
            instance.LocalPosition,
            pageOrigin,
            clusterOrigin);
        if (!IsFinite(instance.Orientation) ||
            !float.IsFinite(instance.UniformScale) ||
            instance.UniformScale <= 0.0f)
        {
            throw new InvalidDataException(
                "Vegetation GPU instance transform is invalid.");
        }

        ulong variationHash = Mix(instance.StableKey ^ 0xA0761D6478BD642FUL);
        ulong colorHash = Mix(instance.StableKey ^ 0xE7037ED1A0B428DBUL);
        uint stableVariation = unchecked((uint)variationHash);
        float windPhase = UInt24ToUnit(variationHash >> 40) * MathF.Tau;
        float colorVariation = 0.85f + UInt24ToUnit(colorHash >> 40) * 0.30f;
        return new VegetationGpuInstance(
            new Vector4(position, instance.UniformScale),
            instance.Orientation,
            stableVariation,
            windPhase,
            colorVariation,
            flags);
    }

    private static Vector3 ToOriginRelativePosition(
        Vector3 localPosition,
        WorldPosition pageOrigin,
        WorldPosition clusterOrigin)
    {
        double x = pageOrigin.X - clusterOrigin.X + localPosition.X;
        double y = pageOrigin.Y - clusterOrigin.Y + localPosition.Y;
        double z = pageOrigin.Z - clusterOrigin.Z + localPosition.Z;
        if (!double.IsFinite(x) ||
            !double.IsFinite(y) ||
            !double.IsFinite(z) ||
            Math.Abs(x) > MaximumExactFloatInteger ||
            Math.Abs(y) > MaximumExactFloatInteger ||
            Math.Abs(z) > MaximumExactFloatInteger)
        {
            throw new InvalidDataException(
                "Vegetation GPU instance exceeds the origin-relative float range.");
        }

        return new Vector3((float)x, (float)y, (float)z);
    }

    private static float UInt24ToUnit(ulong value) =>
        (float)(value & 0x00FF_FFFFUL) * InverseUInt24;

    private static ulong Mix(ulong value)
    {
        value ^= value >> 30;
        value *= 0xBF58476D1CE4E5B9UL;
        value ^= value >> 27;
        value *= 0x94D049BB133111EBUL;
        return value ^ (value >> 31);
    }

    private static bool IsFinite(Quaternion value) =>
        float.IsFinite(value.X) &&
        float.IsFinite(value.Y) &&
        float.IsFinite(value.Z) &&
        float.IsFinite(value.W);
}

[StructLayout(LayoutKind.Sequential)]
internal readonly struct VegetationPreparedMaterialData
{
    public readonly Vector4 BaseColorFactor;
    public readonly float MetallicFactor;
    public readonly float RoughnessFactor;
    public readonly uint BaseColorImageIndex;
    public readonly uint BaseColorSamplerIndex;

    public VegetationPreparedMaterialData(
        Vector4 baseColorFactor,
        float metallicFactor,
        float roughnessFactor,
        uint baseColorImageIndex,
        uint baseColorSamplerIndex)
    {
        BaseColorFactor = baseColorFactor;
        MetallicFactor = metallicFactor;
        RoughnessFactor = roughnessFactor;
        BaseColorImageIndex = baseColorImageIndex;
        BaseColorSamplerIndex = baseColorSamplerIndex;
    }
}

internal readonly struct VegetationPreparedBatch
{
    public VegetationPreparedBatch(
        Guid speciesGuid,
        Guid meshGuid,
        Guid materialGuid,
        RHIBufferHandle vertexBuffer,
        RHIBufferHandle indexBuffer,
        EIndexType indexType,
        uint indexCount,
        uint firstIndex,
        int vertexOffset,
        uint instanceBufferIndex,
        uint firstInstance,
        uint instanceCount,
        VegetationShadowPolicy shadowPolicy,
        in VegetationPreparedMaterialData material)
        : this(
            speciesGuid,
            meshGuid,
            materialGuid,
            vertexBuffer,
            indexBuffer,
            indexType,
            indexCount,
            firstIndex,
            vertexOffset,
            instanceBufferIndex,
            firstInstance,
            instanceCount,
            lodLevel: 0,
            maximumDistance: float.MaxValue,
            maximumScreenError: float.MaxValue,
            shadowPolicy,
            material)
    {
    }

    public VegetationPreparedBatch(
        Guid speciesGuid,
        Guid meshGuid,
        Guid materialGuid,
        RHIBufferHandle vertexBuffer,
        RHIBufferHandle indexBuffer,
        EIndexType indexType,
        uint indexCount,
        uint firstIndex,
        int vertexOffset,
        uint instanceBufferIndex,
        uint firstInstance,
        uint instanceCount,
        int lodLevel,
        float maximumDistance,
        float maximumScreenError,
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
        InstanceBufferIndex = instanceBufferIndex;
        FirstInstance = firstInstance;
        InstanceCount = instanceCount;
        LodLevel = lodLevel;
        MaximumDistance = maximumDistance;
        MaximumScreenError = maximumScreenError;
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
    public uint InstanceBufferIndex { get; }
    public uint FirstInstance { get; }
    public uint InstanceCount { get; }
    public int LodLevel { get; }
    public float MaximumDistance { get; }
    public float MaximumScreenError { get; }
    public VegetationShadowPolicy ShadowPolicy { get; }
    public VegetationPreparedMaterialData Material { get; }
    public bool IsValid =>
        VertexBuffer.IsValid &&
        IndexBuffer.IsValid &&
        IndexCount > 0 &&
        LodLevel >= 0 &&
        float.IsFinite(MaximumDistance) &&
        MaximumDistance >= 0.0f &&
        float.IsFinite(MaximumScreenError) &&
        MaximumScreenError >= 0.0f &&
        InstanceBufferIndex != uint.MaxValue &&
        InstanceCount > 0;
}

internal readonly struct VegetationPreparedClusterView
{
    private readonly VegetationPreparedBatch[]? m_Batches;
    private readonly CookedVegetationSpecies[]? m_Species;

    public VegetationPreparedClusterView(
        Guid clusterGuid,
        ulong generation,
        WorldPosition origin,
        VegetationPreparedBatch[] batches,
        int instanceCount)
        : this(
            clusterGuid,
            generation,
            origin,
            batches,
            instanceCount,
            acceleration: null,
            species: null)
    {
    }

    public VegetationPreparedClusterView(
        Guid clusterGuid,
        ulong generation,
        WorldPosition origin,
        VegetationPreparedBatch[] batches,
        int instanceCount,
        CookedVegetationClusterAcceleration? acceleration,
        CookedVegetationSpecies[]? species)
    {
        ClusterGuid = clusterGuid;
        Generation = generation;
        Origin = origin;
        m_Batches = batches ?? throw new ArgumentNullException(nameof(batches));
        m_Species = species;
        InstanceCount = instanceCount;
        Acceleration = acceleration;
    }

    public Guid ClusterGuid { get; }
    public ulong Generation { get; }
    public WorldPosition Origin { get; }
    public int InstanceCount { get; }
    public CookedVegetationClusterAcceleration? Acceleration { get; }
    public ReadOnlySpan<VegetationPreparedBatch> Batches => m_Batches;
    public ReadOnlySpan<CookedVegetationSpecies> Species => m_Species;
    public bool IsValid =>
        ClusterGuid != Guid.Empty &&
        Generation != 0 &&
        Origin.IsFinite &&
        m_Batches is { Length: > 0 } &&
        InstanceCount > 0;
}

internal enum VegetationGpuResourceBuildStatus
{
    Ready = 0,
    Waiting = 1,
    Failed = 2
}

internal readonly record struct VegetationGpuResourceBuildResult(
    VegetationGpuResourceBuildStatus Status,
    IVegetationClusterGpuResource? Resource,
    string Diagnostic)
{
    public static VegetationGpuResourceBuildResult Ready(
        IVegetationClusterGpuResource resource) =>
        new(VegetationGpuResourceBuildStatus.Ready, resource, string.Empty);

    public static VegetationGpuResourceBuildResult Waiting(string diagnostic) =>
        new(VegetationGpuResourceBuildStatus.Waiting, null, diagnostic);

    public static VegetationGpuResourceBuildResult Failed(string diagnostic) =>
        new(VegetationGpuResourceBuildStatus.Failed, null, diagnostic);
}

internal interface IVegetationClusterGpuResource : IDisposable
{
    long EstimatedGpuBytes { get; }

    bool DependenciesCurrent { get; }

    VegetationPreparedClusterView CreateView(ulong generation);
}

internal interface IVegetationClusterGpuResourceFactory
{
    int PendingDisposalCount { get; }

    void UpdateFrameContext(RHIDevice device, ulong deviceGeneration);

    VegetationGpuResourceBuildResult TryCreate(
        CookedVegetationCluster cluster,
        IReadOnlyList<CookedVegetationSpecies> species,
        IReadOnlyList<CookedVegetationInstancePage> pages);

    void RequestRelease(IVegetationClusterGpuResource resource);

    void UpdateSubmittedTicket(ulong submittedTicket);

    void ReleaseAllDeviceResources();
}
