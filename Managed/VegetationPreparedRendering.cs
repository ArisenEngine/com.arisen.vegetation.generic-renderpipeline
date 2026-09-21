using Arisen.Native.RHI;
using ArisenEngine.Core.RHI;
using ArisenEngine.Rendering;
using ArisenEngine.Rendering.Resources;
using ArisenEngine.Resources.Serialization;
using ArisenEngine.Vegetation.Assets;
using System.Numerics;
using System.Runtime.InteropServices;

namespace ArisenEngine.Vegetation.GenericRenderPipeline;

[StructLayout(LayoutKind.Sequential)]
internal readonly struct VegetationGpuInstance
{
    public const int Stride = 48;

    /// <summary>
    /// XYZ is the instance position relative to its cluster origin. The cluster origin itself is
    /// supplied per draw and is expressed in the view frame, so this record stays independent of
    /// the world origin and can be uploaded once.
    /// </summary>
    public readonly Vector4 ClusterRelativePositionScale;
    public readonly Quaternion Orientation;
    public readonly uint StableVariation;
    public readonly float WindPhase;
    public readonly float ColorVariation;
    public readonly uint Flags;

    public VegetationGpuInstance(
        Vector4 clusterRelativePositionScale,
        Quaternion orientation,
        uint stableVariation,
        float windPhase,
        float colorVariation,
        uint flags)
    {
        ClusterRelativePositionScale = clusterRelativePositionScale;
        Orientation = orientation;
        StableVariation = stableVariation;
        WindPhase = windPhase;
        ColorVariation = colorVariation;
        Flags = flags;
    }
}

/// <summary>
/// The vegetation opaque and shadow passes render in the view frame: every vertex position they
/// consume is relative to the render camera, and the frame's view-projection is the rotation-only
/// <see cref="Camera.ViewRelativeViewMatrix"/> times the projection. Both the anchor and the matrix
/// then come from the camera instead of the rebaseable world origin, which is what keeps the pass
/// bit-stable across an origin rebase: origin-relative float positions at kilometre distances
/// quantize to roughly a tenth of a millimetre, which is enough to move rasterized vegetation
/// depth and alpha coverage.
/// </summary>
internal static class VegetationViewFrame
{
    public static Vector3 ToViewRelativePosition(
        WorldPosition worldPosition,
        WorldPosition cameraWorldPosition)
    {
        double x = worldPosition.X - cameraWorldPosition.X;
        double y = worldPosition.Y - cameraWorldPosition.Y;
        double z = worldPosition.Z - cameraWorldPosition.Z;
        if (!double.IsFinite(x) ||
            !double.IsFinite(y) ||
            !double.IsFinite(z) ||
            Math.Abs(x) > float.MaxValue ||
            Math.Abs(y) > float.MaxValue ||
            Math.Abs(z) > float.MaxValue)
        {
            throw new InvalidOperationException(
                "[Vegetation.GenericRP] Cluster is outside the view-frame float range.");
        }

        return new Vector3((float)x, (float)y, (float)z);
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

[Flags]
internal enum VegetationMaterialFlags : uint
{
    None = 0u,
    AlphaTest = 1u << 0,
    TintVariation = 1u << 1
}

internal readonly record struct VegetationMaterialSemantics(
    Vector4 BaseColorFactor,
    float AlphaCutoff,
    float MetallicFactor,
    float RoughnessFactor,
    float OcclusionStrength,
    float TintVariation,
    VegetationMaterialFlags Flags);

internal static class VegetationMaterialContract
{
    public const string TintVariationProperty = "TintVariation";
    private const uint InvalidBindlessIndex = 0xFFFFFFFFu;

    public static VegetationMaterialSemantics Resolve(
        MaterialAsset asset,
        Guid materialGuid)
    {
        ArgumentNullException.ThrowIfNull(asset);
        if (materialGuid == Guid.Empty)
        {
            throw new ArgumentException(
                "[Vegetation.GenericRP] Vegetation material GUID cannot be empty.",
                nameof(materialGuid));
        }

        MaterialRenderState state = asset.RenderState;
        if (state.BlendEnabled ||
            state.CullMode != ECullModeFlagBits.CULL_MODE_NONE ||
            state.FrontFace != EFrontFace.FRONT_FACE_COUNTER_CLOCKWISE)
        {
            throw new NotSupportedException(
                $"Vegetation material '{materialGuid:D}' must use the opaque, two-sided, " +
                "counter-clockwise render-state contract.");
        }

        RequireTexture(
            asset,
            materialGuid,
            MaterialTextureSlots.BaseColor,
            Texture2DVariantKey.MipmappedSRgb,
            required: true);
        RequireTexture(
            asset,
            materialGuid,
            MaterialTextureSlots.Normal,
            Texture2DVariantKey.MipmappedNormal,
            required: true);
        RequireTexture(
            asset,
            materialGuid,
            MaterialTextureSlots.MetallicRoughness,
            Texture2DVariantKey.MipmappedLinear,
            required: true);
        if (FindTexture(asset, MaterialTextureSlots.Occlusion) != null)
        {
            throw new NotSupportedException(
                $"Vegetation material '{materialGuid:D}' must bind its combined occlusion, " +
                "roughness, and metallic channels through the MetallicRoughness texture; " +
                "the Occlusion slot is not part of the vegetation material contract.");
        }

        Vector4 baseColorFactor = ResolveVector4(
            asset,
            MaterialPropertySlots.BaseColorFactor,
            Vector4.One,
            materialGuid,
            minimum: 0.0f,
            maximum: 1.0f);
        float alphaCutoff = ResolveScalar(
            asset,
            MaterialPropertySlots.AlphaCutoff,
            0.0f,
            materialGuid,
            minimum: 0.0f,
            maximum: 1.0f);
        float metallicFactor = ResolveScalar(
            asset,
            MaterialPropertySlots.MetallicFactor,
            0.0f,
            materialGuid,
            minimum: 0.0f,
            maximum: 1.0f);
        float roughnessFactor = ResolveScalar(
            asset,
            MaterialPropertySlots.RoughnessFactor,
            1.0f,
            materialGuid,
            minimum: 0.04f,
            maximum: 1.0f);
        float occlusionStrength = ResolveScalar(
            asset,
            MaterialPropertySlots.OcclusionStrength,
            MaterialPbrDefaults.OcclusionStrength,
            materialGuid,
            minimum: 0.0f,
            maximum: 1.0f);
        float tintVariation = ResolveScalar(
            asset,
            TintVariationProperty,
            0.0f,
            materialGuid,
            minimum: 0.0f,
            maximum: 1.0f);

        VegetationMaterialFlags flags = VegetationMaterialFlags.None;
        if (alphaCutoff > 0.0f)
        {
            flags |= VegetationMaterialFlags.AlphaTest;
        }
        if (tintVariation > 0.0f)
        {
            flags |= VegetationMaterialFlags.TintVariation;
        }

        return new VegetationMaterialSemantics(
            baseColorFactor,
            alphaCutoff,
            metallicFactor,
            roughnessFactor,
            occlusionStrength,
            tintVariation,
            flags);
    }

    private static void RequireTexture(
        MaterialAsset asset,
        Guid materialGuid,
        string slot,
        Texture2DVariantKey requiredVariant,
        bool required)
    {
        MaterialTexture2DRef? texture = FindTexture(asset, slot);
        if (texture == null)
        {
            if (required)
            {
                throw new NotSupportedException(
                    $"Vegetation material '{materialGuid:D}' must provide a '{slot}' texture.");
            }

            return;
        }

        Texture2DVariantKey declared = texture.Value.Texture.Variant;
        if (declared != requiredVariant)
        {
            throw new NotSupportedException(
                $"Vegetation material '{materialGuid:D}' texture '{slot}' declares variant " +
                $"'{declared.GetCookedVariant()}'; the vegetation contract requires " +
                $"'{requiredVariant.GetCookedVariant()}'.");
        }
    }

    private static MaterialTexture2DRef? FindTexture(MaterialAsset asset, string slot)
    {
        IReadOnlyList<MaterialTexture2DRef>? refs = asset.Texture2DRefs;
        if (refs == null)
        {
            return null;
        }

        for (int index = 0; index < refs.Count; index++)
        {
            if (string.Equals(refs[index].Name, slot, StringComparison.OrdinalIgnoreCase))
            {
                return refs[index];
            }
        }

        return null;
    }

    private static float ResolveScalar(
        MaterialAsset asset,
        string property,
        float defaultValue,
        Guid materialGuid,
        float minimum,
        float maximum)
    {
        float value = defaultValue;
        IReadOnlyList<MaterialScalarProperty>? properties = asset.ScalarProperties;
        if (properties != null)
        {
            for (int index = 0; index < properties.Count; index++)
            {
                if (string.Equals(properties[index].Name, property, StringComparison.OrdinalIgnoreCase))
                {
                    value = properties[index].Value;
                    break;
                }
            }
        }

        if (!float.IsFinite(value) || value < minimum || value > maximum)
        {
            throw new InvalidDataException(
                $"Vegetation material '{materialGuid:D}' property '{property}' value " +
                $"'{value}' must be finite and inside [{minimum}, {maximum}].");
        }

        return value;
    }

    private static Vector4 ResolveVector4(
        MaterialAsset asset,
        string property,
        Vector4 defaultValue,
        Guid materialGuid,
        float minimum,
        float maximum)
    {
        Vector4 value = defaultValue;
        IReadOnlyList<MaterialVector4Property>? properties = asset.Vector4Properties;
        if (properties != null)
        {
            for (int index = 0; index < properties.Count; index++)
            {
                if (string.Equals(properties[index].Name, property, StringComparison.OrdinalIgnoreCase))
                {
                    value = properties[index].Value;
                    break;
                }
            }
        }

        if (!float.IsFinite(value.X) ||
            !float.IsFinite(value.Y) ||
            !float.IsFinite(value.Z) ||
            !float.IsFinite(value.W) ||
            value.X < minimum || value.X > maximum ||
            value.Y < minimum || value.Y > maximum ||
            value.Z < minimum || value.Z > maximum ||
            value.W < minimum || value.W > maximum)
        {
            throw new InvalidDataException(
                $"Vegetation material '{materialGuid:D}' property '{property}' value " +
                $"'{value}' must have finite channels inside [{minimum}, {maximum}].");
        }

        return value;
    }
}

[StructLayout(LayoutKind.Sequential)]
internal readonly struct VegetationPreparedMaterialData
{
    public const uint InvalidImageIndex = 0xFFFFFFFFu;

    public readonly Vector4 BaseColorFactor;
    public readonly float AlphaCutoff;
    public readonly float MetallicFactor;
    public readonly float RoughnessFactor;
    public readonly float OcclusionStrength;
    public readonly float TintVariation;
    public readonly uint Flags;
    public readonly uint BaseColorImageIndex;
    public readonly uint BaseColorSamplerIndex;
    public readonly uint NormalImageIndex;
    public readonly uint NormalSamplerIndex;
    public readonly uint OrmImageIndex;
    public readonly uint OrmSamplerIndex;

    public VegetationPreparedMaterialData(
        Vector4 baseColorFactor,
        float alphaCutoff,
        float metallicFactor,
        float roughnessFactor,
        float occlusionStrength,
        float tintVariation,
        uint flags,
        uint baseColorImageIndex,
        uint baseColorSamplerIndex,
        uint normalImageIndex,
        uint normalSamplerIndex,
        uint ormImageIndex,
        uint ormSamplerIndex)
    {
        BaseColorFactor = baseColorFactor;
        AlphaCutoff = alphaCutoff;
        MetallicFactor = metallicFactor;
        RoughnessFactor = roughnessFactor;
        OcclusionStrength = occlusionStrength;
        TintVariation = tintVariation;
        Flags = flags;
        BaseColorImageIndex = baseColorImageIndex;
        BaseColorSamplerIndex = baseColorSamplerIndex;
        NormalImageIndex = normalImageIndex;
        NormalSamplerIndex = normalSamplerIndex;
        OrmImageIndex = ormImageIndex;
        OrmSamplerIndex = ormSamplerIndex;
    }

    public static VegetationPreparedMaterialData Create(
        in VegetationMaterialSemantics semantics,
        uint baseColorImageIndex,
        uint baseColorSamplerIndex,
        uint normalImageIndex,
        uint normalSamplerIndex,
        uint ormImageIndex,
        uint ormSamplerIndex)
    {
        if (baseColorImageIndex == InvalidImageIndex ||
            baseColorSamplerIndex == InvalidImageIndex ||
            normalImageIndex == InvalidImageIndex ||
            normalSamplerIndex == InvalidImageIndex ||
            ormImageIndex == InvalidImageIndex ||
            ormSamplerIndex == InvalidImageIndex)
        {
            throw new InvalidDataException(
                "Vegetation prepared material is missing a bindless texture descriptor.");
        }

        return new VegetationPreparedMaterialData(
            semantics.BaseColorFactor,
            semantics.AlphaCutoff,
            semantics.MetallicFactor,
            semantics.RoughnessFactor,
            semantics.OcclusionStrength,
            semantics.TintVariation,
            unchecked((uint)semantics.Flags),
            baseColorImageIndex,
            baseColorSamplerIndex,
            normalImageIndex,
            normalSamplerIndex,
            ormImageIndex,
            ormSamplerIndex);
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
            windStiffness: 0.0f,
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
        float windStiffness,
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
        WindStiffness = windStiffness;
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
    public float WindStiffness { get; }
    public VegetationShadowPolicy ShadowPolicy { get; }
    public VegetationPreparedMaterialData Material { get; }
    public float ResolveFadeDistance() =>
        float.IsFinite(MaximumDistance) && MaximumDistance > 0.0f
            ? MaximumDistance
            : 0.0f;
    public bool IsValid =>
        VertexBuffer.IsValid &&
        IndexBuffer.IsValid &&
        IndexCount > 0 &&
        LodLevel >= 0 &&
        float.IsFinite(MaximumDistance) &&
        MaximumDistance >= 0.0f &&
        float.IsFinite(MaximumScreenError) &&
        MaximumScreenError >= 0.0f &&
        float.IsFinite(WindStiffness) &&
        WindStiffness >= 0.0f &&
        WindStiffness <= 1.0f &&
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
