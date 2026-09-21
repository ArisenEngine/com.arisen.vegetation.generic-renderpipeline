using Arisen.Native.RHI;
using ArisenEngine.Core.Assets;
using ArisenEngine.Core.RHI;
using ArisenEngine.Rendering;
using ArisenEngine.Rendering.Resources;
using ArisenEngine.Resources.Serialization;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading;

namespace ArisenEngine.Vegetation.GenericRenderPipeline;

[StructLayout(LayoutKind.Sequential)]
internal readonly struct VegetationOpaqueFrameConstants
{
    public const int ByteSize = 144;

    /// <summary>
    /// View-projection of the view frame: the render camera sits at the origin, so this matrix is
    /// pure rotation plus projection and carries no world-origin translation.
    /// </summary>
    public readonly Vector4 ViewProjectionColumn0;
    public readonly Vector4 ViewProjectionColumn1;
    public readonly Vector4 ViewProjectionColumn2;
    public readonly Vector4 ViewProjectionColumn3;
    /// <summary>
    /// Render-camera position expressed in this frame, i.e. the origin of every vertex position the
    /// vegetation shaders read. Both vegetation passes render view-frame geometry, so they use this
    /// record to measure camera distance and to resolve the view direction and shadow lookup
    /// position without ever referring to the rebaseable world origin.
    /// </summary>
    public readonly Vector4 CameraPosition;
    public readonly Vector4 LightDirectionIntensity;
    public readonly Vector4 LightColorAmbient;
    public readonly Vector4 WindDirectionStrength;
    public readonly Vector4 WindGustParameters;

    private VegetationOpaqueFrameConstants(
        Vector4 viewProjectionColumn0,
        Vector4 viewProjectionColumn1,
        Vector4 viewProjectionColumn2,
        Vector4 viewProjectionColumn3,
        Vector4 cameraPosition,
        Vector4 lightDirectionIntensity,
        Vector4 lightColorAmbient,
        Vector4 windDirectionStrength,
        Vector4 windGustParameters)
    {
        ViewProjectionColumn0 = viewProjectionColumn0;
        ViewProjectionColumn1 = viewProjectionColumn1;
        ViewProjectionColumn2 = viewProjectionColumn2;
        ViewProjectionColumn3 = viewProjectionColumn3;
        CameraPosition = cameraPosition;
        LightDirectionIntensity = lightDirectionIntensity;
        LightColorAmbient = lightColorAmbient;
        WindDirectionStrength = windDirectionStrength;
        WindGustParameters = windGustParameters;
    }

    public static VegetationOpaqueFrameConstants Create(
        Matrix4x4 viewRelativeViewProjection,
        Vector3 cameraPositionInFrame,
        DirectionalLight directionalLight,
        SceneEnvironment environment,
        in VegetationWindSettings wind,
        WorldPosition frameAnchor)
    {
        if (!IsFinite(cameraPositionInFrame))
        {
            throw new ArgumentException(
                "[Vegetation.GenericRP] In-frame camera position is not finite.",
                nameof(cameraPositionInFrame));
        }
        if (!wind.IsValid)
        {
            throw new ArgumentException(
                "[Vegetation.GenericRP] Wind settings are outside their bounded range.",
                nameof(wind));
        }

        DirectionalLight light = directionalLight.IsValid
            ? directionalLight
            : DirectionalLight.Default;
        SceneEnvironment sceneEnvironment = environment.IsValid
            ? environment
            : SceneEnvironment.Default;
        float ambient = MathF.Max(
            MathF.Max(0.0f, light.AmbientIntensity),
            MathF.Max(0.0f, sceneEnvironment.AmbientIntensity));
        (float primaryGustPhase, float secondaryGustPhase) =
            wind.ResolveGustPhases(frameAnchor);
        return new VegetationOpaqueFrameConstants(
            new Vector4(
                viewRelativeViewProjection.M11,
                viewRelativeViewProjection.M21,
                viewRelativeViewProjection.M31,
                viewRelativeViewProjection.M41),
            new Vector4(
                viewRelativeViewProjection.M12,
                viewRelativeViewProjection.M22,
                viewRelativeViewProjection.M32,
                viewRelativeViewProjection.M42),
            new Vector4(
                viewRelativeViewProjection.M13,
                viewRelativeViewProjection.M23,
                viewRelativeViewProjection.M33,
                viewRelativeViewProjection.M43),
            new Vector4(
                viewRelativeViewProjection.M14,
                viewRelativeViewProjection.M24,
                viewRelativeViewProjection.M34,
                viewRelativeViewProjection.M44),
            new Vector4(cameraPositionInFrame, 1.0f),
            new Vector4(light.Direction, light.Intensity),
            new Vector4(light.Color, ambient),
            new Vector4(
                wind.Direction.X,
                0.0f,
                wind.Direction.Z,
                wind.Strength),
            new Vector4(
                wind.GustAmplitude,
                primaryGustPhase,
                secondaryGustPhase,
                wind.BendStrength));
    }

    private static bool IsFinite(Vector3 value) =>
        float.IsFinite(value.X) &&
        float.IsFinite(value.Y) &&
        float.IsFinite(value.Z);
}

[StructLayout(LayoutKind.Sequential)]
internal readonly struct VegetationOpaqueDrawConstants
{
    private const uint InvalidBindlessIndex = 0xFFFFFFFFu;
    public const int ByteSize = 112;

    public readonly Vector4 ClusterOriginInstanceBuffer;
    public readonly Vector4 BaseColorFactor;
    public readonly Vector4 MaterialParameters;
    public readonly Vector4 NormalParameters;
    public readonly Vector4 OrmParameters;
    public readonly Vector4 FrameShadowParameters;
    public readonly Vector4 WindParameters;

    private VegetationOpaqueDrawConstants(
        Vector4 clusterOriginInstanceBuffer,
        Vector4 baseColorFactor,
        Vector4 materialParameters,
        Vector4 normalParameters,
        Vector4 ormParameters,
        Vector4 frameShadowParameters,
        Vector4 windParameters)
    {
        ClusterOriginInstanceBuffer = clusterOriginInstanceBuffer;
        BaseColorFactor = baseColorFactor;
        MaterialParameters = materialParameters;
        NormalParameters = normalParameters;
        OrmParameters = ormParameters;
        FrameShadowParameters = frameShadowParameters;
        WindParameters = windParameters;
    }

    public uint InstanceBufferIndex =>
        BitConverter.SingleToUInt32Bits(ClusterOriginInstanceBuffer.W);
    public uint FrameBufferIndex =>
        BitConverter.SingleToUInt32Bits(FrameShadowParameters.X);
    public uint ShadowBufferIndex =>
        BitConverter.SingleToUInt32Bits(FrameShadowParameters.Y);

    public static VegetationOpaqueDrawConstants Create(
        Vector3 viewRelativeClusterOrigin,
        in VegetationPreparedBatch batch,
        uint frameBufferIndex,
        bool receiveShadows,
        in DirectionalShadowFrameData directionalShadow)
    {
        if (!batch.IsValid)
        {
            throw new ArgumentException(
                "[Vegetation.GenericRP] Cannot prepare an invalid opaque batch.",
                nameof(batch));
        }
        if (frameBufferIndex == InvalidBindlessIndex)
        {
            throw new ArgumentOutOfRangeException(
                nameof(frameBufferIndex),
                "[Vegetation.GenericRP] Opaque frame buffer is not bindless-addressable.");
        }

        uint shadowBufferIndex = receiveShadows && directionalShadow.Enabled
            ? directionalShadow.ConstantsBufferBindlessIndex
            : InvalidBindlessIndex;
        VegetationPreparedMaterialData material = batch.Material;

        return new VegetationOpaqueDrawConstants(
            new Vector4(
                viewRelativeClusterOrigin,
                BitConverter.UInt32BitsToSingle(batch.InstanceBufferIndex)),
            new Vector4(
                material.BaseColorFactor.X,
                material.BaseColorFactor.Y,
                material.BaseColorFactor.Z,
                material.AlphaCutoff),
            new Vector4(
                material.MetallicFactor,
                material.RoughnessFactor,
                BitConverter.UInt32BitsToSingle(material.BaseColorImageIndex),
                BitConverter.UInt32BitsToSingle(material.BaseColorSamplerIndex)),
            new Vector4(
                BitConverter.UInt32BitsToSingle(material.NormalImageIndex),
                BitConverter.UInt32BitsToSingle(material.NormalSamplerIndex),
                material.TintVariation,
                material.BaseColorFactor.W),
            new Vector4(
                BitConverter.UInt32BitsToSingle(material.OrmImageIndex),
                BitConverter.UInt32BitsToSingle(material.OrmSamplerIndex),
                material.OcclusionStrength,
                0.0f),
            new Vector4(
                BitConverter.UInt32BitsToSingle(frameBufferIndex),
                BitConverter.UInt32BitsToSingle(shadowBufferIndex),
                BitConverter.UInt32BitsToSingle(material.Flags),
                0.0f),
            new Vector4(
                batch.WindStiffness,
                batch.ResolveFadeDistance(),
                0.0f,
                0.0f));
    }
}

internal readonly struct VegetationOpaquePreparedDraw
{
    public VegetationOpaquePreparedDraw(
        in VegetationPreparedBatch batch,
        in VegetationOpaqueDrawConstants constants)
    {
        if (!batch.IsValid)
        {
            throw new ArgumentException(
                "[Vegetation.GenericRP] Opaque draw batch is invalid.",
                nameof(batch));
        }
        if (constants.InstanceBufferIndex != batch.InstanceBufferIndex)
        {
            throw new ArgumentException(
                "[Vegetation.GenericRP] Opaque draw constants reference a different instance buffer.",
                nameof(constants));
        }
        if (constants.FrameBufferIndex == uint.MaxValue)
        {
            throw new ArgumentException(
                "[Vegetation.GenericRP] Opaque draw constants reference an invalid frame buffer.",
                nameof(constants));
        }

        VertexBuffer = batch.VertexBuffer;
        IndexBuffer = batch.IndexBuffer;
        IndexType = batch.IndexType;
        IndexCount = batch.IndexCount;
        FirstIndex = batch.FirstIndex;
        VertexOffset = batch.VertexOffset;
        FirstInstance = batch.FirstInstance;
        InstanceCount = batch.InstanceCount;
        Constants = constants;
    }

    public RHIBufferHandle VertexBuffer { get; }
    public RHIBufferHandle IndexBuffer { get; }
    public EIndexType IndexType { get; }
    public uint IndexCount { get; }
    public uint FirstIndex { get; }
    public int VertexOffset { get; }
    public uint FirstInstance { get; }
    public uint InstanceCount { get; }
    public VegetationOpaqueDrawConstants Constants { get; }
    public bool IsValid =>
        VertexBuffer.IsValid &&
        IndexBuffer.IsValid &&
        IndexCount > 0 &&
        InstanceCount > 0;
}

internal sealed class VegetationOpaquePass : RenderPassNode
{
    private const ulong DynamicViewportScissorMask = 0x1UL | 0x2UL;
    private const string VertexStage = "Vertex";
    private const string FragmentStage = "Fragment";
    private const int DrawsPerWorkItem = 256;
    private const int DefaultFrameBufferRingSize = 2;
    private const uint InvalidBindlessIndex = 0xFFFFFFFFu;
    private const int PipelineCleanupLeg = 0;
    private const int PipelineStateCleanupLeg = 1;
    private const int VertexProgramCleanupLeg = 2;
    private const int FragmentProgramCleanupLeg = 3;
    private const int VertexShaderAssetCleanupLeg = 4;
    private const int FragmentShaderAssetCleanupLeg = 5;

    private readonly IAssetDatabase m_AssetDatabase;
    private readonly ShaderAsset m_Shader;
    private VegetationOpaquePreparedDraw[] m_Draws =
        Array.Empty<VegetationOpaquePreparedDraw>();
    private int m_DrawCount;
    private int m_LastRecordedBatchCount;
    private long m_LastRecordedInstanceCount;
    private RHIFactory m_Factory;
    private RHIDevice m_Device;
    private RHIPipelineCache? m_PipelineCache;
    private RHIPipelineState m_PipelineState;
    private RHIPipelineHandle m_Pipeline = RHIPipelineHandle.Invalid;
    private RHIShaderProgramHandle m_VertexProgram = RHIShaderProgramHandle.Invalid;
    private RHIShaderProgramHandle m_FragmentProgram = RHIShaderProgramHandle.Invalid;
    private CookedAssetHandle m_VertexShaderAsset = CookedAssetHandle.Invalid;
    private CookedAssetHandle m_FragmentShaderAsset = CookedAssetHandle.Invalid;
    private AssetDependencyStamp m_ShaderStamp = AssetDependencyStamp.Empty;
    private RHIImageViewHandle m_ColorTarget = RHIImageViewHandle.Invalid;
    private RHIImageViewHandle m_DepthTarget = RHIImageViewHandle.Invalid;
    private EFormat m_ColorFormat = EFormat.FORMAT_UNDEFINED;
    private EFormat m_DepthFormat = EFormat.FORMAT_UNDEFINED;
    private VegetationOpaqueFrameBufferSlot[] m_FrameBufferSlots =
        Array.Empty<VegetationOpaqueFrameBufferSlot>();
    private int m_FrameBufferRingSize;
    private RHIBufferHandle m_FrameConstantsBuffer = RHIBufferHandle.Invalid;
    private uint m_FrameConstantsBufferBindlessIndex = InvalidBindlessIndex;
    private VegetationPassCleanupJournal m_PipelineCleanup;

    public VegetationOpaquePass(IAssetDatabase assetDatabase)
        : base("VegetationOpaquePass")
    {
        m_AssetDatabase = assetDatabase ?? throw new ArgumentNullException(nameof(assetDatabase));
        m_Shader = VegetationGenericRenderPipelineShaderAssets.CreateVegetation();
    }

    public int LastRecordedBatchCount => Volatile.Read(ref m_LastRecordedBatchCount);
    public long LastRecordedInstanceCount => Volatile.Read(ref m_LastRecordedInstanceCount);

    /// <summary>
    /// Frame-global vegetation constants written by <see cref="PrepareFrame"/>. The vegetation
    /// shadow pass reads the same records to reproduce the opaque displacement and fade exactly,
    /// so it needs the handle to publish its own host-write to shader-read dependency.
    /// </summary>
    public RHIBufferHandle FrameConstantsBuffer => m_FrameConstantsBuffer;

    public static void DeclareGraphAccess(
        RenderGraphBuilder builder,
        RenderResource directionalShadow,
        RenderResource sceneColor,
        RenderResource frameDepth)
    {
        ArgumentNullException.ThrowIfNull(builder);
        builder
            .ReadShader(directionalShadow)
            .ReadWriteColorAttachment(
                sceneColor,
                RenderAttachmentIntent.LoadStore)
            .ReadWriteDepthAttachment(
                frameDepth,
                RenderAttachmentIntent.LoadStore);
    }

    public void SetPreparedDraws(
        VegetationOpaquePreparedDraw[] draws,
        int drawCount)
    {
        ArgumentNullException.ThrowIfNull(draws);
        if ((uint)drawCount > (uint)draws.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(drawCount));
        }

        m_Draws = draws;
        m_DrawCount = drawCount;
        Volatile.Write(ref m_LastRecordedBatchCount, 0);
        Volatile.Write(ref m_LastRecordedInstanceCount, 0);
    }

    public void SetTargets(
        RHIImageViewHandle colorTarget,
        RHIImageViewHandle depthTarget)
    {
        m_ColorTarget = colorTarget;
        m_DepthTarget = depthTarget;
    }

    public void Prepare(RenderContext context)
    {
        Volatile.Write(ref m_LastRecordedBatchCount, 0);
        Volatile.Write(ref m_LastRecordedInstanceCount, 0);
        const EFormat colorFormat = EFormat.FORMAT_R16G16B16A16_SFLOAT;
        const EFormat depthFormat = EFormat.FORMAT_D32_SFLOAT;
        AssetDependencyStamp shaderStamp =
            AssetDependencyTracker.GetShaderStamp(m_AssetDatabase, m_Shader);
        if (m_Pipeline.IsValid &&
            m_Device.Handle == context.Device.Handle &&
            m_ColorFormat == colorFormat &&
            m_DepthFormat == depthFormat &&
            m_ShaderStamp == shaderStamp)
        {
            return;
        }

        bool sameDevice = m_Device.IsValid &&
            m_Device.Handle == context.Device.Handle;
        if (sameDevice)
        {
            ReleasePipelineResources();
        }
        else
        {
            ReleaseDeviceResources();
        }
        m_PipelineCleanup.BeginOwnership();
        m_Device = context.Device;
        m_Factory = context.Device.GetFactory();
        RHIPipelineCache pipelineCache = context.Device.PipelineCache;
        m_PipelineCache = pipelineCache;
        try
        {
            m_VertexProgram = CompileProgram(
                EShaderStage.SHADER_STAGE_VERTEX_BIT,
                VertexStage,
                out m_VertexShaderAsset);
            m_FragmentProgram = CompileProgram(
                EShaderStage.SHADER_STAGE_FRAGMENT_BIT,
                FragmentStage,
                out m_FragmentShaderAsset);

            m_PipelineState = pipelineCache.GetPipelineState();
            m_PipelineState.AddProgram(m_VertexProgram);
            m_PipelineState.AddProgram(m_FragmentProgram);
            m_PipelineState.SetBindPoint(
                EPipelineBindPoint.PIPELINE_BIND_POINT_GRAPHICS);
            m_PipelineState.SetInputAssemblyState(
                EPrimitiveTopology.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
            AddStaticMeshVertexLayout(m_PipelineState);
            m_PipelineState.SetRasterizationState(
                EPolygonMode.EPOLYGON_MODE_FILL,
                ECullModeFlagBits.CULL_MODE_NONE,
                EFrontFace.FRONT_FACE_COUNTER_CLOCKWISE);
            m_PipelineState.SetColorBlendState(false);
            m_PipelineState.SetDepthStencilState(
                true,
                true,
                ECompareOp.COMPARE_OP_LESS_OR_EQUAL);
            m_PipelineState.SetDynamicStateMask(DynamicViewportScissorMask);
            m_PipelineState.SetRenderingFormats([colorFormat], depthFormat);
            m_PipelineState.BuildDescriptorSetLayout();
            m_Pipeline = pipelineCache.GetGraphicsPipeline(m_PipelineState);
            if (!m_Pipeline.IsValid)
            {
                throw new InvalidOperationException(
                    "[Vegetation.GenericRP] Failed to create vegetation opaque pipeline.");
            }

            m_ColorFormat = colorFormat;
            m_DepthFormat = depthFormat;
            m_ShaderStamp = shaderStamp;
        }
        catch (Exception preparationFailure)
        {
            try
            {
                ReleasePipelineResources();
            }
            catch (Exception cleanupFailure)
            {
                throw new AggregateException(
                    "Vegetation opaque pipeline preparation and rollback both failed.",
                    preparationFailure,
                    cleanupFailure);
            }

            throw;
        }
    }

    public unsafe uint PrepareFrame(
        RenderContext context,
        Matrix4x4 viewRelativeViewProjection,
        Vector3 cameraPositionInFrame,
        DirectionalLight directionalLight,
        SceneEnvironment environment,
        in VegetationWindSettings wind,
        WorldPosition frameAnchor)
    {
        if (sizeof(VegetationOpaqueFrameConstants) !=
                VegetationOpaqueFrameConstants.ByteSize ||
            sizeof(VegetationOpaqueDrawConstants) !=
                VegetationOpaqueDrawConstants.ByteSize)
        {
            throw new InvalidOperationException(
                "[Vegetation.GenericRP] Opaque CPU shader constants layout is invalid.");
        }
        if (!m_Pipeline.IsValid ||
            !m_Factory.IsValid ||
            m_Device.Handle != context.Device.Handle)
        {
            throw new InvalidOperationException(
                "[Vegetation.GenericRP] Opaque pipeline must be prepared before frame data.");
        }

        int ringSize = GetFrameBufferRingSize(context);
        EnsureFrameBufferRing(ringSize);
        int slotIndex = checked((int)(context.FrameResourceIndex % (uint)ringSize));
        ref VegetationOpaqueFrameBufferSlot slot = ref m_FrameBufferSlots[slotIndex];
        EnsureFrameBufferSlot(ref slot, slotIndex);

        VegetationOpaqueFrameConstants constants =
            VegetationOpaqueFrameConstants.Create(
                viewRelativeViewProjection,
                cameraPositionInFrame,
                directionalLight,
                environment,
                wind,
                frameAnchor);
        IntPtr mapped = m_Factory.MapBuffer(slot.Buffer);
        if (mapped == IntPtr.Zero)
        {
            throw new InvalidOperationException(
                "[Vegetation.GenericRP] Failed to map opaque frame constants buffer.");
        }

        try
        {
            *(VegetationOpaqueFrameConstants*)mapped.ToPointer() = constants;
        }
        finally
        {
            m_Factory.UnmapBuffer(slot.Buffer);
        }

        m_FrameConstantsBuffer = slot.Buffer;
        m_FrameConstantsBufferBindlessIndex = slot.BindlessIndex;
        return slot.BindlessIndex;
    }

    protected override int GetWorkItemCount(RenderContext context) =>
        m_DrawCount <= 0
            ? 0
            : checked((m_DrawCount + DrawsPerWorkItem - 1) / DrawsPerWorkItem);

    protected override RenderPassWorkItem GetWorkItem(
        RenderContext context,
        int workItemIndex)
    {
        int drawStart = checked(workItemIndex * DrawsPerWorkItem);
        if (drawStart < 0 || drawStart >= m_DrawCount)
        {
            return RenderPassWorkItem.Pass(workItemIndex);
        }

        int drawCount = Math.Min(DrawsPerWorkItem, m_DrawCount - drawStart);
        return RenderPassWorkItem.DrawRange(workItemIndex, drawStart, drawCount);
    }

    protected override void Record(
        RenderContext context,
        RenderCommandList commandList)
    {
        Volatile.Write(ref m_LastRecordedBatchCount, 0);
        Volatile.Write(ref m_LastRecordedInstanceCount, 0);
        RecordDrawRange(context, commandList, 0, m_DrawCount);
    }

    protected override void Record(
        RenderContext context,
        RenderCommandList commandList,
        RenderPassWorkItem workItem)
    {
        if (workItem.HasDrawRange)
        {
            RecordDrawRange(
                context,
                commandList,
                workItem.DrawStart,
                workItem.DrawCount);
        }
    }

    private void RecordDrawRange(
        RenderContext context,
        RenderCommandList commandList,
        int drawStart,
        int drawCount)
    {
        if (!m_Pipeline.IsValid ||
            !m_ColorTarget.IsValid ||
            !m_DepthTarget.IsValid ||
            !m_FrameConstantsBuffer.IsValid ||
            m_FrameConstantsBufferBindlessIndex == InvalidBindlessIndex ||
            drawStart < 0 ||
            drawCount <= 0 ||
            drawStart >= m_DrawCount)
        {
            return;
        }

        int drawEnd = Math.Min(m_DrawCount, checked(drawStart + drawCount));
        RecordFrameDataBarrier(commandList);
        commandList.BeginRendering(
            m_ColorTarget,
            EImageLayout.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            EAttachmentLoadOp.ATTACHMENT_LOAD_OP_LOAD,
            EAttachmentStoreOp.ATTACHMENT_STORE_OP_STORE,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            m_DepthTarget,
            EImageLayout.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            EAttachmentLoadOp.ATTACHMENT_LOAD_OP_LOAD,
            EAttachmentStoreOp.ATTACHMENT_STORE_OP_STORE,
            1.0f,
            0,
            0,
            0,
            context.Width,
            context.Height);
        commandList.BindPipeline(m_Pipeline);
        commandList.SetViewport(0, 0, context.Width, context.Height);
        commandList.SetScissor(0, 0, context.Width, context.Height);

        int recordedBatches = 0;
        long recordedInstances = 0;
        for (int drawIndex = drawStart; drawIndex < drawEnd; drawIndex++)
        {
            ref readonly VegetationOpaquePreparedDraw draw =
                ref m_Draws[drawIndex];
            if (!draw.IsValid)
            {
                continue;
            }

            commandList.PushConstants(
                draw.Constants,
                EShaderStage.SHADER_STAGE_VERTEX_BIT |
                EShaderStage.SHADER_STAGE_FRAGMENT_BIT);
            commandList.BindVertexBuffers(draw.VertexBuffer);
            commandList.BindIndexBuffer(
                draw.IndexBuffer,
                0,
                draw.IndexType);
            commandList.DrawIndexed(
                draw.IndexCount,
                instanceCount: draw.InstanceCount,
                firstIndex: draw.FirstIndex,
                vertexOffset: draw.VertexOffset,
                firstInstance: draw.FirstInstance);
            recordedBatches++;
            recordedInstances += draw.InstanceCount;
        }

        commandList.EndRendering();
        Interlocked.Add(ref m_LastRecordedBatchCount, recordedBatches);
        Interlocked.Add(ref m_LastRecordedInstanceCount, recordedInstances);
    }

    public void ReleaseDeviceResources()
    {
        ReleasePipelineResources();
        ReleaseFrameBuffers();
        m_ColorTarget = RHIImageViewHandle.Invalid;
        m_DepthTarget = RHIImageViewHandle.Invalid;
        Volatile.Write(ref m_LastRecordedBatchCount, 0);
        Volatile.Write(ref m_LastRecordedInstanceCount, 0);
        m_Factory = default;
        m_Device = default;
    }

    private void RecordFrameDataBarrier(RenderCommandList commandList)
    {
        if (!m_FrameConstantsBuffer.IsValid ||
            m_FrameConstantsBufferBindlessIndex == InvalidBindlessIndex)
        {
            return;
        }

        Span<RHIBufferMemoryBarrier> barriers = stackalloc RHIBufferMemoryBarrier[1];
        barriers[0] = new RHIBufferMemoryBarrier
        {
            SrcAccessMask = EAccessFlag.ACCESS_HOST_WRITE_BIT,
            DstAccessMask = EAccessFlag.ACCESS_SHADER_READ_BIT,
            SrcQueueFamilyIndex = RHIQueueFamily.Ignored,
            DstQueueFamilyIndex = RHIQueueFamily.Ignored,
            Buffer = m_FrameConstantsBuffer,
            SrcStageMask = EPipelineStageFlagBits.PIPELINE_STAGE_HOST_BIT,
            DstStageMask = EPipelineStageFlagBits.PIPELINE_STAGE_VERTEX_SHADER_BIT |
                           EPipelineStageFlagBits.PIPELINE_STAGE_FRAGMENT_SHADER_BIT
        };
        commandList.PipelineBarrier(
            EPipelineStageFlagBits.PIPELINE_STAGE_HOST_BIT,
            EPipelineStageFlagBits.PIPELINE_STAGE_VERTEX_SHADER_BIT |
            EPipelineStageFlagBits.PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            barriers);
    }

    private static int GetFrameBufferRingSize(RenderContext context)
    {
        uint maxFramesInFlight = context.Device.GetInstance().MaxFramesInFlight;
        return maxFramesInFlight == 0
            ? DefaultFrameBufferRingSize
            : checked((int)Math.Max(1u, maxFramesInFlight));
    }

    private void EnsureFrameBufferRing(int ringSize)
    {
        if (m_FrameBufferRingSize == ringSize &&
            m_FrameBufferSlots.Length == ringSize)
        {
            return;
        }
        if (m_FrameBufferSlots.Length != 0)
        {
            throw new InvalidOperationException(
                "[Vegetation.GenericRP] Cannot resize an active opaque frame buffer ring.");
        }

        m_FrameBufferSlots = new VegetationOpaqueFrameBufferSlot[ringSize];
        m_FrameBufferRingSize = ringSize;
    }

    private void EnsureFrameBufferSlot(
        ref VegetationOpaqueFrameBufferSlot slot,
        int slotIndex)
    {
        if (slot.IsValid)
        {
            return;
        }
        if (slot.Buffer.IsValid)
        {
            ReleaseFrameBufferSlot(ref slot);
        }

        RHIBufferHandle buffer = m_Factory.CreateBuffer(
            VegetationOpaqueFrameConstants.ByteSize,
            (uint)EBufferUsageFlagBits.BUFFER_USAGE_STORAGE_BUFFER_BIT,
            ESharingMode.SHARING_MODE_EXCLUSIVE,
            ERHIMemoryUsage.Upload,
            $"Vegetation.OpaqueFrameData[{slotIndex}]");
        if (!buffer.IsValid)
        {
            throw new InvalidOperationException(
                "[Vegetation.GenericRP] Failed to create opaque frame constants buffer.");
        }

        slot = new VegetationOpaqueFrameBufferSlot(buffer, InvalidBindlessIndex);
        try
        {
            uint bindlessIndex = m_Factory.RegisterBindlessResourceBuffer(buffer);
            if (bindlessIndex == InvalidBindlessIndex)
            {
                throw new InvalidOperationException(
                    "[Vegetation.GenericRP] Failed to register opaque frame constants buffer.");
            }

            slot.BindlessIndex = bindlessIndex;
        }
        catch (Exception registrationFailure)
        {
            try
            {
                ReleaseFrameBufferSlot(ref slot);
            }
            catch (Exception releaseFailure)
            {
                throw new AggregateException(
                    "Opaque frame-buffer registration and rollback both failed.",
                    registrationFailure,
                    releaseFailure);
            }

            throw;
        }
    }

    private void ReleaseFrameBuffers()
    {
        for (int slotIndex = 0;
             slotIndex < m_FrameBufferSlots.Length;
             slotIndex++)
        {
            ReleaseFrameBufferSlot(ref m_FrameBufferSlots[slotIndex]);
        }

        m_FrameBufferSlots = Array.Empty<VegetationOpaqueFrameBufferSlot>();
        m_FrameBufferRingSize = 0;
        m_FrameConstantsBuffer = RHIBufferHandle.Invalid;
        m_FrameConstantsBufferBindlessIndex = InvalidBindlessIndex;
    }

    private void ReleaseFrameBufferSlot(ref VegetationOpaqueFrameBufferSlot slot)
    {
        if (!slot.Buffer.IsValid)
        {
            slot = default;
            return;
        }
        if (!m_Factory.IsValid)
        {
            throw new InvalidOperationException(
                "[Vegetation.GenericRP] Cannot release opaque frame data without its factory.");
        }

        if (slot.BindlessIndex != InvalidBindlessIndex)
        {
            m_Factory.UnregisterBindlessResourceBuffer(slot.BindlessIndex);
            slot.BindlessIndex = InvalidBindlessIndex;
        }

        m_Factory.ReleaseBuffer(slot.Buffer);
        slot.Buffer = RHIBufferHandle.Invalid;
    }

    private void ReleasePipelineResources()
    {
        m_PipelineCleanup.Release(PipelineCleanupLeg, ReleasePipeline);
        m_PipelineCleanup.Release(PipelineStateCleanupLeg, ReleasePipelineState);
        m_PipelineCleanup.Release(VertexProgramCleanupLeg, ReleaseVertexProgram);
        m_PipelineCleanup.Release(FragmentProgramCleanupLeg, ReleaseFragmentProgram);
        m_PipelineCleanup.Release(VertexShaderAssetCleanupLeg, ReleaseVertexShaderAsset);
        m_PipelineCleanup.Release(FragmentShaderAssetCleanupLeg, ReleaseFragmentShaderAsset);

        m_PipelineCache = null;
        m_ShaderStamp = AssetDependencyStamp.Empty;
        m_ColorFormat = EFormat.FORMAT_UNDEFINED;
        m_DepthFormat = EFormat.FORMAT_UNDEFINED;
    }

    private void ReleasePipeline()
    {
        if (!m_Pipeline.IsValid)
        {
            return;
        }

        RHIPipelineCache pipelineCache = m_PipelineCache ??
            throw new InvalidOperationException(
                "[Vegetation.GenericRP] Cannot release the opaque pipeline without its cache.");
        pipelineCache.ReleasePipeline(m_Pipeline);
        m_Pipeline = RHIPipelineHandle.Invalid;
    }

    private void ReleasePipelineState()
    {
        if (!m_PipelineState.IsValid)
        {
            return;
        }

        m_PipelineState.Release();
        m_PipelineState = default;
    }

    private void ReleaseVertexProgram()
    {
        if (!m_VertexProgram.IsValid)
        {
            return;
        }
        if (!m_Factory.IsValid)
        {
            throw new InvalidOperationException(
                "[Vegetation.GenericRP] Cannot release the opaque vertex program without its factory.");
        }

        m_Factory.ReleaseGPUProgram(m_VertexProgram);
        m_VertexProgram = RHIShaderProgramHandle.Invalid;
    }

    private void ReleaseFragmentProgram()
    {
        if (!m_FragmentProgram.IsValid)
        {
            return;
        }
        if (!m_Factory.IsValid)
        {
            throw new InvalidOperationException(
                "[Vegetation.GenericRP] Cannot release the opaque fragment program without its factory.");
        }

        m_Factory.ReleaseGPUProgram(m_FragmentProgram);
        m_FragmentProgram = RHIShaderProgramHandle.Invalid;
    }

    private void ReleaseVertexShaderAsset()
    {
        if (!m_VertexShaderAsset.IsValid)
        {
            return;
        }

        m_AssetDatabase.Release(m_VertexShaderAsset);
        m_VertexShaderAsset = CookedAssetHandle.Invalid;
    }

    private void ReleaseFragmentShaderAsset()
    {
        if (!m_FragmentShaderAsset.IsValid)
        {
            return;
        }

        m_AssetDatabase.Release(m_FragmentShaderAsset);
        m_FragmentShaderAsset = CookedAssetHandle.Invalid;
    }

    private RHIShaderProgramHandle CompileProgram(
        EShaderStage rhiStage,
        string stageName,
        out CookedAssetHandle shaderAssetHandle)
    {
        shaderAssetHandle = CookedAssetHandle.Invalid;
        RHIShaderProgramHandle program = RHIShaderProgramHandle.Invalid;
        try
        {
            CookedShaderStage cookedStage = ShaderAssetCooker.LoadOrCookStage(
                m_AssetDatabase,
                m_Shader,
                stageName);
            shaderAssetHandle = cookedStage.Handle;
            ReadOnlyMemory<byte> shaderCode =
                m_AssetDatabase.GetCookedAssetBytes(shaderAssetHandle);
            program = m_Factory.CreateGPUProgram();
            if (!program.IsValid ||
                !m_Factory.AttachProgramByteCode(
                    program,
                    rhiStage,
                    shaderCode,
                    cookedStage.Stage.EntryPoint))
            {
                throw new InvalidOperationException(
                    $"[Vegetation.GenericRP] Failed to prepare shader stage '{stageName}'.");
            }

            return program;
        }
        catch
        {
            if (program.IsValid)
            {
                m_Factory.ReleaseGPUProgram(program);
            }
            if (shaderAssetHandle.IsValid)
            {
                m_AssetDatabase.Release(shaderAssetHandle);
                shaderAssetHandle = CookedAssetHandle.Invalid;
            }
            throw;
        }
    }

    private static void AddStaticMeshVertexLayout(RHIPipelineState state)
    {
        state.ClearVertexInputDescriptions();
        state.AddVertexBindingDescription(
            0,
            MeshAssetCooker.StaticMeshVertexStride,
            EVertexInputRate.VERTEX_INPUT_RATE_VERTEX);
        state.AddVertexInputAttributeDescription(
            0,
            0,
            EFormat.FORMAT_R32G32B32_SFLOAT,
            0);
        state.AddVertexInputAttributeDescription(
            1,
            0,
            EFormat.FORMAT_R32G32B32_SFLOAT,
            12);
        state.AddVertexInputAttributeDescription(
            2,
            0,
            EFormat.FORMAT_R32G32B32A32_SFLOAT,
            24);
        state.AddVertexInputAttributeDescription(
            3,
            0,
            EFormat.FORMAT_R32G32_SFLOAT,
            40);
        state.AddVertexInputAttributeDescription(
            4,
            0,
            EFormat.FORMAT_R32G32B32_SFLOAT,
            48);
    }

    private struct VegetationOpaqueFrameBufferSlot
    {
        public VegetationOpaqueFrameBufferSlot(
            RHIBufferHandle buffer,
            uint bindlessIndex)
        {
            Buffer = buffer;
            BindlessIndex = bindlessIndex;
        }

        public RHIBufferHandle Buffer;
        public uint BindlessIndex;
        public readonly bool IsValid =>
            Buffer.IsValid && BindlessIndex != InvalidBindlessIndex;
    }
}
