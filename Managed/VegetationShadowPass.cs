using Arisen.Native.RHI;
using ArisenEngine.Core.Assets;
using ArisenEngine.Core.RHI;
using ArisenEngine.Rendering;
using ArisenEngine.Rendering.Resources;
using ArisenEngine.Resources.Serialization;
using ArisenEngine.Vegetation.Assets;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading;

namespace ArisenEngine.Vegetation.GenericRenderPipeline;

[StructLayout(LayoutKind.Sequential)]
internal readonly struct VegetationShadowDrawConstants
{
    public const int ByteSize = 112;

    public readonly Vector4 ViewProjectionColumn0;
    public readonly Vector4 ViewProjectionColumn1;
    public readonly Vector4 ViewProjectionColumn2;
    public readonly Vector4 ViewProjectionColumn3;
    public readonly Vector4 ClusterOriginInstanceBuffer;
    public readonly Vector4 MaterialParameters;
    public readonly Vector4 MaterialFlags;

    private VegetationShadowDrawConstants(
        Vector4 viewProjectionColumn0,
        Vector4 viewProjectionColumn1,
        Vector4 viewProjectionColumn2,
        Vector4 viewProjectionColumn3,
        Vector4 clusterOriginInstanceBuffer,
        Vector4 materialParameters,
        Vector4 materialFlags)
    {
        ViewProjectionColumn0 = viewProjectionColumn0;
        ViewProjectionColumn1 = viewProjectionColumn1;
        ViewProjectionColumn2 = viewProjectionColumn2;
        ViewProjectionColumn3 = viewProjectionColumn3;
        ClusterOriginInstanceBuffer = clusterOriginInstanceBuffer;
        MaterialParameters = materialParameters;
        MaterialFlags = materialFlags;
    }

    public uint InstanceBufferIndex =>
        BitConverter.SingleToUInt32Bits(ClusterOriginInstanceBuffer.W);

    public uint BaseColorImageIndex =>
        BitConverter.SingleToUInt32Bits(MaterialParameters.Y);

    public uint BaseColorSamplerIndex =>
        BitConverter.SingleToUInt32Bits(MaterialParameters.Z);

    public uint MaterialFlagBits =>
        BitConverter.SingleToUInt32Bits(MaterialFlags.X);

    public float WindStiffness => MaterialFlags.Y;

    public float FadeDistance => MaterialFlags.Z;

    public uint FrameBufferIndex =>
        BitConverter.SingleToUInt32Bits(MaterialFlags.W);

    public static VegetationShadowDrawConstants Create(
        in DirectionalShadowCascade cascade,
        Vector3 viewRelativeClusterOrigin,
        in VegetationPreparedBatch batch,
        uint frameBufferIndex)
    {
        if (!batch.IsValid || batch.ShadowPolicy != VegetationShadowPolicy.Cast)
        {
            throw new ArgumentException(
                "[Vegetation.GenericRP] Cannot prepare a non-casting shadow batch.",
                nameof(batch));
        }
        if (frameBufferIndex == uint.MaxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(frameBufferIndex),
                "[Vegetation.GenericRP] Shadow draws require the vegetation wind frame buffer.");
        }
        Matrix4x4 viewProjection = cascade.ViewProjection;
        VegetationPreparedMaterialData material = batch.Material;
        return new VegetationShadowDrawConstants(
            new Vector4(
                viewProjection.M11,
                viewProjection.M21,
                viewProjection.M31,
                viewProjection.M41),
            new Vector4(
                viewProjection.M12,
                viewProjection.M22,
                viewProjection.M32,
                viewProjection.M42),
            new Vector4(
                viewProjection.M13,
                viewProjection.M23,
                viewProjection.M33,
                viewProjection.M43),
            new Vector4(
                viewProjection.M14,
                viewProjection.M24,
                viewProjection.M34,
                viewProjection.M44),
            new Vector4(
                viewRelativeClusterOrigin,
                BitConverter.UInt32BitsToSingle(batch.InstanceBufferIndex)),
            new Vector4(
                material.AlphaCutoff,
                BitConverter.UInt32BitsToSingle(material.BaseColorImageIndex),
                BitConverter.UInt32BitsToSingle(material.BaseColorSamplerIndex),
                material.BaseColorFactor.W),
            new Vector4(
                BitConverter.UInt32BitsToSingle(material.Flags),
                batch.WindStiffness,
                batch.ResolveFadeDistance(),
                BitConverter.UInt32BitsToSingle(frameBufferIndex)));
    }
}

internal readonly struct VegetationShadowPreparedDraw
{
    public VegetationShadowPreparedDraw(
        in VegetationPreparedBatch batch,
        in VegetationShadowDrawConstants constants)
    {
        if (!batch.IsValid || batch.ShadowPolicy != VegetationShadowPolicy.Cast)
        {
            throw new ArgumentException(
                "[Vegetation.GenericRP] Shadow draw batch is invalid.",
                nameof(batch));
        }
        if (constants.InstanceBufferIndex != batch.InstanceBufferIndex)
        {
            throw new ArgumentException(
                "[Vegetation.GenericRP] Shadow draw constants reference a different instance buffer.",
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
    public VegetationShadowDrawConstants Constants { get; }
    public bool IsValid =>
        VertexBuffer.IsValid &&
        IndexBuffer.IsValid &&
        IndexCount > 0 &&
        InstanceCount > 0;
}

internal sealed class VegetationShadowPass : RenderPassNode
{
    private const ulong DynamicViewportScissorMask = 0x1UL | 0x2UL;
    private const string VertexStage = "Vertex";
    private const string FragmentStage = "Fragment";
    private const float RasterDepthBiasConstantFactor = 1.25f;
    private const float RasterDepthBiasSlopeFactor = 1.75f;
    private const int PipelineCleanupLeg = 0;
    private const int PipelineStateCleanupLeg = 1;
    private const int VertexProgramCleanupLeg = 2;
    private const int VertexShaderAssetCleanupLeg = 3;
    private const int FragmentProgramCleanupLeg = 4;
    private const int FragmentShaderAssetCleanupLeg = 5;

    private readonly IAssetDatabase m_AssetDatabase;
    private readonly ShaderAsset m_Shader;
    private readonly RHIImageViewHandle[] m_DepthTargets =
        new RHIImageViewHandle[4];
    private VegetationShadowPreparedDraw[] m_Draws =
        Array.Empty<VegetationShadowPreparedDraw>();
    private int m_DrawCount;
    private DirectionalShadowCascadeDrawRangeSet m_DrawRanges;
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
    private EFormat m_DepthFormat = EFormat.FORMAT_UNDEFINED;
    private uint m_Width;
    private uint m_Height;
    private VegetationPassCleanupJournal m_PipelineCleanup;
    private RHIBufferHandle m_FrameDataBuffer = RHIBufferHandle.Invalid;

    public VegetationShadowPass(IAssetDatabase assetDatabase)
        : base("VegetationDirectionalShadowPass")
    {
        m_AssetDatabase = assetDatabase ?? throw new ArgumentNullException(nameof(assetDatabase));
        m_Shader = VegetationGenericRenderPipelineShaderAssets.CreateVegetationShadow();
    }

    public int LastRecordedBatchCount => Volatile.Read(ref m_LastRecordedBatchCount);
    public long LastRecordedInstanceCount => Volatile.Read(ref m_LastRecordedInstanceCount);

    /// <summary>
    /// Declares the host-written vegetation frame constants that the shadow draw constants point
    /// at. The shadow vertex shader reproduces the opaque displacement and fade from those same
    /// records, so this pass has to publish its own host-write to shader-read dependency instead of
    /// relying on the opaque pass having published one earlier in the queue.
    /// </summary>
    public void SetFrameDataBuffer(RHIBufferHandle buffer)
    {
        m_FrameDataBuffer = buffer;
    }

    public static void DeclareGraphAccess(
        RenderGraphBuilder builder,
        RenderResource directionalShadow)
    {
        ArgumentNullException.ThrowIfNull(builder);
        builder.ReadWriteDepthAttachment(
            directionalShadow,
            RenderAttachmentIntent.LoadStore);
    }

    public void SetPreparedDraws(
        VegetationShadowPreparedDraw[] draws,
        int drawCount,
        in DirectionalShadowCascadeDrawRangeSet drawRanges)
    {
        ArgumentNullException.ThrowIfNull(draws);
        if ((uint)drawCount > (uint)draws.Length ||
            drawRanges.TotalDrawCount != drawCount)
        {
            throw new ArgumentOutOfRangeException(nameof(drawCount));
        }

        m_Draws = draws;
        m_DrawCount = drawCount;
        m_DrawRanges = drawRanges;
        Volatile.Write(ref m_LastRecordedBatchCount, 0);
        Volatile.Write(ref m_LastRecordedInstanceCount, 0);
    }

    public void SetTargets(RenderGraphTexture shadowTexture, int cascadeCount)
    {
        ArgumentNullException.ThrowIfNull(shadowTexture);
        if (!shadowTexture.IsValid ||
            cascadeCount < 1 ||
            cascadeCount > m_DepthTargets.Length ||
            shadowTexture.ArrayLayers < (uint)cascadeCount)
        {
            throw new ArgumentException(
                "[Vegetation.GenericRP] Cascade shadow target is invalid.",
                nameof(shadowTexture));
        }

        for (int targetIndex = 0;
             targetIndex < m_DepthTargets.Length;
             targetIndex++)
        {
            m_DepthTargets[targetIndex] = targetIndex < cascadeCount
                ? shadowTexture.GetLayerImageView(checked((uint)targetIndex))
                : RHIImageViewHandle.Invalid;
        }

        m_Width = shadowTexture.Width;
        m_Height = shadowTexture.Height;
    }

    public void Prepare(RenderContext context)
    {
        Volatile.Write(ref m_LastRecordedBatchCount, 0);
        Volatile.Write(ref m_LastRecordedInstanceCount, 0);
        const EFormat depthFormat = EFormat.FORMAT_D32_SFLOAT;
        AssetDependencyStamp shaderStamp =
            AssetDependencyTracker.GetShaderStamp(m_AssetDatabase, m_Shader);
        if (m_Pipeline.IsValid &&
            m_Device.Handle == context.Device.Handle &&
            m_DepthFormat == depthFormat &&
            m_ShaderStamp == shaderStamp)
        {
            return;
        }

        ReleaseDeviceResources();
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
            m_PipelineState.SetRasterizationStateWithDepthBias(
                EPolygonMode.EPOLYGON_MODE_FILL,
                ECullModeFlagBits.CULL_MODE_NONE,
                EFrontFace.FRONT_FACE_COUNTER_CLOCKWISE,
                RasterDepthBiasConstantFactor,
                depthBiasClamp: 0.0f,
                depthBiasSlopeFactor: RasterDepthBiasSlopeFactor);
            m_PipelineState.SetColorBlendState(false);
            m_PipelineState.SetDepthStencilState(
                true,
                true,
                ECompareOp.COMPARE_OP_LESS_OR_EQUAL);
            m_PipelineState.SetDynamicStateMask(DynamicViewportScissorMask);
            m_PipelineState.SetRenderingFormats(
                Array.Empty<EFormat>(),
                depthFormat);
            m_PipelineState.BuildDescriptorSetLayout();
            m_Pipeline = pipelineCache.GetGraphicsPipeline(m_PipelineState);
            if (!m_Pipeline.IsValid)
            {
                throw new InvalidOperationException(
                    "[Vegetation.GenericRP] Failed to create vegetation shadow pipeline.");
            }

            m_DepthFormat = depthFormat;
            m_ShaderStamp = shaderStamp;
        }
        catch (Exception preparationFailure)
        {
            try
            {
                ReleaseDeviceResources();
            }
            catch (Exception cleanupFailure)
            {
                throw new AggregateException(
                    "Vegetation shadow pipeline preparation and rollback both failed.",
                    preparationFailure,
                    cleanupFailure);
            }

            throw;
        }
    }

    protected override void Record(
        RenderContext context,
        RenderCommandList commandList)
    {
        Volatile.Write(ref m_LastRecordedBatchCount, 0);
        Volatile.Write(ref m_LastRecordedInstanceCount, 0);
        for (int cascadeIndex = 0;
             cascadeIndex < m_DrawRanges.Count;
             cascadeIndex++)
        {
            DirectionalShadowCascadeDrawRange range =
                m_DrawRanges.GetRange(cascadeIndex);
            if (range.IsEmpty)
            {
                continue;
            }

            RecordCascadeRange(commandList, cascadeIndex, range.Start, range.Count);
        }
    }

    protected override int GetWorkItemCount(RenderContext context) =>
        VegetationShadowDrawWorkPartition.GetWorkItemCount(m_DrawRanges);

    protected override RenderPassWorkItem GetWorkItem(
        RenderContext context,
        int workItemIndex)
    {
        if (!VegetationShadowDrawWorkPartition.TryGetRange(
                m_DrawRanges,
                workItemIndex,
                out VegetationShadowDrawRange range))
        {
            return RenderPassWorkItem.Pass(workItemIndex);
        }

        return RenderPassWorkItem.DrawRange(workItemIndex, range.Start, range.Count);
    }

    protected override void Record(
        RenderContext context,
        RenderCommandList commandList,
        RenderPassWorkItem workItem)
    {
        if (!workItem.HasDrawRange ||
            !VegetationShadowDrawWorkPartition.TryGetRange(
                m_DrawRanges,
                workItem.Index,
                out VegetationShadowDrawRange range))
        {
            return;
        }

        RecordCascadeRange(commandList, range.CascadeIndex, range.Start, range.Count);
    }

    private void RecordCascadeRange(
        RenderCommandList commandList,
        int cascadeIndex,
        int drawStart,
        int drawCount)
    {
        if (!m_Pipeline.IsValid ||
            m_Width == 0 ||
            m_Height == 0 ||
            drawCount <= 0 ||
            (uint)cascadeIndex >= (uint)m_DepthTargets.Length)
        {
            return;
        }

        int drawEnd = checked(drawStart + drawCount);
        int recordedBatches = 0;
        long recordedInstances = 0;
        RecordFrameDataBarrier(commandList);
        commandList.BeginRenderingDepthOnly(
            m_DepthTargets[cascadeIndex],
            EImageLayout.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            EAttachmentLoadOp.ATTACHMENT_LOAD_OP_LOAD,
            EAttachmentStoreOp.ATTACHMENT_STORE_OP_STORE,
            1.0f,
            0,
            0,
            0,
            m_Width,
            m_Height);
        commandList.BindPipeline(m_Pipeline);
        commandList.SetViewport(0, 0, m_Width, m_Height);
        commandList.SetScissor(0, 0, m_Width, m_Height);
        for (int drawIndex = drawStart;
             drawIndex < drawEnd;
             drawIndex++)
        {
            ref readonly VegetationShadowPreparedDraw draw =
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
        m_PipelineCleanup.Release(PipelineCleanupLeg, ReleasePipeline);
        m_PipelineCleanup.Release(PipelineStateCleanupLeg, ReleasePipelineState);
        m_PipelineCleanup.Release(VertexProgramCleanupLeg, ReleaseVertexProgram);
        m_PipelineCleanup.Release(VertexShaderAssetCleanupLeg, ReleaseVertexShaderAsset);
        m_PipelineCleanup.Release(FragmentProgramCleanupLeg, ReleaseFragmentProgram);
        m_PipelineCleanup.Release(FragmentShaderAssetCleanupLeg, ReleaseFragmentShaderAsset);

        m_PipelineCache = null;
        m_ShaderStamp = AssetDependencyStamp.Empty;
        m_DepthFormat = EFormat.FORMAT_UNDEFINED;
        Array.Fill(m_DepthTargets, RHIImageViewHandle.Invalid);
        m_Width = 0;
        m_Height = 0;
        Volatile.Write(ref m_LastRecordedBatchCount, 0);
        Volatile.Write(ref m_LastRecordedInstanceCount, 0);
        m_FrameDataBuffer = RHIBufferHandle.Invalid;
        m_Factory = default;
        m_Device = default;
    }

    private void RecordFrameDataBarrier(RenderCommandList commandList)
    {
        if (!m_FrameDataBuffer.IsValid)
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
            Buffer = m_FrameDataBuffer,
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

    private void ReleasePipeline()
    {
        if (!m_Pipeline.IsValid)
        {
            return;
        }

        RHIPipelineCache pipelineCache = m_PipelineCache ??
            throw new InvalidOperationException(
                "[Vegetation.GenericRP] Cannot release the shadow pipeline without its cache.");
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
                "[Vegetation.GenericRP] Cannot release the shadow vertex program without its factory.");
        }

        m_Factory.ReleaseGPUProgram(m_VertexProgram);
        m_VertexProgram = RHIShaderProgramHandle.Invalid;
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

    private void ReleaseFragmentProgram()
    {
        if (!m_FragmentProgram.IsValid)
        {
            return;
        }
        if (!m_Factory.IsValid)
        {
            throw new InvalidOperationException(
                "[Vegetation.GenericRP] Cannot release the shadow fragment program without its factory.");
        }

        m_Factory.ReleaseGPUProgram(m_FragmentProgram);
        m_FragmentProgram = RHIShaderProgramHandle.Invalid;
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
                    $"[Vegetation.GenericRP] Failed to prepare vegetation shadow shader stage '{stageName}'.");
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
            EFormat.FORMAT_R32G32_SFLOAT,
            40);
    }
}
