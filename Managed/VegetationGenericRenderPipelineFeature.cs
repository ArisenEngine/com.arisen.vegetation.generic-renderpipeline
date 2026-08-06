using ArisenEngine.Core.Diagnostics;
using ArisenEngine.Rendering;
using ArisenEngine.Resources.Serialization;
using ArisenEngine.Vegetation.Assets;
using ArisenKernel.Diagnostics;

namespace ArisenEngine.Vegetation.GenericRenderPipeline;

internal sealed class VegetationGenericRenderPipelineFeature : IGenericRenderPipelineFeature
{
    public const string Id = "com.arisen.vegetation.generic-renderpipeline";
    private const int ReportedSurfaceCapacity = 8;

    private readonly IVegetationClusterRenderSource m_RenderSource;
    private readonly IVegetationClusterDataSource m_ClusterData;
    private readonly IVegetationDiagnosticsPublisher m_Diagnostics;
    private readonly IVegetationAuthoringPreviewService m_AuthoringPreviews;
    private readonly VegetationPreparedAssetProvider m_PreparedAssets;
    private readonly VegetationOpaquePass m_OpaquePass;
    private readonly VegetationShadowPass m_ShadowPass;
    private readonly VegetationRenderValidationMode m_ValidationMode;
    private VegetationClusterComponent[] m_ExtractedClusters =
        Array.Empty<VegetationClusterComponent>();
    private PreparedClusterFrame[] m_PreparedClusters = Array.Empty<PreparedClusterFrame>();
    private VegetationOpaquePreparedDraw[] m_OpaqueDraws =
        Array.Empty<VegetationOpaquePreparedDraw>();
    private VegetationShadowPreparedDraw[] m_ShadowDraws =
        Array.Empty<VegetationShadowPreparedDraw>();
    private readonly ReportedSurfaceGeneration[] m_ReportedSurfaces =
        new ReportedSurfaceGeneration[ReportedSurfaceCapacity];
    private VegetationClusterDataSnapshot m_RuntimeSnapshot =
        VegetationClusterDataSnapshot.Empty;
    private VegetationAuthoringPreviewSnapshot m_PreviewSnapshot =
        VegetationAuthoringPreviewSnapshot.Empty;
    private DirectionalShadowCascadeDrawRangeSet m_ShadowDrawRanges;
    private int m_ExtractedClusterCount;
    private int m_PreparedClusterCount;
    private int m_OpaqueDrawCount;
    private int m_ShadowDrawCount;
    private int m_DroppedDrawCount;
    private int m_ReportedSurfaceCount;
    private bool m_DeviceResourcesReleased;

    public VegetationGenericRenderPipelineFeature(
        IVegetationClusterRenderSource renderSource,
        IVegetationClusterDataSource clusterData,
        IVegetationDiagnosticsPublisher diagnostics,
        IVegetationAuthoringPreviewService authoringPreviews,
        VegetationPreparedAssetProvider preparedAssets,
        VegetationOpaquePass opaquePass,
        VegetationShadowPass shadowPass,
        VegetationRenderValidationMode validationMode)
    {
        m_RenderSource = renderSource ?? throw new ArgumentNullException(nameof(renderSource));
        m_ClusterData = clusterData ?? throw new ArgumentNullException(nameof(clusterData));
        m_Diagnostics = diagnostics ?? throw new ArgumentNullException(nameof(diagnostics));
        m_AuthoringPreviews = authoringPreviews
            ?? throw new ArgumentNullException(nameof(authoringPreviews));
        m_PreparedAssets = preparedAssets
            ?? throw new ArgumentNullException(nameof(preparedAssets));
        m_OpaquePass = opaquePass ?? throw new ArgumentNullException(nameof(opaquePass));
        m_ShadowPass = shadowPass ?? throw new ArgumentNullException(nameof(shadowPass));
        m_ValidationMode = validationMode;
    }

    public string FeatureId => Id;

    public int Order => 200;

    public void ConsumeExtractedFrame(in GenericRenderPipelineFeatureFrameContext context)
    {
        using var zone = Profiler.Zone("Vegetation.ExtractVisibleClusters");
        ReadOnlySpan<VegetationClusterComponent> visible =
            m_RenderSource.ExtractVisibleClusters();
        EnsureCapacity(ref m_ExtractedClusters, visible.Length);
        visible.CopyTo(m_ExtractedClusters);
        m_ExtractedClusterCount = visible.Length;
        m_RuntimeSnapshot = m_ClusterData.GetSnapshot();
        m_PreviewSnapshot = m_AuthoringPreviews.GetSnapshot();
        Profiler.PlotValue("Vegetation.ExtractedClusterCount", m_ExtractedClusterCount);
    }

    public void PrepareResources(in GenericRenderPipelineFeatureFrameContext context)
    {
        using var zone = Profiler.Zone("Vegetation.PrepareResources");
        m_PreparedAssets.UpdateFrameContext(
            context.RenderContext.Device,
            context.RenderContext.DeviceGeneration);
        if (m_PreparedAssets.InvalidateStaleDependencies())
        {
            ClearPreparedFrameState(
                context.DirectionalShadow.Cascades.Count,
                m_ExtractedClusterCount);
            m_DeviceResourcesReleased = false;
            PlotPreparedState();
            return;
        }

        if (m_ValidationMode == VegetationRenderValidationMode.Disabled)
        {
            ClearPreparedFrameState(
                context.DirectionalShadow.Cascades.Count,
                droppedDrawCount: 0);
            m_DeviceResourcesReleased = false;
            PlotPreparedState();
            return;
        }

        m_OpaquePass.Prepare(context.RenderContext);
        uint opaqueFrameBufferIndex = m_OpaquePass.PrepareFrame(
            context.RenderContext,
            context.ViewProjection,
            context.CameraPosition,
            context.DirectionalLight,
            context.SceneEnvironment);
        m_ShadowPass.Prepare(context.RenderContext);

        PrepareClustersAndOpaqueDraws(context, opaqueFrameBufferIndex);
        if (m_ValidationMode == VegetationRenderValidationMode.Full)
        {
            PrepareShadowDraws(context);
        }
        else
        {
            m_ShadowDrawCount = 0;
            m_ShadowDrawRanges = CreateEmptyShadowRanges(
                context.DirectionalShadow.Cascades.Count);
            m_ShadowPass.SetPreparedDraws(
                m_ShadowDraws,
                drawCount: 0,
                drawRanges: m_ShadowDrawRanges);
        }
        m_DeviceResourcesReleased = false;
        PlotPreparedState();
    }

    private void PlotPreparedState()
    {
        Profiler.PlotValue("Vegetation.PreparedClusterCount", m_PreparedClusterCount);
        Profiler.PlotValue("Vegetation.OpaqueBatchCount", m_OpaqueDrawCount);
        Profiler.PlotValue("Vegetation.ShadowBatchCount", m_ShadowDrawCount);
        Profiler.PlotValue("Vegetation.DroppedDrawCount", m_DroppedDrawCount);
        Profiler.PlotValue(
            "Vegetation.RenderValidationMode",
            (double)(int)m_ValidationMode);
    }

    public void AddRenderGraphPasses(
        GenericRenderPipelineFeatureGraphStage stage,
        in GenericRenderPipelineFeatureGraphContext context)
    {
        if (stage == GenericRenderPipelineFeatureGraphStage.DirectionalShadow)
        {
            if (m_ShadowDrawCount == 0)
            {
                return;
            }

            m_ShadowPass.SetTargets(
                context.DirectionalShadow,
                context.DirectionalShadowFrame.Cascades.Count);
            RenderResource shadowResource = context.DirectionalShadow.Resource;
            context.Graph.AddPass(
                m_ShadowPass,
                builder => VegetationShadowPass.DeclareGraphAccess(
                    builder,
                    shadowResource));
            return;
        }

        if (stage != GenericRenderPipelineFeatureGraphStage.Opaque ||
            m_OpaqueDrawCount == 0)
        {
            return;
        }

        m_OpaquePass.SetTargets(
            context.SceneColor.ImageView,
            context.FrameDepth.ImageView);
        RenderResource directionalShadowResource = context.DirectionalShadow.Resource;
        RenderResource sceneColorResource = context.SceneColor.Resource;
        RenderResource frameDepthResource = context.FrameDepth.Resource;
        context.Graph.AddPass(
            m_OpaquePass,
            builder => VegetationOpaquePass.DeclareGraphAccess(
                builder,
                directionalShadowResource,
                sceneColorResource,
                frameDepthResource));
    }

    public void OnFrameSubmitted(in GenericRenderPipelineFeatureSubmissionContext context)
    {
        m_PreparedAssets.UpdateSubmittedTicket(context.SubmittedTicket);
        int opaqueBatches = m_OpaquePass.LastRecordedBatchCount;
        long opaqueInstances = m_OpaquePass.LastRecordedInstanceCount;
        int shadowBatches = m_ShadowPass.LastRecordedBatchCount;
        long shadowInstances = m_ShadowPass.LastRecordedInstanceCount;
        Profiler.PlotValue("Vegetation.SubmittedOpaqueBatchCount", opaqueBatches);
        Profiler.PlotValue("Vegetation.SubmittedOpaqueInstanceCount", opaqueInstances);
        Profiler.PlotValue("Vegetation.SubmittedShadowBatchCount", shadowBatches);
        Profiler.PlotValue("Vegetation.SubmittedShadowInstanceCount", shadowInstances);

        if (opaqueBatches > 0 &&
            opaqueInstances > 0 &&
            shadowBatches > 0 &&
            shadowInstances > 0 &&
            context.SubmittedTicket > 0 &&
            TryMarkSurfaceReported(
                context.Frame.RenderContext.SurfaceId,
                context.Frame.RenderContext.DeviceGeneration))
        {
            LogSubmittedDrawValidation(
                context,
                opaqueBatches,
                opaqueInstances,
                shadowBatches,
                shadowInstances);
        }

        if (m_RuntimeSnapshot.IsEmpty && !m_PreviewSnapshot.HasPreview)
        {
            m_Diagnostics.Clear();
        }
    }

    public void ReleaseDeviceResources()
    {
        if (m_DeviceResourcesReleased)
        {
            return;
        }

        m_OpaquePass.ReleaseDeviceResources();
        m_ShadowPass.ReleaseDeviceResources();
        m_PreparedAssets.ReleaseAllDeviceResources();
        m_ExtractedClusterCount = 0;
        m_PreparedClusterCount = 0;
        m_OpaqueDrawCount = 0;
        m_ShadowDrawCount = 0;
        m_DroppedDrawCount = 0;
        m_RuntimeSnapshot = VegetationClusterDataSnapshot.Empty;
        m_PreviewSnapshot = VegetationAuthoringPreviewSnapshot.Empty;
        ClearPreparedFrameState(m_ShadowDrawRanges.Count, droppedDrawCount: 0);
        m_ReportedSurfaceCount = 0;
        m_Diagnostics.Clear();
        m_DeviceResourcesReleased = true;
        KernelLog.Info("[Vegetation.GenericRP] Feature device-resource release completed.");
    }

    private void PrepareClustersAndOpaqueDraws(
        in GenericRenderPipelineFeatureFrameContext context,
        uint frameBufferIndex)
    {
        m_PreparedClusterCount = 0;
        m_OpaqueDrawCount = 0;
        m_DroppedDrawCount = 0;
        ReadOnlySpan<VegetationResidentClusterData> residentClusters =
            m_RuntimeSnapshot.Clusters;
        for (int componentIndex = 0;
             componentIndex < m_ExtractedClusterCount;
             componentIndex++)
        {
            ref readonly VegetationClusterComponent component =
                ref m_ExtractedClusters[componentIndex];
            if (!TryFindResidentCluster(
                    residentClusters,
                    component.ClusterGuid,
                    out VegetationResidentClusterData resident) ||
                !MatchesComponent(component, resident) ||
                !m_PreparedAssets.TryGetCluster(
                    component.ClusterGuid,
                    resident.Generation,
                    out VegetationPreparedClusterView prepared))
            {
                m_DroppedDrawCount++;
                continue;
            }

            EnsureCapacity(ref m_PreparedClusters, m_PreparedClusterCount + 1);
            m_PreparedClusters[m_PreparedClusterCount++] = new PreparedClusterFrame(
                component,
                prepared);
            bool receiveShadows =
                (component.Flags & VegetationClusterFlags.ReceiveShadows) != 0;
            ReadOnlySpan<VegetationPreparedBatch> batches = prepared.Batches;
            EnsureCapacity(ref m_OpaqueDraws, checked(m_OpaqueDrawCount + batches.Length));
            for (int batchIndex = 0; batchIndex < batches.Length; batchIndex++)
            {
                ref readonly VegetationPreparedBatch batch = ref batches[batchIndex];
                VegetationOpaqueDrawConstants constants =
                    VegetationOpaqueDrawConstants.Create(
                        context.RenderContext.RenderOrigin,
                        prepared.Origin,
                        batch,
                        frameBufferIndex,
                        receiveShadows,
                        context.DirectionalShadow);
                m_OpaqueDraws[m_OpaqueDrawCount++] = new VegetationOpaquePreparedDraw(
                    batch,
                    constants);
            }
        }

        m_OpaquePass.SetPreparedDraws(m_OpaqueDraws, m_OpaqueDrawCount);
    }

    private void PrepareShadowDraws(
        in GenericRenderPipelineFeatureFrameContext context)
    {
        m_ShadowDrawCount = 0;
        DirectionalShadowFrameData shadow = context.DirectionalShadow;
        int cascadeCount = shadow.Cascades.Count;
        if (!shadow.Enabled || cascadeCount <= 0)
        {
            m_ShadowDrawRanges = CreateEmptyShadowRanges(cascadeCount);
            m_ShadowPass.SetPreparedDraws(
                m_ShadowDraws,
                drawCount: 0,
                drawRanges: m_ShadowDrawRanges);
            return;
        }

        Span<DirectionalShadowCascadeDrawRange> ranges =
            stackalloc DirectionalShadowCascadeDrawRange[4];
        for (int cascadeIndex = 0; cascadeIndex < cascadeCount; cascadeIndex++)
        {
            int rangeStart = m_ShadowDrawCount;
            DirectionalShadowCascade cascade = shadow.Cascades.GetCascade(cascadeIndex);
            for (int clusterIndex = 0;
                 clusterIndex < m_PreparedClusterCount;
                 clusterIndex++)
            {
                ref readonly PreparedClusterFrame cluster =
                    ref m_PreparedClusters[clusterIndex];
                if ((cluster.Component.Flags & VegetationClusterFlags.CastShadows) == 0)
                {
                    continue;
                }

                ReadOnlySpan<VegetationPreparedBatch> batches = cluster.Prepared.Batches;
                EnsureCapacity(
                    ref m_ShadowDraws,
                    checked(m_ShadowDrawCount + batches.Length));
                for (int batchIndex = 0; batchIndex < batches.Length; batchIndex++)
                {
                    ref readonly VegetationPreparedBatch batch = ref batches[batchIndex];
                    if (batch.ShadowPolicy != VegetationShadowPolicy.Cast)
                    {
                        continue;
                    }

                    VegetationShadowDrawConstants constants =
                        VegetationShadowDrawConstants.Create(
                            cascade,
                            shadow.Cascades.CameraPosition,
                            context.RenderContext.RenderOrigin,
                            cluster.Prepared.Origin,
                            batch);
                    m_ShadowDraws[m_ShadowDrawCount++] =
                        new VegetationShadowPreparedDraw(batch, constants);
                }
            }

            ranges[cascadeIndex] = new DirectionalShadowCascadeDrawRange(
                rangeStart,
                m_ShadowDrawCount - rangeStart);
        }

        m_ShadowDrawRanges = new DirectionalShadowCascadeDrawRangeSet(
            cascadeCount,
            m_ShadowDrawCount,
            droppedDrawCount: 0,
            ranges[0],
            ranges[1],
            ranges[2],
            ranges[3]);
        m_ShadowPass.SetPreparedDraws(
            m_ShadowDraws,
            m_ShadowDrawCount,
            m_ShadowDrawRanges);
    }

    private static bool TryFindResidentCluster(
        ReadOnlySpan<VegetationResidentClusterData> clusters,
        Guid clusterGuid,
        out VegetationResidentClusterData cluster)
    {
        for (int index = 0; index < clusters.Length; index++)
        {
            VegetationResidentClusterData candidate = clusters[index];
            if (candidate.Guid == clusterGuid)
            {
                cluster = candidate;
                return true;
            }
        }

        cluster = null!;
        return false;
    }

    private static bool MatchesComponent(
        in VegetationClusterComponent component,
        VegetationResidentClusterData resident) =>
        resident.BiomeGuid == component.BiomeGuid &&
        resident.ContainsSpecies(component.SpeciesGuid) &&
        resident.Origin == new WorldPosition(
            component.OriginX,
            component.OriginY,
            component.OriginZ) &&
        resident.PageCount == component.PageCount &&
        resident.InstanceCount == component.InstanceCount;

    private static DirectionalShadowCascadeDrawRangeSet CreateEmptyShadowRanges(
        int cascadeCount) => cascadeCount == 0
            ? default
            : new DirectionalShadowCascadeDrawRangeSet(
                cascadeCount,
                totalDrawCount: 0,
                droppedDrawCount: 0,
                default,
                default,
                default,
                default);

    private void ClearPreparedFrameState(int cascadeCount, int droppedDrawCount)
    {
        m_PreparedClusterCount = 0;
        m_OpaqueDrawCount = 0;
        m_ShadowDrawCount = 0;
        m_DroppedDrawCount = droppedDrawCount;
        m_OpaquePass.SetPreparedDraws(m_OpaqueDraws, drawCount: 0);
        m_ShadowDrawRanges = CreateEmptyShadowRanges(cascadeCount);
        m_ShadowPass.SetPreparedDraws(
            m_ShadowDraws,
            drawCount: 0,
            drawRanges: m_ShadowDrawRanges);
    }

    private bool TryMarkSurfaceReported(uint surfaceId, ulong deviceGeneration)
    {
        for (int index = 0; index < m_ReportedSurfaceCount; index++)
        {
            ref readonly ReportedSurfaceGeneration reported =
                ref m_ReportedSurfaces[index];
            if (reported.SurfaceId == surfaceId &&
                reported.DeviceGeneration == deviceGeneration)
            {
                return false;
            }
        }

        if (m_ReportedSurfaceCount >= m_ReportedSurfaces.Length)
        {
            return false;
        }

        m_ReportedSurfaces[m_ReportedSurfaceCount++] = new ReportedSurfaceGeneration(
            surfaceId,
            deviceGeneration);
        return true;
    }

    private void LogSubmittedDrawValidation(
        in GenericRenderPipelineFeatureSubmissionContext context,
        int opaqueBatches,
        long opaqueInstances,
        int shadowBatches,
        long shadowInstances)
    {
        Guid clusterGuid = m_PreparedClusterCount == 1
            ? m_PreparedClusters[0].Prepared.ClusterGuid
            : Guid.Empty;
        Guid speciesGuid = Guid.Empty;
        if (m_PreparedClusterCount == 1 &&
            m_PreparedClusters[0].Prepared.Batches.Length > 0)
        {
            speciesGuid = m_PreparedClusters[0].Prepared.Batches[0].SpeciesGuid;
        }

        KernelLog.InfoFormat(
            "[Vegetation.GenericRP.Validation] Surface=0x{0:X} Frame={1} " +
            "DeviceGeneration={2} Revision={3} PreparedClusters={4} Cluster={5:D} " +
            "Species={6:D} OpaqueBatches={7} OpaqueInstances={8} " +
            "RecordedShadowBatches={9} RecordedShadowInstances={10} Cascades={11} " +
            "ShadowBatches={12},{13},{14},{15} ShadowInstances={16},{17},{18},{19} " +
            "Dropped={20} Ticket={21}",
            context.Frame.RenderContext.SurfaceId,
            context.Frame.RenderContext.FrameIndex,
            context.Frame.RenderContext.DeviceGeneration,
            m_RuntimeSnapshot.Revision,
            m_PreparedClusterCount,
            clusterGuid,
            speciesGuid,
            opaqueBatches,
            opaqueInstances,
            shadowBatches,
            shadowInstances,
            m_ShadowDrawRanges.Count,
            GetCascadeBatchCount(0),
            GetCascadeBatchCount(1),
            GetCascadeBatchCount(2),
            GetCascadeBatchCount(3),
            GetCascadeInstanceCount(0),
            GetCascadeInstanceCount(1),
            GetCascadeInstanceCount(2),
            GetCascadeInstanceCount(3),
            m_DroppedDrawCount,
            context.SubmittedTicket);
    }

    private int GetCascadeBatchCount(int cascadeIndex) =>
        cascadeIndex < m_ShadowDrawRanges.Count
            ? m_ShadowDrawRanges.GetRange(cascadeIndex).Count
            : 0;

    private long GetCascadeInstanceCount(int cascadeIndex)
    {
        if (cascadeIndex >= m_ShadowDrawRanges.Count)
        {
            return 0;
        }

        DirectionalShadowCascadeDrawRange range =
            m_ShadowDrawRanges.GetRange(cascadeIndex);
        long instanceCount = 0;
        int end = checked(range.Start + range.Count);
        for (int index = range.Start; index < end; index++)
        {
            instanceCount = checked(instanceCount + m_ShadowDraws[index].InstanceCount);
        }
        return instanceCount;
    }

    private static void EnsureCapacity<T>(ref T[] storage, int required)
    {
        if (required <= storage.Length)
        {
            return;
        }

        int capacity = Math.Max(4, storage.Length);
        while (capacity < required)
        {
            capacity = checked(capacity * 2);
        }
        Array.Resize(ref storage, capacity);
    }

    private readonly struct PreparedClusterFrame
    {
        public PreparedClusterFrame(
            in VegetationClusterComponent component,
            in VegetationPreparedClusterView prepared)
        {
            Component = component;
            Prepared = prepared;
        }

        public VegetationClusterComponent Component { get; }
        public VegetationPreparedClusterView Prepared { get; }
    }

    private readonly record struct ReportedSurfaceGeneration(
        uint SurfaceId,
        ulong DeviceGeneration);
}
