using ArisenEngine.Core.Diagnostics;
using ArisenEngine.Rendering;
using ArisenEngine.Resources.Serialization;
using ArisenEngine.Vegetation.Assets;
using ArisenEngine.Threading;
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
    private readonly VegetationCullingPlanner m_CullingPlanner;
    private readonly VegetationRenderValidationMode m_ValidationMode;
    private readonly VegetationSetupWorkDispatcher m_SetupDispatcher;
    private readonly Action<int> m_GatherSetupWork;
    private readonly Action<int> m_PrepareSetupWork;
    private VegetationClusterComponent[] m_ExtractedClusters =
        Array.Empty<VegetationClusterComponent>();
    private VegetationPreparedClusterFrame[] m_PreparedClusters =
        Array.Empty<VegetationPreparedClusterFrame>();
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
    private VegetationClusterCullingInput[] m_CullingInputs =
        Array.Empty<VegetationClusterCullingInput>();
    private VegetationCullingSelection[] m_CullingSelections =
        Array.Empty<VegetationCullingSelection>();
    private int m_CullingInputCount;
    private int m_CullingSelectionCount;
    private readonly VegetationSetupShardBuffer<VegetationClusterCullingInput>
        m_CullingInputRegions = new();
    private readonly VegetationSetupShardBuffer<VegetationPreparedClusterFrame>
        m_PreparedClusterRegions = new();

    public VegetationGenericRenderPipelineFeature(
        IVegetationClusterRenderSource renderSource,
        IVegetationClusterDataSource clusterData,
        IVegetationDiagnosticsPublisher diagnostics,
        IVegetationAuthoringPreviewService authoringPreviews,
        VegetationPreparedAssetProvider preparedAssets,
        VegetationOpaquePass opaquePass,
        VegetationShadowPass shadowPass,
        VegetationRenderValidationMode validationMode,
        ITaskGraph? taskSystem = null)
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
        m_SetupDispatcher = new VegetationSetupWorkDispatcher(taskSystem);
        m_CullingPlanner = new VegetationCullingPlanner(taskSystem);
        m_GatherSetupWork = RunGatherSetupWorkItem;
        m_PrepareSetupWork = RunPrepareSetupWorkItem;
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
        VegetationCullingMetrics culling = m_CullingPlanner.Metrics;
        Profiler.PlotValue("Vegetation.CullingCandidateSpecies", culling.CandidateSpeciesCount);
        Profiler.PlotValue("Vegetation.CullingVisibleSpecies", culling.VisibleSpeciesCount);
        Profiler.PlotValue("Vegetation.CullingCulledClusters", culling.CulledClusterCount);
        Profiler.PlotValue("Vegetation.CullingCulledPages", culling.CulledPageCount);
        Profiler.PlotValue("Vegetation.CullingCulledSpecies", culling.CulledSpeciesCount);
        Profiler.PlotValue("Vegetation.CullingDroppedSpecies", culling.DroppedSpeciesCount);
        Profiler.PlotValue("Vegetation.CullingSelectedSpecies", culling.SelectedSpeciesCount);
        Profiler.PlotValue("Vegetation.CullingSelectedInstances", culling.SelectedInstanceCount);
        Profiler.PlotValue("Vegetation.CullingSelectedBatches", culling.SelectedBatchCount);
        Profiler.PlotValue("Vegetation.CullingMaximumLod", culling.MaximumSelectedLod);
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
        m_CullingInputCount = 0;
        m_CullingSelectionCount = 0;
        m_CullingPlanner.Reset();
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
        m_CullingInputCount = 0;
        m_CullingSelectionCount = 0;
        using (Profiler.Zone("Vegetation.GatherCullingInputs"))
        {
            int gatherWorkItemCount =
                VegetationSetupWorkPartition.GetWorkItemCount(m_ExtractedClusterCount);
            m_CullingInputRegions.EnsureRegions(
                gatherWorkItemCount,
                VegetationSetupWorkPartition.MaximumInputsPerWorkItem);
            m_SetupDispatcher.Dispatch(m_ExtractedClusterCount, m_GatherSetupWork);
            EnsureCapacity(ref m_CullingInputs, m_ExtractedClusterCount);
            m_CullingInputCount = m_CullingInputRegions.MergeInto(
                new Span<VegetationClusterCullingInput>(
                    m_CullingInputs,
                    0,
                    m_ExtractedClusterCount));
            m_DroppedDrawCount += m_ExtractedClusterCount - m_CullingInputCount;
        }

        using var cullingZone = Profiler.Zone("Vegetation.CullingPlan");
        ReadOnlySpan<VegetationCullingSelection> selections = m_CullingPlanner.Plan(
            new ReadOnlySpan<VegetationClusterCullingInput>(
                m_CullingInputs,
                0,
                m_CullingInputCount),
            CreateCullingView(context),
            VegetationCullingSettings.Default);
        EnsureCapacity(ref m_CullingSelections, selections.Length);
        selections.CopyTo(m_CullingSelections);
        m_CullingSelectionCount = selections.Length;

        using (Profiler.Zone("Vegetation.PrepareDrawSetup"))
        {
            int preparedWorkItemCount =
                VegetationSetupWorkPartition.GetWorkItemCount(m_CullingInputCount);
            m_PreparedClusterRegions.EnsureRegions(
                preparedWorkItemCount,
                VegetationSetupWorkPartition.MaximumInputsPerWorkItem);
            m_SetupDispatcher.Dispatch(m_CullingInputCount, m_PrepareSetupWork);
            EnsureCapacity(ref m_PreparedClusters, m_CullingInputCount);
            m_PreparedClusterCount = m_PreparedClusterRegions.MergeInto(
                new Span<VegetationPreparedClusterFrame>(
                    m_PreparedClusters,
                    0,
                    m_CullingInputCount));
            m_DroppedDrawCount += m_CullingInputCount - m_PreparedClusterCount;
        }

        using (Profiler.Zone("Vegetation.EmitOpaqueDraws"))
        {
            for (int clusterIndex = 0;
                 clusterIndex < m_PreparedClusterCount;
                 clusterIndex++)
            {
                ref readonly VegetationPreparedClusterFrame cluster =
                    ref m_PreparedClusters[clusterIndex];
                bool receiveShadows =
                    (cluster.Component.Flags & VegetationClusterFlags.ReceiveShadows) != 0;
                ReadOnlySpan<VegetationPreparedBatch> batches = cluster.Prepared.Batches;
                for (int batchIndex = 0; batchIndex < batches.Length; batchIndex++)
                {
                    ref readonly VegetationPreparedBatch batch = ref batches[batchIndex];
                    if (batch.SpeciesGuid != cluster.Selection.SpeciesGuid ||
                        batch.LodLevel != cluster.Selection.LodLevel)
                    {
                        continue;
                    }

                    EnsureCapacity(
                        ref m_OpaqueDraws,
                        checked(m_OpaqueDrawCount + 1));
                    VegetationOpaqueDrawConstants constants =
                        VegetationOpaqueDrawConstants.Create(
                            context.RenderContext.RenderOrigin,
                            cluster.Prepared.Origin,
                            batch,
                            frameBufferIndex,
                            receiveShadows,
                            context.DirectionalShadow);
                    m_OpaqueDraws[m_OpaqueDrawCount++] =
                        new VegetationOpaquePreparedDraw(batch, constants);
                }
            }
        }

        m_OpaquePass.SetPreparedDraws(m_OpaqueDraws, m_OpaqueDrawCount);
    }

    private void RunGatherSetupWorkItem(int workItemIndex)
    {
        if (!VegetationSetupWorkPartition.TryGetRange(
                m_ExtractedClusterCount,
                workItemIndex,
                out int start,
                out int count))
        {
            throw new InvalidOperationException(
                $"Vegetation gather work item '{workItemIndex}' is outside the " +
                $"dispatched range of {m_ExtractedClusterCount} items.");
        }

        int written = VegetationPreparedSetup.GatherCullingInputs(
            m_ExtractedClusters,
            start,
            count,
            m_RuntimeSnapshot.Clusters,
            m_PreparedAssets,
            m_CullingInputRegions.GetRegion(workItemIndex));
        m_CullingInputRegions.SetCount(workItemIndex, written);
    }

    private void RunPrepareSetupWorkItem(int workItemIndex)
    {
        if (!VegetationSetupWorkPartition.TryGetRange(
                m_CullingInputCount,
                workItemIndex,
                out int start,
                out int count))
        {
            throw new InvalidOperationException(
                $"Vegetation prepare work item '{workItemIndex}' is outside the " +
                $"dispatched range of {m_CullingInputCount} items.");
        }

        int written = VegetationPreparedSetup.BuildPreparedFrames(
            new ReadOnlySpan<VegetationClusterCullingInput>(
                m_CullingInputs,
                0,
                m_CullingInputCount),
            start,
            count,
            new ReadOnlySpan<VegetationCullingSelection>(
                m_CullingSelections,
                0,
                m_CullingSelectionCount),
            m_PreparedClusterRegions.GetRegion(workItemIndex));
        m_PreparedClusterRegions.SetCount(workItemIndex, written);
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
                ref readonly VegetationPreparedClusterFrame cluster =
                    ref m_PreparedClusters[clusterIndex];
                if ((cluster.Component.Flags & VegetationClusterFlags.CastShadows) == 0)
                {
                    continue;
                }

                ReadOnlySpan<VegetationPreparedBatch> batches = cluster.Prepared.Batches;
                for (int batchIndex = 0; batchIndex < batches.Length; batchIndex++)
                {
                    ref readonly VegetationPreparedBatch batch = ref batches[batchIndex];
                    if (batch.SpeciesGuid != cluster.Selection.SpeciesGuid ||
                        batch.LodLevel != cluster.Selection.LodLevel ||
                        batch.ShadowPolicy != VegetationShadowPolicy.Cast)
                    {
                        continue;
                    }

                    EnsureCapacity(
                        ref m_ShadowDraws,
                        checked(m_ShadowDrawCount + 1));
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

    private static VegetationCullingView CreateCullingView(
        in GenericRenderPipelineFeatureFrameContext context)
    {
        if (context.Snapshot.CameraCount == 0)
        {
            return default;
        }

        Camera camera = context.Snapshot.Cameras[0];
        WorldPosition renderOrigin = context.RenderContext.RenderOrigin;
        Vector3 cameraPosition = context.CameraPosition;
        return new VegetationCullingView(
            new WorldPosition(
                renderOrigin.X + cameraPosition.X,
                renderOrigin.Y + cameraPosition.Y,
                renderOrigin.Z + cameraPosition.Z),
            renderOrigin,
            context.ViewProjection,
            camera.ProjectionType == CameraProjectionType.Perspective
                ? VegetationCullingProjection.Perspective
                : VegetationCullingProjection.Orthographic,
            camera.FieldOfView * (Math.PI / 180.0),
            camera.OrthographicSize * 2.0,
            context.RenderContext.Height > int.MaxValue
                ? 0
                : (int)context.RenderContext.Height);
    }

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
        m_CullingInputCount = 0;
        m_CullingSelectionCount = 0;
        m_CullingPlanner.Reset();
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

    private readonly record struct ReportedSurfaceGeneration(
        uint SurfaceId,
        ulong DeviceGeneration);
}
