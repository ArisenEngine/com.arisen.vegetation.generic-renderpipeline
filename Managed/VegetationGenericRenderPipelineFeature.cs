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
    private const int MaximumLoggedClusters = 8;

    private readonly IVegetationClusterRenderSource m_RenderSource;
    private readonly IVegetationClusterDataSource m_ClusterData;
    private readonly IVegetationDiagnosticsPublisher m_Diagnostics;
    private readonly IVegetationAuthoringPreviewService m_AuthoringPreviews;
    private readonly IVegetationWindSource m_WindSource;
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
    private readonly Guid[] m_LoggedClusterGuids = new Guid[MaximumLoggedClusters];
    private readonly Guid[] m_LoggedSpeciesGuids = new Guid[MaximumLoggedClusters];
    private readonly long[] m_LoggedClusterInstances = new long[MaximumLoggedClusters];
    private readonly System.Text.StringBuilder m_LoggedClusterList = new();
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
        IVegetationWindSource windSource,
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
        m_WindSource = windSource ?? throw new ArgumentNullException(nameof(windSource));
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

    /// <summary>
    /// Every vertex position the vegetation passes consume is relative to the render camera, so the
    /// camera sits exactly at the origin of the frame those passes render in.
    /// </summary>
    private static Vector3 ViewFrameCameraPosition => Vector3.Zero;

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
        if (!TryResolveCameraWorldPosition(context, out WorldPosition cameraWorldPosition))
        {
            ClearPreparedFrameState(
                context.DirectionalShadow.Cascades.Count,
                droppedDrawCount: 0);
            m_DeviceResourcesReleased = false;
            PlotPreparedState();
            return;
        }

        VegetationWindSettings wind = m_WindSource.Current;
        uint opaqueFrameBufferIndex = m_OpaquePass.PrepareFrame(
            context.RenderContext,
            context.ViewRelativeViewProjection,
            ViewFrameCameraPosition,
            context.DirectionalLight,
            context.SceneEnvironment,
            wind,
            cameraWorldPosition);
        m_ShadowPass.Prepare(context.RenderContext);

        PrepareClustersAndOpaqueDraws(
            context,
            opaqueFrameBufferIndex,
            cameraWorldPosition);
        if (m_ValidationMode == VegetationRenderValidationMode.Full)
        {
            PrepareShadowDraws(
                context,
                opaqueFrameBufferIndex,
                cameraWorldPosition);
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
        uint frameBufferIndex,
        WorldPosition cameraWorldPosition)
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
            m_DroppedDrawCount += CountAcceptedCullingInputs() - m_PreparedClusterCount;
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
                            VegetationViewFrame.ToViewRelativePosition(
                                cluster.Prepared.Origin,
                                cameraWorldPosition),
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
        in GenericRenderPipelineFeatureFrameContext context,
        uint opaqueFrameBufferIndex,
        WorldPosition cameraWorldPosition)
    {
        m_ShadowDrawCount = 0;
        DirectionalShadowFrameData shadow = context.DirectionalShadow;
        m_ShadowPass.SetFrameDataBuffer(m_OpaquePass.FrameConstantsBuffer);
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
                            VegetationViewFrame.ToViewRelativePosition(
                                cluster.Prepared.Origin,
                                cameraWorldPosition),
                            batch,
                            opaqueFrameBufferIndex);
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
        if (!TryResolveCameraWorldPosition(context, out WorldPosition cameraWorldPosition))
        {
            return default;
        }

        Camera camera = context.Snapshot.Cameras[0];
        return new VegetationCullingView(
            cameraWorldPosition,
            context.RenderContext.RenderOrigin,
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

    /// <summary>
    /// Resolves the render camera's absolute world position, which is the anchor of the view frame
    /// the vegetation passes render in. The cached snapshot only stores the origin-relative camera
    /// position, so the world origin is added back in double precision.
    /// </summary>
    private static bool TryResolveCameraWorldPosition(
        in GenericRenderPipelineFeatureFrameContext context,
        out WorldPosition cameraWorldPosition)
    {
        if (context.Snapshot.CameraCount == 0)
        {
            cameraWorldPosition = default;
            return false;
        }

        WorldPosition renderOrigin = context.RenderContext.RenderOrigin;
        Vector3 cameraPosition = context.CameraPosition;
        cameraWorldPosition = new WorldPosition(
            renderOrigin.X + cameraPosition.X,
            renderOrigin.Y + cameraPosition.Y,
            renderOrigin.Z + cameraPosition.Z);
        return true;
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

    /// <summary>
    /// Counts the verified clusters the plan accepted, which is the only set that can become a
    /// prepared frame. Every accepted input is prepared, so the difference between this count and
    /// the prepared count is a dropped draw rather than a selection decision: a cluster the plan
    /// evaluated and did not accept is culled work - it is invisible, beyond its LOD distance, or
    /// outside the batch and instance budget - and the validation record publishes the two sets
    /// apart so the dropped count stays a defect signal in a world that streams more than one cell.
    /// </summary>
    private int CountAcceptedCullingInputs()
    {
        ReadOnlySpan<VegetationCullingSelection> selections = new(
            m_CullingSelections,
            0,
            m_CullingSelectionCount);
        int accepted = 0;
        for (int inputIndex = 0; inputIndex < m_CullingInputCount; inputIndex++)
        {
            if (VegetationClusterLookup.TryFindSelection(
                    selections,
                    m_CullingInputs[inputIndex].Resident.Guid,
                    out VegetationCullingSelection selection) &&
                selection.Accepted)
            {
                accepted++;
            }
        }

        return accepted;
    }

    private void LogSubmittedDrawValidation(
        in GenericRenderPipelineFeatureSubmissionContext context,
        int opaqueBatches,
        long opaqueInstances,
        int shadowBatches,
        long shadowInstances)
    {
        int loggedClusterCount = CollectLoggedClusters(out int clustersOverflow);
        Guid clusterGuid = loggedClusterCount > 0
            ? m_LoggedClusterGuids[0]
            : Guid.Empty;
        Guid speciesGuid = loggedClusterCount > 0
            ? m_LoggedSpeciesGuids[0]
            : Guid.Empty;
        m_LoggedClusterList.Clear();
        for (int index = 0; index < loggedClusterCount; index++)
        {
            if (index > 0)
            {
                m_LoggedClusterList.Append(',');
            }

            m_LoggedClusterList
                .Append(m_LoggedClusterGuids[index].ToString("N"))
                .Append(':')
                .Append(m_LoggedSpeciesGuids[index].ToString("N"))
                .Append(':')
                .Append(m_LoggedClusterInstances[index]);
        }

        KernelLog.InfoFormat(
            "[Vegetation.GenericRP.Validation] Surface=0x{0:X} Frame={1} " +
            "DeviceGeneration={2} Revision={3} Extracted={4} CullingInputs={5} " +
            "PreparedClusters={6} Cluster={7:D} Species={8:D} Clusters={9} " +
            "ClustersOverflow={10} OpaqueBatches={11} OpaqueInstances={12} " +
            "RecordedShadowBatches={13} RecordedShadowInstances={14} Cascades={15} " +
            "ShadowBatches={16},{17},{18},{19} ShadowInstances={20},{21},{22},{23} " +
            "Dropped={24} Ticket={25}",
            context.Frame.RenderContext.SurfaceId,
            context.Frame.RenderContext.FrameIndex,
            context.Frame.RenderContext.DeviceGeneration,
            m_RuntimeSnapshot.Revision,
            m_ExtractedClusterCount,
            m_CullingInputCount,
            m_PreparedClusterCount,
            clusterGuid,
            speciesGuid,
            m_LoggedClusterList.ToString(),
            clustersOverflow,
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

    // Validation records stay bounded and canonical: at most MaximumLoggedClusters prepared
    // clusters are reported in ascending Guid order and every omitted cluster is counted in the
    // overflow field, so the record never silently truncates the submitted draw set.
    private int CollectLoggedClusters(out int clustersOverflow)
    {
        int loggedCount = 0;
        int overflow = 0;
        for (int index = 0; index < m_PreparedClusterCount; index++)
        {
            VegetationPreparedClusterView prepared = m_PreparedClusters[index].Prepared;
            Guid candidateCluster = prepared.ClusterGuid;
            Guid candidateSpecies = prepared.Batches.Length > 0
                ? prepared.Batches[0].SpeciesGuid
                : Guid.Empty;
            long candidateInstances = prepared.InstanceCount;

            int insertion = 0;
            while (insertion < loggedCount &&
                m_LoggedClusterGuids[insertion].CompareTo(candidateCluster) <= 0)
            {
                insertion++;
            }

            if (insertion >= MaximumLoggedClusters)
            {
                overflow++;
                continue;
            }

            if (loggedCount < MaximumLoggedClusters)
            {
                loggedCount++;
            }
            else
            {
                overflow++;
            }

            for (int move = loggedCount - 1; move > insertion; move--)
            {
                m_LoggedClusterGuids[move] = m_LoggedClusterGuids[move - 1];
                m_LoggedSpeciesGuids[move] = m_LoggedSpeciesGuids[move - 1];
                m_LoggedClusterInstances[move] = m_LoggedClusterInstances[move - 1];
            }

            m_LoggedClusterGuids[insertion] = candidateCluster;
            m_LoggedSpeciesGuids[insertion] = candidateSpecies;
            m_LoggedClusterInstances[insertion] = candidateInstances;
        }

        clustersOverflow = overflow;
        return loggedCount;
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
