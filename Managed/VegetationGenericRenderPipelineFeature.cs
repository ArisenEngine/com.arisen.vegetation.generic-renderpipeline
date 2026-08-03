using ArisenEngine.Rendering;

namespace ArisenEngine.Vegetation.GenericRenderPipeline;

internal sealed class VegetationGenericRenderPipelineFeature : IGenericRenderPipelineFeature
{
    public const string Id = "com.arisen.vegetation.generic-renderpipeline";

    private readonly IVegetationClusterDataSource m_ClusterData;
    private readonly IVegetationDiagnosticsPublisher m_Diagnostics;
    private readonly IVegetationAuthoringPreviewService m_AuthoringPreviews;
    private VegetationClusterDataSnapshot m_RuntimeSnapshot =
        VegetationClusterDataSnapshot.Empty;
    private VegetationAuthoringPreviewSnapshot m_PreviewSnapshot =
        VegetationAuthoringPreviewSnapshot.Empty;
    private bool m_DeviceResourcesReleased;

    public VegetationGenericRenderPipelineFeature(
        IVegetationClusterDataSource clusterData,
        IVegetationDiagnosticsPublisher diagnostics,
        IVegetationAuthoringPreviewService authoringPreviews)
    {
        m_ClusterData = clusterData ?? throw new ArgumentNullException(nameof(clusterData));
        m_Diagnostics = diagnostics ?? throw new ArgumentNullException(nameof(diagnostics));
        m_AuthoringPreviews = authoringPreviews
            ?? throw new ArgumentNullException(nameof(authoringPreviews));
    }

    public string FeatureId => Id;

    public int Order => 200;

    public void ConsumeExtractedFrame(in GenericRenderPipelineFeatureFrameContext context)
    {
        m_RuntimeSnapshot = m_ClusterData.GetSnapshot();
        m_PreviewSnapshot = m_AuthoringPreviews.GetSnapshot();
    }

    public void PrepareResources(in GenericRenderPipelineFeatureFrameContext context)
    {
        m_DeviceResourcesReleased = false;
    }

    public void AddRenderGraphPasses(
        GenericRenderPipelineFeatureGraphStage stage,
        in GenericRenderPipelineFeatureGraphContext context)
    {
    }

    public void OnFrameSubmitted(in GenericRenderPipelineFeatureSubmissionContext context)
    {
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

        m_RuntimeSnapshot = VegetationClusterDataSnapshot.Empty;
        m_PreviewSnapshot = VegetationAuthoringPreviewSnapshot.Empty;
        m_Diagnostics.Clear();
        m_DeviceResourcesReleased = true;
    }
}
