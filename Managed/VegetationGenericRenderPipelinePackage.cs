using ArisenEngine.Rendering;
using ArisenKernel.Diagnostics;
using ArisenKernel.Packages;
using ArisenKernel.Services;

namespace ArisenEngine.Vegetation.GenericRenderPipeline;

public sealed class VegetationGenericRenderPipelinePackage : IPackageEntry
{
    private IGenericRenderPipelineFeatureRegistry? m_FeatureRegistry;
    private VegetationGenericRenderPipelineFeature? m_Feature;

    public void OnLoad(IServiceRegistry services)
    {
        var featureRegistry = services.GetService<IGenericRenderPipelineFeatureRegistry>();
        var feature = new VegetationGenericRenderPipelineFeature(
            services.GetService<IVegetationClusterDataSource>(),
            services.GetService<IVegetationDiagnosticsPublisher>(),
            services.GetService<IVegetationAuthoringPreviewService>());

        try
        {
            featureRegistry.Register(feature);
        }
        catch
        {
            feature.ReleaseDeviceResources();
            throw;
        }

        m_FeatureRegistry = featureRegistry;
        m_Feature = feature;
        KernelLog.Info("[Vegetation.GenericRP] Render feature registered.");
    }

    public void OnUnload(IServiceRegistry services)
    {
        m_Feature?.ReleaseDeviceResources();
        if (m_FeatureRegistry != null && m_Feature != null)
        {
            m_FeatureRegistry.Unregister(m_Feature);
        }

        m_Feature = null;
        m_FeatureRegistry = null;
        KernelLog.Info("[Vegetation.GenericRP] Render feature unregistered.");
    }
}
