using ArisenEngine.Core.Assets;
using ArisenEngine.Rendering;
using ArisenEngine.Resources.Serialization;
using ArisenEngine.Threading;
using ArisenKernel.Packages;
using ArisenKernel.Services;

namespace ArisenEngine.Vegetation.GenericRenderPipeline;

public sealed class VegetationGenericRenderPipelinePackage : IPackageEntry
{
    private IGenericRenderPipelineFeatureRegistry? m_FeatureRegistry;
    private IRuntimeAssetResidencyService? m_ResidencyService;
    private VegetationPreparedAssetProvider? m_PreparedAssets;
    private VegetationGenericRenderPipelineFeature? m_Feature;
    private bool m_FeatureRegistered;
    private bool m_FeatureResourcesReleased;
    private bool m_PreparedProviderRegistered;
    private bool m_PreparedAssetsDisposed;

    public bool HasPendingOwnership =>
        m_Feature != null ||
        m_PreparedAssets != null ||
        m_FeatureRegistered ||
        m_PreparedProviderRegistered;

    public void OnLoad(IServiceRegistry services)
    {
        if (HasPendingOwnership)
        {
            throw new InvalidOperationException(
                "Vegetation GenericRP still owns resources from a prior load attempt.");
        }

        m_FeatureRegistry = services.GetService<IGenericRenderPipelineFeatureRegistry>();
        m_ResidencyService = services.GetService<IRuntimeAssetResidencyService>();
        m_FeatureRegistered = false;
        m_FeatureResourcesReleased = false;
        m_PreparedProviderRegistered = false;
        m_PreparedAssetsDisposed = false;

        try
        {
            m_PreparedAssets = new VegetationPreparedAssetProvider(
                services.GetService<IAssetDatabase>(),
                services.GetService<IBackgroundTaskScheduler>(),
                services.GetService<IVegetationRuntimeDataStore>(),
                m_ResidencyService);
            m_Feature = new VegetationGenericRenderPipelineFeature(
                services.GetService<IVegetationClusterDataSource>(),
                services.GetService<IVegetationDiagnosticsPublisher>(),
                services.GetService<IVegetationAuthoringPreviewService>());

            m_FeatureRegistered = true;
            m_FeatureRegistry.Register(m_Feature);

            m_PreparedProviderRegistered = true;
            m_ResidencyService.RegisterPreparedProvider(m_PreparedAssets);
        }
        catch (Exception loadError)
        {
            var cleanupFailures = new List<Exception>();
            CleanupOwnedState(cleanupFailures);
            if (cleanupFailures.Count != 0)
            {
                cleanupFailures.Insert(0, loadError);
                throw new AggregateException(
                    "Vegetation GenericRP load failed and ownership rollback reported " +
                    "additional errors.",
                    cleanupFailures);
            }

            throw;
        }
    }

    public void OnUnload(IServiceRegistry services)
    {
        var failures = new List<Exception>();
        CleanupOwnedState(failures);
        if (failures.Count != 0)
        {
            throw new AggregateException(
                "Vegetation GenericRP package teardown failed.",
                failures);
        }
    }

    private void CleanupOwnedState(ICollection<Exception> failures)
    {
        if (m_Feature != null && !m_FeatureResourcesReleased)
        {
            AttemptCleanup(
                "render-feature device-resource release",
                m_Feature.ReleaseDeviceResources,
                () => m_FeatureResourcesReleased = true,
                failures);
        }

        if (m_FeatureRegistered && m_FeatureRegistry != null && m_Feature != null)
        {
            AttemptCleanup(
                "render-feature unregister",
                () =>
                {
                    if (m_FeatureRegistry.IsRegistered(m_Feature))
                    {
                        m_FeatureRegistry.Unregister(m_Feature);
                    }
                },
                () => m_FeatureRegistered = false,
                failures);
        }

        bool preparedProviderUnregisterFailed = false;
        if (m_PreparedProviderRegistered &&
            m_ResidencyService != null &&
            m_PreparedAssets != null)
        {
            try
            {
                if (m_ResidencyService.IsPreparedProviderRegistered(m_PreparedAssets))
                {
                    m_ResidencyService.UnregisterPreparedProvider(m_PreparedAssets.ProviderId);
                }

                m_PreparedProviderRegistered = false;
            }
            catch (Exception ex)
            {
                preparedProviderUnregisterFailed = true;
                failures.Add(new InvalidOperationException(
                    "Vegetation GenericRP failed to complete prepared-provider unregister.",
                    ex));
            }
        }

        if (m_PreparedAssets != null &&
            !m_PreparedAssetsDisposed)
        {
            AttemptCleanup(
                "prepared-provider disposal",
                m_PreparedAssets.Dispose,
                () => m_PreparedAssetsDisposed = true,
                failures);
        }

        if (preparedProviderUnregisterFailed &&
            m_PreparedProviderRegistered &&
            m_PreparedAssetsDisposed &&
            m_ResidencyService != null &&
            m_PreparedAssets != null)
        {
            AttemptCleanup(
                "prepared-provider unregister after disposal",
                () =>
                {
                    if (m_ResidencyService.IsPreparedProviderRegistered(m_PreparedAssets))
                    {
                        m_ResidencyService.UnregisterPreparedProvider(m_PreparedAssets.ProviderId);
                    }
                },
                () => m_PreparedProviderRegistered = false,
                failures);
        }

        if (m_FeatureResourcesReleased && !m_FeatureRegistered)
        {
            m_Feature = null;
            m_FeatureRegistry = null;
        }

        if (m_PreparedAssetsDisposed && !m_PreparedProviderRegistered)
        {
            m_PreparedAssets = null;
            m_ResidencyService = null;
        }

        if (m_Feature == null)
        {
            m_FeatureRegistry = null;
        }

        if (m_PreparedAssets == null)
        {
            m_ResidencyService = null;
        }
    }

    private static void AttemptCleanup(
        string stage,
        Action cleanup,
        Action releaseOwnership,
        ICollection<Exception> failures)
    {
        try
        {
            cleanup();
            releaseOwnership();
        }
        catch (Exception ex)
        {
            failures.Add(new InvalidOperationException(
                $"Vegetation GenericRP failed to complete {stage}.",
                ex));
        }
    }
}
