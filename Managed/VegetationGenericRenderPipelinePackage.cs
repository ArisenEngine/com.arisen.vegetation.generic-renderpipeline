using ArisenEngine.Core.Assets;
using ArisenEngine.Rendering;
using ArisenEngine.Resources.Serialization;
using ArisenEngine.Threading;
using ArisenKernel.Packages;
using ArisenKernel.Services;
using ArisenKernel.Diagnostics;

namespace ArisenEngine.Vegetation.GenericRenderPipeline;

public sealed class VegetationGenericRenderPipelinePackage : IPackageEntry
{
    private IGenericRenderPipelineFeatureRegistry? m_FeatureRegistry;
    private IGenericRenderPipelineRuntimeShaderRegistry? m_ShaderRegistry;
    private IRuntimeAssetResidencyService? m_ResidencyService;
    private VegetationPreparedAssetProvider? m_PreparedAssets;
    private VegetationGenericRenderPipelineFeature? m_Feature;
    private bool m_FeatureRegistered;
    private bool m_ShaderRegistered;
    private bool m_FeatureResourcesReleased;
    private bool m_PreparedProviderRegistered;
    private bool m_PreparedAssetsDisposed;

    public bool HasPendingOwnership =>
        m_Feature != null ||
        m_PreparedAssets != null ||
        m_FeatureRegistered ||
        m_ShaderRegistered ||
        m_PreparedProviderRegistered;

    public void OnLoad(IServiceRegistry services)
    {
        if (HasPendingOwnership)
        {
            throw new InvalidOperationException(
                "Vegetation GenericRP still owns resources from a prior load attempt.");
        }

        m_FeatureRegistry = services.GetService<IGenericRenderPipelineFeatureRegistry>();
        m_ShaderRegistry =
            services.GetService<IGenericRenderPipelineRuntimeShaderRegistry>();
        m_ResidencyService = services.GetService<IRuntimeAssetResidencyService>();
        m_FeatureRegistered = false;
        m_ShaderRegistered = false;
        m_FeatureResourcesReleased = false;
        m_PreparedProviderRegistered = false;
        m_PreparedAssetsDisposed = false;

        try
        {
            IAssetDatabase assetDatabase = services.GetService<IAssetDatabase>();
            VegetationRenderValidationMode validationMode =
                VegetationRenderValidationPolicy.ResolveFromEnvironment();
            if (validationMode != VegetationRenderValidationMode.Full)
            {
                KernelLog.InfoFormat(
                    "[Vegetation.GenericRP.VisualValidation] Mode={0}",
                    validationMode);
            }
            var gpuResources = new VegetationGpuResourceFactory(
                services.GetService<IGenericRenderPipelinePreparedAssetSource>());
            m_PreparedAssets = new VegetationPreparedAssetProvider(
                assetDatabase,
                services.GetService<IBackgroundTaskScheduler>(),
                services.GetService<IVegetationRuntimeDataStore>(),
                m_ResidencyService,
                gpuResources);
            m_Feature = new VegetationGenericRenderPipelineFeature(
                services.GetService<IVegetationClusterRenderSource>(),
                services.GetService<IVegetationClusterDataSource>(),
                services.GetService<IVegetationDiagnosticsPublisher>(),
                services.GetService<IVegetationAuthoringPreviewService>(),
                m_PreparedAssets,
                new VegetationOpaquePass(assetDatabase),
                new VegetationShadowPass(assetDatabase),
                validationMode);

            m_ShaderRegistry.RegisterRuntimeShaders(
                VegetationGenericRenderPipelineFeature.Id,
                VegetationGenericRenderPipelineShaderAssets.CreateRuntimeShaders());
            m_ShaderRegistered = true;

            try
            {
                m_FeatureRegistry.Register(m_Feature);
                m_FeatureRegistered = true;
            }
            catch
            {
                m_FeatureRegistered = m_FeatureRegistry.IsRegistered(m_Feature);
                throw;
            }

            try
            {
                m_ResidencyService.RegisterPreparedProvider(m_PreparedAssets);
                UpdatePreparedProviderOwnership();
                if (!m_PreparedProviderRegistered)
                {
                    throw new InvalidOperationException(
                        "Vegetation prepared-provider registration returned without " +
                        "registering the package-owned instance.");
                }
            }
            catch
            {
                UpdatePreparedProviderOwnership();
                throw;
            }
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
                UnregisterOwnedPreparedProvider();
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
                UnregisterOwnedPreparedProvider,
                static () => { },
                failures);
        }

        if (m_ShaderRegistered && m_ShaderRegistry != null)
        {
            AttemptCleanup(
                "runtime-shader unregister",
                () => m_ShaderRegistry.UnregisterRuntimeShaders(
                    VegetationGenericRenderPipelineFeature.Id),
                () => m_ShaderRegistered = false,
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

        if (!m_ShaderRegistered)
        {
            m_ShaderRegistry = null;
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

    private void UnregisterOwnedPreparedProvider()
    {
        if (m_ResidencyService == null || m_PreparedAssets == null)
        {
            m_PreparedProviderRegistered = false;
            return;
        }

        UpdatePreparedProviderOwnership();
        if (!m_PreparedProviderRegistered)
        {
            return;
        }

        try
        {
            m_ResidencyService.UnregisterPreparedProvider(m_PreparedAssets.ProviderId);
        }
        finally
        {
            UpdatePreparedProviderOwnership();
        }

        if (m_PreparedProviderRegistered)
        {
            throw new InvalidOperationException(
                "Vegetation prepared-provider unregister returned while the package-owned " +
                "instance remained registered.");
        }
    }

    private void UpdatePreparedProviderOwnership()
    {
        bool ownsRegistration =
            m_ResidencyService != null &&
            m_PreparedAssets != null &&
            m_ResidencyService.IsPreparedProviderRegistered(m_PreparedAssets);
        m_PreparedProviderRegistered = ownsRegistration;
        m_PreparedAssets?.SetResidencyRegistrationOwned(ownsRegistration);
    }
}
