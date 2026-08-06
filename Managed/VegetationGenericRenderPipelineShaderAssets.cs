using Arisen.Native.RHI;
using ArisenEngine.Rendering;

namespace ArisenEngine.Vegetation.GenericRenderPipeline;

internal static class VegetationGenericRenderPipelineShaderAssets
{
    public static readonly Guid VegetationShaderGuid =
        Guid.Parse("2a536b1f-81cf-4d91-a84f-39bc6f7e15a2");
    public static readonly Guid VegetationShadowShaderGuid =
        Guid.Parse("9d7a4c3e-f2b6-46a1-8c59-5e1087b34d20");

    public static ShaderAsset CreateVegetation()
    {
        return new ShaderAsset(
            VegetationShaderGuid,
            "VegetationGenericRP/Vegetation",
            [
                new ShaderStageAsset("Vertex", EProgramStage.Vertex, "VSMain"),
                new ShaderStageAsset("Fragment", EProgramStage.Fragment, "PSMain")
            ],
            ShaderVariantKey.VulkanDebug);
    }

    public static ShaderAsset CreateVegetationShadow()
    {
        return new ShaderAsset(
            VegetationShadowShaderGuid,
            "VegetationGenericRP/DirectionalShadow",
            [new ShaderStageAsset("Vertex", EProgramStage.Vertex, "VSMain")],
            ShaderVariantKey.VulkanDebug);
    }

    public static ShaderAsset[] CreateRuntimeShaders() =>
        [CreateVegetation(), CreateVegetationShadow()];
}
