[[vk::binding(0, 3)]]
Texture2D<float4> BindlessImages[] : register(t0, space3);

[[vk::binding(1, 3)]]
SamplerState BindlessSamplers[] : register(s0, space3);

[[vk::binding(2, 3)]]
ByteAddressBuffer BindlessBuffers[] : register(t2, space3);

// @arisen.material.texture2d BaseColor
// @arisen.material.texture2dvariant BaseColor r8g8b8a8unorm.srgb.mips

[[vk::push_constant]]
struct
{
    float4 viewProjectionColumn0;
    float4 viewProjectionColumn1;
    float4 viewProjectionColumn2;
    float4 viewProjectionColumn3;
    float4 clusterOriginInstanceBuffer;
    float4 materialParameters;
    // x = material flag bits, y = species wind stiffness,
    // z = distance fade distance, w = vegetation wind frame buffer index.
    float4 materialFlags;
} DrawConstants;

struct VSInput
{
    float3 Position : POSITION0;
    float2 UV : TEXCOORD0;
    uint InstanceIndex : SV_InstanceID;
};

struct VSOutput
{
    float4 Position : SV_Position;
    float2 UV : TEXCOORD0;
    nointerpolation float2 FadeState : TEXCOORD1;
};

struct VegetationInstance
{
    float3 Position;
    float Scale;
    float4 Orientation;
    uint StableVariation;
    float WindPhase;
};

static const uint VEGETATION_INSTANCE_STRIDE = 48;
static const uint VEGETATION_FLAG_ALPHA_TEST = 1u;
// The vegetation passes render in the view frame: vertex positions are relative to the render
// camera, the cascade and frame view-projections are rotation-only, and the camera vector below is
// the render camera expressed in that same frame. Nothing here reads the rebaseable world origin,
// which is what keeps the shadow map identical across an origin rebase.
static const uint VEGETATION_WIND_DIRECTION_STRENGTH_VECTOR = 7u;
static const uint VEGETATION_WIND_GUST_PARAMETERS_VECTOR = 8u;
static const uint VEGETATION_WIND_CAMERA_VECTOR = 4u;
static const float VEGETATION_GUST_SPATIAL_FREQUENCY = 0.35;
static const float VEGETATION_FADE_FRACTION = 0.15;
static const float VEGETATION_INSTANCE_VARIATION_SCALE = 1.0 / 16777215.0;
static const float VEGETATION_GUST_PRIMARY_WEIGHT = 0.6;
static const float VEGETATION_GUST_SECONDARY_WEIGHT = 0.4;
static const float VEGETATION_GUST_SECONDARY_PHASE_RATIO = 1.618;
static const float VEGETATION_GUST_SPATIAL_SECONDARY_SCALE = 0.5;

float4 NormalizeQuaternion(float4 value)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 0.000001
        ? value * rsqrt(lengthSquared)
        : float4(0.0, 0.0, 0.0, 1.0);
}

float3 RotateByQuaternion(float3 value, float4 orientation)
{
    float4 normalized = NormalizeQuaternion(orientation);
    float3 twiceCross = 2.0 * cross(normalized.xyz, value);
    return value + normalized.w * twiceCross +
        cross(normalized.xyz, twiceCross);
}

float4 LoadFrameVector(uint vectorIndex)
{
    uint frameBufferIndex = asuint(DrawConstants.materialFlags.w);
    return asfloat(BindlessBuffers[
        NonUniformResourceIndex(frameBufferIndex)].Load4(vectorIndex * 16));
}

VegetationInstance LoadVegetationInstance(uint instanceIndex)
{
    uint instanceBufferIndex = asuint(
        DrawConstants.clusterOriginInstanceBuffer.w);
    uint byteOffset = instanceIndex * VEGETATION_INSTANCE_STRIDE;
    ByteAddressBuffer instanceBuffer = BindlessBuffers[
        NonUniformResourceIndex(instanceBufferIndex)];
    float4 positionScale = asfloat(instanceBuffer.Load4(byteOffset));
    float4 variation = asfloat(instanceBuffer.Load4(byteOffset + 32));

    VegetationInstance result;
    result.Position = positionScale.xyz;
    result.Scale = positionScale.w;
    result.Orientation = asfloat(instanceBuffer.Load4(byteOffset + 16));
    result.StableVariation = asuint(variation.x);
    result.WindPhase = variation.y;
    return result;
}

float ComputeVegetationWindGust(float3 worldPosition, float windPhase)
{
    float4 windDirectionStrength = LoadFrameVector(
        VEGETATION_WIND_DIRECTION_STRENGTH_VECTOR);
    float4 windGustParameters = LoadFrameVector(
        VEGETATION_WIND_GUST_PARAMETERS_VECTOR);
    float spatial = dot(
        windDirectionStrength.xz,
        worldPosition.xz) * VEGETATION_GUST_SPATIAL_FREQUENCY;
    float primaryWave = 0.5 + 0.5 * sin(
        windGustParameters.y + windPhase + spatial);
    float secondaryWave = 0.5 + 0.5 * sin(
        windGustParameters.z +
        windPhase * VEGETATION_GUST_SECONDARY_PHASE_RATIO +
        spatial * VEGETATION_GUST_SPATIAL_SECONDARY_SCALE);
    return 1.0 + windGustParameters.x * (
        VEGETATION_GUST_PRIMARY_WEIGHT * primaryWave +
        VEGETATION_GUST_SECONDARY_WEIGHT * secondaryWave);
}

float3 ComputeVegetationWindOffset(
    float3 worldPosition,
    float localHeight,
    float windPhase)
{
    float4 windDirectionStrength = LoadFrameVector(
        VEGETATION_WIND_DIRECTION_STRENGTH_VECTOR);
    float height = max(localHeight, 0.0);
    if (height <= 0.0 || windDirectionStrength.w <= 0.0)
    {
        return float3(0.0, 0.0, 0.0);
    }

    float4 windGustParameters = LoadFrameVector(
        VEGETATION_WIND_GUST_PARAMETERS_VECTOR);
    float lean = saturate(
        windGustParameters.w *
        DrawConstants.materialFlags.y *
        ComputeVegetationWindGust(worldPosition, windPhase));
    return float3(windDirectionStrength.x, 0.0, windDirectionStrength.z) *
        (lean * height) -
        float3(0.0, height * (1.0 - sqrt(max(1.0 - lean * lean, 0.0))), 0.0);
}

float2 ComputeVegetationFadeState(
    float3 worldPosition,
    uint stableVariation)
{
    float threshold =
        float(stableVariation & 0x00ffffffu) * VEGETATION_INSTANCE_VARIATION_SCALE;
    float fadeDistance = DrawConstants.materialFlags.z;
    if (!(fadeDistance > 0.0) || fadeDistance >= 3.0e30)
    {
        return float2(1.0, threshold);
    }

    float3 cameraPosition = LoadFrameVector(
        VEGETATION_WIND_CAMERA_VECTOR).xyz;
    float distanceToCamera = length(worldPosition - cameraPosition);
    float fadeStart = fadeDistance * (1.0 - VEGETATION_FADE_FRACTION);
    float fadeRange = max(fadeDistance - fadeStart, 0.0001);
    return float2(
        saturate((fadeDistance - distanceToCamera) / fadeRange),
        threshold);
}

VSOutput VSMain(VSInput input)
{
    VegetationInstance instance = LoadVegetationInstance(input.InstanceIndex);
    float3 scaledLocalPosition = input.Position * instance.Scale;
    // View-frame instance origin. The pushed cluster origin is already relative to the render
    // camera, so wind, fade, and clip all read the same undisplaced position the opaque pass uses.
    float3 instanceOrigin = DrawConstants.clusterOriginInstanceBuffer.xyz +
        instance.Position;
    float3 position = instanceOrigin +
        RotateByQuaternion(
            scaledLocalPosition,
            instance.Orientation) +
        ComputeVegetationWindOffset(
            instanceOrigin,
            scaledLocalPosition.y,
            instance.WindPhase);
    float4 shadowPosition = float4(position, 1.0);

    VSOutput output;
    output.Position = float4(
        dot(shadowPosition, DrawConstants.viewProjectionColumn0),
        dot(shadowPosition, DrawConstants.viewProjectionColumn1),
        dot(shadowPosition, DrawConstants.viewProjectionColumn2),
        dot(shadowPosition, DrawConstants.viewProjectionColumn3));
    output.UV = input.UV;
    output.FadeState = ComputeVegetationFadeState(
        instanceOrigin,
        instance.StableVariation);
    return output;
}

void PSMain(VSOutput input)
{
    clip(input.FadeState.x - input.FadeState.y);

    uint materialFlagBits = asuint(DrawConstants.materialFlags.x);
    if ((materialFlagBits & VEGETATION_FLAG_ALPHA_TEST) == 0u)
    {
        return;
    }

    uint baseColorImageIndex = asuint(DrawConstants.materialParameters.y);
    uint baseColorSamplerIndex = asuint(DrawConstants.materialParameters.z);
    float alpha = BindlessImages[
        NonUniformResourceIndex(baseColorImageIndex)].Sample(
            BindlessSamplers[
                NonUniformResourceIndex(baseColorSamplerIndex)],
            input.UV).a * DrawConstants.materialParameters.w;
    clip(alpha - DrawConstants.materialParameters.x);
}
