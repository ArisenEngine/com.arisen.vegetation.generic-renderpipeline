[[vk::binding(0, 3)]]
Texture2D<float4> BindlessImages[] : register(t0, space3);

[[vk::binding(1, 3)]]
SamplerState BindlessSamplers[] : register(s0, space3);

[[vk::binding(2, 3)]]
ByteAddressBuffer BindlessBuffers[] : register(t2, space3);

// @arisen.material.texture2d BaseColor
// @arisen.material.texture2dvariant BaseColor r8g8b8a8unorm.srgb.mips
// @arisen.material.texture2d Normal
// @arisen.material.texture2dvariant Normal r8g8b8a8unorm.linear.mips.normalmap
// @arisen.material.texture2d MetallicRoughness
// @arisen.material.texture2dvariant MetallicRoughness r8g8b8a8unorm.linear.mips

[[vk::push_constant]]
struct
{
    float4 clusterOriginInstanceBuffer;
    float4 baseColorFactor;
    float4 materialParameters;
    float4 normalParameters;
    float4 ormParameters;
    float4 frameShadowParameters;
    float4 windParameters;
} DrawConstants;

struct VSInput
{
    float3 Position : POSITION0;
    float3 Normal : NORMAL0;
    float4 Tangent : TANGENT0;
    float2 UV : TEXCOORD0;
    float3 Color : COLOR0;
    uint InstanceIndex : SV_InstanceID;
};

struct VSOutput
{
    float4 Position : SV_Position;
    float3 WorldPosition : TEXCOORD0;
    float3 WorldNormal : NORMAL0;
    float4 WorldTangent : TANGENT0;
    float2 UV : TEXCOORD1;
    float3 VertexColor : COLOR0;
    float CameraDepth : TEXCOORD2;
    nointerpolation float ColorVariation : TEXCOORD3;
    nointerpolation float2 FadeState : TEXCOORD4;
};

struct VegetationInstance
{
    float3 Position;
    float Scale;
    float4 Orientation;
    uint StableVariation;
    float WindPhase;
    float ColorVariation;
};

static const float PI = 3.14159265359;
static const uint INVALID_BINDLESS_INDEX = 0xffffffffu;
static const uint VEGETATION_INSTANCE_STRIDE = 48;
static const uint VEGETATION_FLAG_ALPHA_TEST = 1u;
static const uint VEGETATION_FLAG_TINT_VARIATION = 2u;
// The vegetation passes render in the view frame: vertex positions are relative to the render
// camera, the frame constants' view-projection is computed from the rotation-only view matrix, and
// the camera vector below is the render camera expressed in that same frame. Nothing here reads the
// rebaseable world origin, which is what keeps depth, coverage, and colour identical across an
// origin rebase.
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

float3 SafeNormalize(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 0.000001
        ? value * rsqrt(lengthSquared)
        : fallback;
}

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

VegetationInstance LoadVegetationInstance(uint instanceIndex)
{
    uint instanceBufferIndex = asuint(
        DrawConstants.clusterOriginInstanceBuffer.w);
    uint byteOffset = instanceIndex * VEGETATION_INSTANCE_STRIDE;
    ByteAddressBuffer instanceBuffer = BindlessBuffers[
        NonUniformResourceIndex(instanceBufferIndex)];
    float4 positionScale = asfloat(instanceBuffer.Load4(byteOffset));

    VegetationInstance result;
    result.Position = positionScale.xyz;
    result.Scale = positionScale.w;
    result.Orientation = asfloat(instanceBuffer.Load4(byteOffset + 16));
    float4 variation = asfloat(instanceBuffer.Load4(byteOffset + 32));
    result.StableVariation = asuint(variation.x);
    result.WindPhase = variation.y;
    result.ColorVariation = variation.z;
    return result;
}

float4 LoadFrameVector(uint vectorIndex)
{
    uint frameBufferIndex = asuint(DrawConstants.frameShadowParameters.x);
    return asfloat(BindlessBuffers[
        NonUniformResourceIndex(frameBufferIndex)].Load4(vectorIndex * 16));
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

// Lean as a fraction of a right angle. Wind strength reaches the shader already scaled by the
// CPU-owned bend constant, and saturate keeps the lean inside a right angle, so a displaced
// vertex never travels further than its own height above the instance origin.
float ComputeVegetationWindLean(float3 worldPosition, float windPhase)
{
    float4 windGustParameters = LoadFrameVector(
        VEGETATION_WIND_GUST_PARAMETERS_VECTOR);
    return saturate(
        windGustParameters.w *
        DrawConstants.windParameters.x *
        ComputeVegetationWindGust(worldPosition, windPhase));
}

// Shear along the wind plus the arc drop that preserves blade length: rotating a point at height
// h by asin(lean) moves it to (h * lean, h * sqrt(1 - lean * lean)).
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

    float lean = ComputeVegetationWindLean(worldPosition, windPhase);
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
    float fadeDistance = DrawConstants.windParameters.y;
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
    float3 rotatedPosition = RotateByQuaternion(
        scaledLocalPosition,
        instance.Orientation);
    float3 instanceOrigin = DrawConstants.clusterOriginInstanceBuffer.xyz +
        instance.Position;
    float3 windOffset = ComputeVegetationWindOffset(
        instanceOrigin,
        scaledLocalPosition.y,
        instance.WindPhase);
    float3 worldPosition = instanceOrigin + rotatedPosition + windOffset;
    float4 position = float4(worldPosition, 1.0);
    float4 viewProjectionColumn0 = LoadFrameVector(0);
    float4 viewProjectionColumn1 = LoadFrameVector(1);
    float4 viewProjectionColumn2 = LoadFrameVector(2);
    float4 viewProjectionColumn3 = LoadFrameVector(3);

    VSOutput output;
    output.Position = float4(
        dot(position, viewProjectionColumn0),
        dot(position, viewProjectionColumn1),
        dot(position, viewProjectionColumn2),
        dot(position, viewProjectionColumn3));
    output.WorldPosition = worldPosition;
    output.WorldNormal = SafeNormalize(
        RotateByQuaternion(input.Normal, instance.Orientation),
        float3(0.0, 1.0, 0.0));
    output.WorldTangent = float4(
        RotateByQuaternion(input.Tangent.xyz, instance.Orientation),
        input.Tangent.w);
    output.UV = input.UV;
    output.VertexColor = input.Color;
    output.CameraDepth = max(output.Position.w, 0.0);
    output.ColorVariation = instance.ColorVariation;
    // Fade is resolved at the undisplaced instance origin so the main pass and the shadow pass
    // dither the same instances for the same frame, and so the coverage never depends on wind.
    output.FadeState = ComputeVegetationFadeState(
        instanceOrigin,
        instance.StableVariation);
    return output;
}

float4 LoadBufferVector(uint bufferIndex, uint vectorIndex)
{
    return asfloat(BindlessBuffers[
        NonUniformResourceIndex(bufferIndex)].Load4(vectorIndex * 16));
}

float DistributionGGX(float3 normal, float3 halfway, float roughness)
{
    float alpha = roughness * roughness;
    float alphaSquared = alpha * alpha;
    float ndoth = saturate(dot(normal, halfway));
    float denominator = ndoth * ndoth * (alphaSquared - 1.0) + 1.0;
    return alphaSquared / max(PI * denominator * denominator, 0.0001);
}

float GeometrySchlickGGX(float ndotDirection, float roughness)
{
    float offsetRoughness = roughness + 1.0;
    float k = (offsetRoughness * offsetRoughness) * 0.125;
    return ndotDirection / max(
        ndotDirection * (1.0 - k) + k,
        0.0001);
}

float GeometrySmith(
    float3 normal,
    float3 viewDirection,
    float3 lightDirection,
    float roughness)
{
    return GeometrySchlickGGX(
        saturate(dot(normal, viewDirection)),
        roughness) * GeometrySchlickGGX(
        saturate(dot(normal, lightDirection)),
        roughness);
}

float3 FresnelSchlick(float cosine, float3 reflectanceAtNormal)
{
    return reflectanceAtNormal +
        (1.0 - reflectanceAtNormal) *
        pow(saturate(1.0 - cosine), 5.0);
}

float SampleDirectionalShadowCascade(
    uint shadowBufferIndex,
    uint cascadeIndex,
    float3 cameraRelativePosition,
    float ndotl)
{
    uint matrixOffset = cascadeIndex * 4;
    float4 position = float4(cameraRelativePosition, 1.0);
    float4 shadowClip = float4(
        dot(position, LoadBufferVector(shadowBufferIndex, matrixOffset + 0)),
        dot(position, LoadBufferVector(shadowBufferIndex, matrixOffset + 1)),
        dot(position, LoadBufferVector(shadowBufferIndex, matrixOffset + 2)),
        dot(position, LoadBufferVector(shadowBufferIndex, matrixOffset + 3)));
    if (shadowClip.w <= 0.0)
    {
        return 1.0;
    }

    float3 projected = shadowClip.xyz / shadowClip.w;
    if (projected.x < -1.0 || projected.x > 1.0 ||
        projected.y < -1.0 || projected.y > 1.0 ||
        projected.z < 0.0 || projected.z > 1.0)
    {
        return 1.0;
    }

    float4 shadowImageIndices = LoadBufferVector(shadowBufferIndex, 18);
    float4 samplingParameters = LoadBufferVector(shadowBufferIndex, 19);
    uint4 metadata = asuint(LoadBufferVector(shadowBufferIndex, 20));
    uint shadowImageIndex = asuint(shadowImageIndices[cascadeIndex]);
    float2 shadowUV = float2(
        projected.x * 0.5 + 0.5,
        0.5 - projected.y * 0.5);
    float bias = max(samplingParameters.x, 0.0) +
        max(samplingParameters.y, 0.0) *
        (1.0 - saturate(ndotl));
    int pcfRadius = clamp((int)metadata.y, 0, 3);
    float visible = 0.0;
    float sampleCount = 0.0;
    [loop]
    for (int y = -pcfRadius; y <= pcfRadius; y++)
    {
        [loop]
        for (int x = -pcfRadius; x <= pcfRadius; x++)
        {
            float2 sampleUV = shadowUV + float2(x, y) *
                max(samplingParameters.w, 0.00001);
            float sampledDepth = BindlessImages[
                NonUniformResourceIndex(shadowImageIndex)].SampleLevel(
                    BindlessSamplers[NonUniformResourceIndex(metadata.x)],
                    sampleUV,
                    0.0).r;
            visible += projected.z - bias <= sampledDepth ? 1.0 : 0.0;
            sampleCount += 1.0;
        }
    }

    visible /= max(sampleCount, 1.0);
    return lerp(
        1.0 - saturate(samplingParameters.z),
        1.0,
        visible);
}

float SampleDirectionalShadow(
    VSOutput input,
    float3 cameraPosition,
    float ndotl)
{
    uint shadowBufferIndex = asuint(
        DrawConstants.frameShadowParameters.y);
    if (shadowBufferIndex == INVALID_BINDLESS_INDEX)
    {
        return 1.0;
    }

    float4 splitFar = LoadBufferVector(shadowBufferIndex, 16);
    float4 transitionStart = LoadBufferVector(shadowBufferIndex, 17);
    uint4 metadata = asuint(LoadBufferVector(shadowBufferIndex, 20));
    float4 distanceParameters = LoadBufferVector(shadowBufferIndex, 21);
    uint cascadeCount = min(metadata.z, 4u);
    if (metadata.w == 0 ||
        cascadeCount == 0 ||
        input.CameraDepth > distanceParameters.y)
    {
        return 1.0;
    }

    uint cascadeIndex = cascadeCount - 1;
    [unroll]
    for (uint index = 0; index < 4; index++)
    {
        if (index < cascadeCount && input.CameraDepth <= splitFar[index])
        {
            cascadeIndex = index;
            break;
        }
    }

    float3 cameraRelativePosition = input.WorldPosition -
        cameraPosition;
    float shadow = SampleDirectionalShadowCascade(
        shadowBufferIndex,
        cascadeIndex,
        cameraRelativePosition,
        ndotl);
    if (cascadeIndex + 1 < cascadeCount &&
        input.CameraDepth > transitionStart[cascadeIndex])
    {
        float transition = saturate(
            (input.CameraDepth - transitionStart[cascadeIndex]) /
            max(
                splitFar[cascadeIndex] - transitionStart[cascadeIndex],
                0.0001));
        shadow = lerp(
            shadow,
            SampleDirectionalShadowCascade(
                shadowBufferIndex,
                cascadeIndex + 1,
                cameraRelativePosition,
                ndotl),
            transition);
    }

    float terminalFade = saturate(
        (input.CameraDepth - distanceParameters.z) /
        max(distanceParameters.y - distanceParameters.z, 0.0001));
    return lerp(shadow, 1.0, terminalFade);
}

float3 ResolveTwoSidedNormal(
    VSOutput input,
    bool isFrontFace,
    float4 sampledNormal,
    out float3 geometricNormal)
{
    geometricNormal = SafeNormalize(
        input.WorldNormal,
        float3(0.0, 1.0, 0.0));
    float3 geometricTangent = SafeNormalize(
        input.WorldTangent.xyz,
        float3(1.0, 0.0, 0.0));
    float tangentSign = input.WorldTangent.w < 0.0 ? -1.0 : 1.0;

    if (!isFrontFace)
    {
        geometricNormal = -geometricNormal;
        geometricTangent = -geometricTangent;
    }

    float3 tangentNormal = float3(
        sampledNormal.xy * 2.0 - 1.0,
        sampledNormal.z * 2.0 - 1.0);
    float3 bitangent = SafeNormalize(
        cross(geometricNormal, geometricTangent) * tangentSign,
        geometricNormal);
    return SafeNormalize(
        geometricTangent * tangentNormal.x +
        bitangent * tangentNormal.y +
        geometricNormal * tangentNormal.z,
        geometricNormal);
}

float4 PSMain(VSOutput input, bool isFrontFace : SV_IsFrontFace) : SV_Target0
{
    clip(input.FadeState.x - input.FadeState.y);

    float4 cameraPosition = LoadFrameVector(4);
    float4 lightDirectionIntensity = LoadFrameVector(5);
    float4 lightColorAmbient = LoadFrameVector(6);
    uint materialFlags = asuint(DrawConstants.frameShadowParameters.z);
    uint baseColorImageIndex = asuint(DrawConstants.materialParameters.z);
    uint baseColorSamplerIndex = asuint(DrawConstants.materialParameters.w);
    float4 sampledColor = BindlessImages[
        NonUniformResourceIndex(baseColorImageIndex)].Sample(
            BindlessSamplers[
                NonUniformResourceIndex(baseColorSamplerIndex)],
            input.UV);

    float alphaCoverage = saturate(
        sampledColor.a * DrawConstants.normalParameters.w);
    if ((materialFlags & VEGETATION_FLAG_ALPHA_TEST) != 0u &&
        alphaCoverage < DrawConstants.baseColorFactor.a)
    {
        discard;
    }

    uint normalImageIndex = asuint(DrawConstants.normalParameters.x);
    uint normalSamplerIndex = asuint(DrawConstants.normalParameters.y);
    float4 sampledNormal = BindlessImages[
        NonUniformResourceIndex(normalImageIndex)].Sample(
            BindlessSamplers[
                NonUniformResourceIndex(normalSamplerIndex)],
            input.UV);
    float3 geometricNormal;
    float3 normal = ResolveTwoSidedNormal(
        input,
        isFrontFace,
        sampledNormal,
        geometricNormal);

    uint ormImageIndex = asuint(DrawConstants.ormParameters.x);
    uint ormSamplerIndex = asuint(DrawConstants.ormParameters.y);
    float4 sampledOrm = BindlessImages[
        NonUniformResourceIndex(ormImageIndex)].Sample(
            BindlessSamplers[
                NonUniformResourceIndex(ormSamplerIndex)],
            input.UV);
    float metallic = saturate(
        DrawConstants.materialParameters.x * sampledOrm.b);
    float roughness = clamp(
        DrawConstants.materialParameters.y * max(sampledOrm.g, 0.04),
        0.08,
        1.0);
    float occlusion = lerp(
        1.0,
        saturate(sampledOrm.r),
        saturate(DrawConstants.ormParameters.z));

    float3 albedo = max(
        sampledColor.rgb *
        DrawConstants.baseColorFactor.rgb *
        input.VertexColor,
        0.0);
    if ((materialFlags & VEGETATION_FLAG_TINT_VARIATION) != 0u)
    {
        albedo *= lerp(
            1.0,
            input.ColorVariation,
            saturate(DrawConstants.normalParameters.z));
    }

    float3 viewDirection = SafeNormalize(
        cameraPosition.xyz - input.WorldPosition,
        normal);
    float3 lightDirection = SafeNormalize(
        lightDirectionIntensity.xyz,
        normal);
    float3 halfway = SafeNormalize(
        lightDirection + viewDirection,
        normal);
    float ndotl = saturate(dot(normal, lightDirection));
    float ndotv = saturate(dot(normal, viewDirection));
    float3 reflectanceAtNormal = lerp(
        float3(0.04, 0.04, 0.04),
        albedo,
        metallic);
    float distribution = DistributionGGX(normal, halfway, roughness);
    float geometry = GeometrySmith(
        normal,
        viewDirection,
        lightDirection,
        roughness);
    float3 fresnel = FresnelSchlick(
        saturate(dot(halfway, viewDirection)),
        reflectanceAtNormal);
    float3 specular = distribution * geometry * fresnel /
        max(4.0 * ndotv * ndotl, 0.0001);
    float3 diffuseWeight = (1.0 - fresnel) * (1.0 - metallic);
    float3 radiance = lightColorAmbient.rgb *
        max(lightDirectionIntensity.w, 0.0);
    float3 direct = (diffuseWeight * albedo / PI + specular) *
        radiance * ndotl;
    direct *= SampleDirectionalShadow(input, cameraPosition.xyz, ndotl);
    float3 ambient = albedo * (1.0 - metallic) *
        max(lightColorAmbient.w, 0.0) * occlusion;
    return float4(max(direct + ambient, 0.0), 1.0);
}
