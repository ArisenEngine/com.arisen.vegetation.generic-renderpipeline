[[vk::binding(0, 3)]]
Texture2D<float4> BindlessImages[] : register(t0, space3);

[[vk::binding(1, 3)]]
SamplerState BindlessSamplers[] : register(s0, space3);

[[vk::binding(2, 3)]]
ByteAddressBuffer BindlessBuffers[] : register(t2, space3);

[[vk::push_constant]]
struct
{
    float4 clusterOriginInstanceBuffer;
    float4 baseColorFactor;
    float4 materialParameters;
    float4 frameShadowParameters;
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
    float2 UV : TEXCOORD1;
    float3 VertexColor : COLOR0;
    float CameraDepth : TEXCOORD2;
};

struct VegetationInstance
{
    float3 Position;
    float Scale;
    float4 Orientation;
};

static const float PI = 3.14159265359;
static const uint INVALID_BINDLESS_INDEX = 0xffffffffu;
static const uint VEGETATION_INSTANCE_STRIDE = 48;

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
    return result;
}

float4 LoadFrameVector(uint vectorIndex)
{
    uint frameBufferIndex = asuint(DrawConstants.frameShadowParameters.x);
    return asfloat(BindlessBuffers[
        NonUniformResourceIndex(frameBufferIndex)].Load4(vectorIndex * 16));
}

VSOutput VSMain(VSInput input)
{
    VegetationInstance instance = LoadVegetationInstance(input.InstanceIndex);
    float3 rotatedPosition = RotateByQuaternion(
        input.Position * instance.Scale,
        instance.Orientation);
    float3 worldPosition = DrawConstants.clusterOriginInstanceBuffer.xyz +
        instance.Position + rotatedPosition;
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
    output.UV = input.UV;
    output.VertexColor = input.Color;
    output.CameraDepth = max(output.Position.w, 0.0);
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

float4 PSMain(VSOutput input) : SV_Target0
{
    float4 cameraPosition = LoadFrameVector(4);
    float4 lightDirectionIntensity = LoadFrameVector(5);
    float4 lightColorAmbient = LoadFrameVector(6);
    uint baseColorImageIndex = asuint(DrawConstants.materialParameters.z);
    uint baseColorSamplerIndex = asuint(DrawConstants.materialParameters.w);
    float4 sampledColor = BindlessImages[
        NonUniformResourceIndex(baseColorImageIndex)].Sample(
            BindlessSamplers[
                NonUniformResourceIndex(baseColorSamplerIndex)],
            input.UV);
    float3 albedo = max(
        sampledColor.rgb *
        DrawConstants.baseColorFactor.rgb *
        input.VertexColor,
        0.0);
    float metallic = saturate(DrawConstants.materialParameters.x);
    float roughness = clamp(
        DrawConstants.materialParameters.y,
        0.08,
        1.0);
    float3 normal = SafeNormalize(
        input.WorldNormal,
        float3(0.0, 1.0, 0.0));
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
        max(lightColorAmbient.w, 0.0);
    return float4(max(direct + ambient, 0.0), 1.0);
}
