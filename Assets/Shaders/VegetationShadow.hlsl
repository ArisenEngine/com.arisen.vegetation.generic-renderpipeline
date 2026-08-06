[[vk::binding(2, 3)]]
ByteAddressBuffer BindlessBuffers[] : register(t2, space3);

[[vk::push_constant]]
struct
{
    float4 viewProjectionColumn0;
    float4 viewProjectionColumn1;
    float4 viewProjectionColumn2;
    float4 viewProjectionColumn3;
    float4 clusterOriginInstanceBuffer;
} DrawConstants;

struct VSInput
{
    float3 Position : POSITION0;
    uint InstanceIndex : SV_InstanceID;
};

struct VegetationInstance
{
    float3 Position;
    float Scale;
    float4 Orientation;
};

static const uint VEGETATION_INSTANCE_STRIDE = 48;

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

float4 VSMain(VSInput input) : SV_Position
{
    VegetationInstance instance = LoadVegetationInstance(input.InstanceIndex);
    float3 position = DrawConstants.clusterOriginInstanceBuffer.xyz +
        instance.Position + RotateByQuaternion(
            input.Position * instance.Scale,
            instance.Orientation);
    float4 worldPosition = float4(position, 1.0);
    return float4(
        dot(worldPosition, DrawConstants.viewProjectionColumn0),
        dot(worldPosition, DrawConstants.viewProjectionColumn1),
        dot(worldPosition, DrawConstants.viewProjectionColumn2),
        dot(worldPosition, DrawConstants.viewProjectionColumn3));
}
