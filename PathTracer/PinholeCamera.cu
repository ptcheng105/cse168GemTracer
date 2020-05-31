#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "random.h"

#include "Payloads.h"
#include "Config.h"

using namespace optix;

rtBuffer<float3, 2> resultBuffer; // used to store the render result
rtBuffer<float3, 2> accumBuffer; // used to accumulate the render result

rtDeclareVariable(rtObject, root, , ); // Optix graph

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, ); // a 2d index (x, y)

rtDeclareVariable(int1, frameID, , );

rtBuffer<Config> config; // Config

RT_PROGRAM void generateRays()
{
    size_t2 resultSize = resultBuffer.size();
    unsigned int index = launchIndex.x * resultSize.y + launchIndex.y;
    unsigned int seed = tea<16>(index * frameID.x, 0);
    Config cf = config[0];
    float3 result = make_float3(0.f);

    // Compute the ray direction
    float2 xy = make_float2(launchIndex);
    xy.x += frameID.x == 1 ? 0.5f : rnd(seed);
    xy.y += frameID.x == 1 ? 0.5f : rnd(seed);
    float2 ab = cf.tanHFov * (xy - cf.hSize) / cf.hSize;
    float3 dir = normalize(ab.x * cf.u + ab.y * cf.v - cf.w); // ray direction
    float3 origin = cf.eye; // ray origin

    // Prepare a payload
    Payload payload;
    payload.radiance = make_float3(0.f);
    payload.throughput = make_float3(1.f);
    payload.depth = 0;
    payload.done = false;
    payload.isTransparent = false;
    int i = 0;

    // Iteratively trace rays (recursion is very expensive on GPU)
    do
    {
        payload.seed = tea<16>(index * frameID.x, i++);

        // Trace a ray
        Ray ray = make_Ray(origin, dir, 0, cf.epsilon, RT_DEFAULT_MAX);
        rtTrace(root, ray, payload);
        // Accumulate radiance
        result += payload.radiance;
        payload.radiance = make_float3(0.f);

        // Prepare to shoot next ray
        origin = payload.origin;
        dir = payload.dir;
    } while (!payload.done && payload.depth != cf.maxDepth);

    if (frameID.x == 1) {
        accumBuffer[launchIndex] = result;
        resultBuffer[launchIndex].x = pow(accumBuffer[launchIndex].x, (1 / cf.gamma));
        resultBuffer[launchIndex].y = pow(accumBuffer[launchIndex].y, (1 / cf.gamma));
        resultBuffer[launchIndex].z = pow(accumBuffer[launchIndex].z, (1 / cf.gamma));
    }
    else
    {
        float u = 1.0f / (float)frameID.x;
        float3 oldResult = accumBuffer[launchIndex];
        accumBuffer[launchIndex] = lerp(oldResult, result, u);
        resultBuffer[launchIndex].x = pow(accumBuffer[launchIndex].x, (1 / cf.gamma));
        resultBuffer[launchIndex].y = pow(accumBuffer[launchIndex].y, (1 / cf.gamma));
        resultBuffer[launchIndex].z = pow(accumBuffer[launchIndex].z, (1 / cf.gamma));
    }
}