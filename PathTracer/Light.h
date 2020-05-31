#pragma once

#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

/**
 * Structures describing different light sources should be defined here.
 */

struct PointLight
{
    optix::float3 color, attenuation, location;
};

struct DirectionalLight
{
    optix::float3 color, direction;
};

struct QuadLight
{
    optix::float3 a, ab, ac, intensity;
    int sampleCount;
};