#pragma once

#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

/**
 * Structures describing different geometries should be defined here.
 */

struct MaterialValue
{
    optix::float3 ambient, diffuse, specular, emission;
    float shininess;
    float roughness;
    float ior; 
    unsigned int BRDFMode;
};

struct Triangle
{
    optix::float3 v1, v2, v3, normal; // transformed
    int lightID;
    MaterialValue mv;
};

struct Sphere
{
    optix::Matrix4x4 trans;
    MaterialValue mv;
};

struct Attributes
{
    optix::float3 intersection, normal, wo;
    MaterialValue mv;
    int lightID;
};