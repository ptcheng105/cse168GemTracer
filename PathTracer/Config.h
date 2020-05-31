#pragma once

#include <optixu/optixu_math_namespace.h>

struct Config
{
    // Camera 
    optix::float3 w, u, v, eye; // w, u, v: orthonormal basis of camera; eye: eye location 
    optix::float2 hSize, tanHFov; // hSize: half size; tanHFov: tan of 0.5 * fov
    
    // Ray tracing 
    unsigned int maxDepth;
    float epsilon;

    //Monte Carlo stratification
    unsigned int useStratify;

    //indirect sample number(sample per pixel)
    unsigned int spp;
    unsigned int useNEE;
    unsigned int useRR; //russian roulette
    unsigned int importanceSamplingMode;
    float gamma;
    Config()
    {
        w = optix::make_float3(1.f, 0.f, 0.f);
        u = optix::make_float3(0.f, 1.f, 0.f);
        v = optix::make_float3(0.f, 0.f, 1.f);
        eye = optix::make_float3(0.f);
        hSize = optix::make_float2(100.f);
        tanHFov = optix::make_float2(tanf(0.25f * M_PIf));

        maxDepth = 5;
        epsilon = 0.0001f;

        useStratify = 0;
        spp = 1;
        useNEE = 0;
        useRR = 0;
        importanceSamplingMode = 0;//init to hemisphere
        gamma = 1;
    }
};