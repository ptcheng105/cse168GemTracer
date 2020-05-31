#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "random.h"

#include "Payloads.h"
#include "Geometries.h"
#include "Light.h"
#include "Config.h"

using namespace optix;

// Declare light buffers
rtBuffer<PointLight> plights;
rtBuffer<DirectionalLight> dlights;
rtBuffer<QuadLight> qlights;

// Declare variables
rtDeclareVariable(Payload, payload, rtPayload, );
rtDeclareVariable(rtObject, root, , );

rtBuffer<Config> config; // Config

// Declare attibutes 
rtDeclareVariable(Attributes, attrib, attribute attrib, );

float3 get_comp(float3 k, float3 l, float3 r) 
{
    float A = acos(dot(normalize(k - r), normalize(l - r)));
    float3 B = normalize(cross((k - r), (l - r)));
    return A * B;
}

RT_PROGRAM void closestHit()
{
    MaterialValue mv = attrib.mv;
    Config cf = config[0];

    float3 result = mv.ambient + mv.emission;
    for (int i = 0; i < qlights.size(); i++) 
    {
        QuadLight light = qlights[i];
        float3 V1 = light.a;//a
        float3 V2 = light.a + light.ab; //b
        float3 V3 = light.a + light.ab + light.ac; //d
        float3 V4 = light.a + light.ac; //c

        float3 A = get_comp(V1, V2, attrib.intersection);
        float3 B = get_comp(V2, V3, attrib.intersection);
        float3 C = get_comp(V3, V4, attrib.intersection);
        float3 D = get_comp(V4, V1, attrib.intersection);

        float3 phi = (A + B + C + D) / 2;
        result = result + (mv.diffuse / M_PIf)* light.intensity* dot(phi, attrib.normal);
    }

    // Compute the final radiance
    payload.radiance = result;
    payload.done = true;
}