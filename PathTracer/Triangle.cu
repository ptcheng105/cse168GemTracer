#include "optix.h"
#include "optix_device.h"
#include "Geometries.h"

using namespace optix;

rtBuffer<Triangle> triangles; // a buffer of all spheres

rtDeclareVariable(Ray, ray, rtCurrentRay, );

// Attributes to be passed to material programs 
rtDeclareVariable(Attributes, attrib, attribute attrib, );

RT_PROGRAM void intersect(int primIndex)
{
    // Find the intersection of the current ray and triangle
    Triangle tri = triangles[primIndex];

    float nDotWo = dot(tri.normal, -ray.direction);
    if (nDotWo == 0.0f) return;

    float t = dot(tri.v1 - ray.origin, tri.normal) / dot(ray.direction, tri.normal);
    float3 P = ray.origin + t * ray.direction; // intersection in the object space

    if (t < 0.001) return;

    float3 tmp0 = tri.v3 - tri.v1;
    float3 tmp1 = tri.v2 - tri.v1;
    float3 tmp2 = P - tri.v1;
    float tmp0dot0 = dot(tmp0, tmp0);
    float tmp0dot1 = dot(tmp0, tmp1);
    float tmp0dot2 = dot(tmp0, tmp2);
    float tmp1dot1 = dot(tmp1, tmp1);
    float tmp1dot2 = dot(tmp1, tmp2);
    float denom = tmp0dot0 * tmp1dot1 - tmp0dot1 * tmp0dot1;

    float u = (tmp1dot1 * tmp0dot2 - tmp0dot1 * tmp1dot2) / denom;
    float v = (tmp0dot0 * tmp1dot2 - tmp0dot1 * tmp0dot2) / denom;

    if (0 > u || u > 1 || 0 > v || v > 1 || u + v > 1) return;

    // Report intersection (material programs will handle the rest)
    if (rtPotentialIntersection(t))
    {
        // Pass attributes
        attrib.intersection = P;
        attrib.wo = -ray.direction;
        //attrib.normal = nDotWo > 0 ? tri.normal : -tri.normal;
        attrib.normal = tri.normal;
        attrib.mv = tri.mv;
        attrib.lightID = tri.lightID;
        rtReportIntersection(0);
    }
}

RT_PROGRAM void bound(int primIndex, float result[6])
{
    Triangle tri = triangles[primIndex];

    result[0] = fminf(fminf(tri.v1.x, tri.v2.x), tri.v3.x);
    result[1] = fminf(fminf(tri.v1.y, tri.v2.y), tri.v3.y);
    result[2] = fminf(fminf(tri.v1.z, tri.v2.z), tri.v3.z);
    result[3] = fmaxf(fmaxf(tri.v1.x, tri.v2.x), tri.v3.x);
    result[4] = fmaxf(fmaxf(tri.v1.y, tri.v2.y), tri.v3.y);
    result[5] = fmaxf(fmaxf(tri.v1.z, tri.v2.z), tri.v3.z);
}