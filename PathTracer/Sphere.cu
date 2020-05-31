#include <optix.h>
#include <optix_device.h>
#include "Geometries.h"

using namespace optix;

rtBuffer<Sphere> spheres; // a buffer of all spheres

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float1, epsilon, , ); // scene epsilon

// Attributes to be passed to material programs 
rtDeclareVariable(Attributes, attrib, attribute attrib, );

RT_PROGRAM void intersect(int primIndex)
{
    // Find the intersection of the current ray and sphere
    Sphere sphere = spheres[primIndex];
    Matrix4x4 itrans = sphere.trans.inverse();
    float4 rayOriH = itrans * make_float4(ray.origin, 1);
    float3 rayOri = make_float3(rayOriH) / rayOriH.w;
    float3 rayDir = normalize(make_float3(itrans * make_float4(ray.direction, 0)));

    float t = 0;
    float3 CP0 = rayOri;
    float P1dotCP0 = dot(rayDir, CP0);
    float CP0dotCP0 = dot(CP0, CP0);
    float r2 = 1.f;
    float disc = P1dotCP0 * P1dotCP0 - CP0dotCP0 + r2;
    if (disc < 0) return;
    if (disc == 0)
        t = -P1dotCP0;
    else if (CP0dotCP0 > r2)
        t = -P1dotCP0 - sqrt(disc);
    else
        t = -P1dotCP0 + sqrt(disc);

    if (t < epsilon.x) return;

    // Intersection is found
    float3 P = rayOri + t * rayDir; // intersection in the object space
    float4 intersectionH = sphere.trans * make_float4(P, 1);
    P = make_float3(intersectionH) / intersectionH.w; // intersection in the world space
    t = length(P - ray.origin); // distance

    // Report intersection (material programs will handle the rest)
    if (rtPotentialIntersection(t))
    {
        // Pass attributes
        attrib.intersection = P;
        attrib.wo = -ray.direction;
        float4 tintersection = sphere.trans.inverse() * make_float4(attrib.intersection, 1);
        attrib.normal = normalize(make_float3(tintersection) / tintersection.w);
        attrib.normal = normalize(make_float3(sphere.trans.inverse().transpose() * make_float4(attrib.normal, 0)));
        attrib.mv = sphere.mv;
        attrib.lightID = -1;
        rtReportIntersection(0);
    }
}

RT_PROGRAM void bound(int primIndex, float result[6])
{
    Sphere sphere = spheres[primIndex];
    float x, y, z;
    x = length(make_float3(sphere.trans.getRow(0)));
    y = length(make_float3(sphere.trans.getRow(1)));
    z = length(make_float3(sphere.trans.getRow(2)));
    result[0] = sphere.trans[3] - x;
    result[1] = sphere.trans[7] - y;
    result[2] = sphere.trans[11] - z;
    result[3] = sphere.trans[3] + x;
    result[4] = sphere.trans[7] + y;
    result[5] = sphere.trans[11] + z;
}