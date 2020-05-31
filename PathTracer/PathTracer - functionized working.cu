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

float3 getBRDF(float3 omega_i) {
    MaterialValue mv = attrib.mv;
    Config cf = config[0];

    float3 f;
    if (mv.BRDFMode == 0) {
        //modified Phong
        float3 reflected = reflect(-attrib.wo, attrib.normal);
        f = (mv.diffuse / M_PIf) + (mv.specular * ((mv.shininess + 2) / (2 * M_PIf)) * pow(fmaxf(dot(reflected, omega_i), 0.0), mv.shininess));
    }
    else if (mv.BRDFMode == 1) {
        //GGX
        float nDotwi = dot(attrib.normal, omega_i);
        float nDotwo = dot(attrib.normal, attrib.wo);
        if (nDotwi <= 0 || nDotwo <= 0) {
            f = make_float3(0, 0, 0);
        }
        else {
            float3 h = normalize(omega_i + attrib.wo);
            float theta_h = acos(dot(h, attrib.normal));
            float D = pow(mv.roughness, 2.0f) / (M_PIf * pow(cos(theta_h), 4.0f) * pow((pow(mv.roughness, 2.0f) + pow(tan(theta_h), 2.0f)), 2.0f));

            float theta_v = acos(nDotwi);
            float G1wi = 2 / (1 + sqrtf(1 + (pow(mv.roughness, 2.0f) * pow(tan(theta_v), 2.0f))));
            theta_v = acos(nDotwo);
            float G1wo = 2 / (1 + sqrtf(1 + (pow(mv.roughness, 2.0f) * pow(tan(theta_v), 2.0f))));
            float G = G1wi * G1wo;

            float3 F = mv.specular + (1 - mv.specular) * pow((1 - fmaxf(dot(attrib.wo, h), 0)), 5.0f);
            f = (mv.diffuse / M_PIf) + (F * G * D / (4 * nDotwi * nDotwo));
        }
    }
    return f;
}

float3 generateWi(float theta, float phi, unsigned int thetaIsSpecular) {
    float3 s = make_float3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
    float3 w;
    if (thetaIsSpecular == 1) {
        float3 reflected = reflect(-attrib.wo, attrib.normal);
        w = normalize(reflected);
    }
    else {
        w = normalize(attrib.normal);
    }
    float3 u = normalize(cross(make_float3(1, 0, 0), w));
    float3 v = normalize(cross(w, u));
    return (s.x * u + s.y * v + s.z * w);
}

float getPDF(float3 Wi, float t, float3 h) {
    float3 reflected = reflect(-attrib.wo, attrib.normal);
    float nDotwi = fmaxf(dot(attrib.normal, Wi), 0);
    float rDotwi = fmaxf(dot(reflected, Wi), 0);
    MaterialValue mv = attrib.mv;
    Config cf = config[0];
    if (cf.importanceSamplingMode == 0) {
        return 1 / (2 * M_PIf);
    }
    else if (cf.importanceSamplingMode == 1) {//cosine
        return (nDotwi / M_PIf);
    }
    else if (cf.importanceSamplingMode == 2) {//brdf
        if (mv.BRDFMode == 0) { // phong
            return ((1 - t) * nDotwi / M_PIf) + (t * ((mv.shininess + 1) / (2 * M_PIf)) * pow(rDotwi, mv.shininess));
        }
        else {
            float theta_h = acos(dot(h, attrib.normal));
            float D = pow(mv.roughness, 2.0f) / (M_PIf * pow(cos(theta_h), 4.0f) * pow((pow(mv.roughness, 2.0f) + pow(tan(theta_h), 2.0f)), 2.0f));
            return ((1 - t) * nDotwi / M_PIf) + (t * D * dot(h, attrib.normal) / (4 * dot(h, Wi)));
        }

    }
}

RT_PROGRAM void closestHit()
{
    MaterialValue mv = attrib.mv;
    Config cf = config[0];

    float3 result = make_float3(0, 0, 0);

    if (cf.useNEE == 0) {//not using NEE
        result += mv.emission;
    }
    else { //using NEE
        //direction lighting term
        for (int i = 0; i < qlights.size(); i++)
        {
            QuadLight light = qlights[i];
            //generate random samples points to the light and add them up
            float3 summed_samples = make_float3(0);
            //not using stratification pretty much the same expect for the loop
            for (int j = 0; j < light.sampleCount; j++) {
                float ran1 = rnd(payload.seed);
                float ran2 = rnd(payload.seed);
                float3 lightPoint = light.a + ran1 * light.ab + ran2 * light.ac;
                float3 lightDir = normalize(lightPoint - attrib.intersection);
                float lightDist = length(lightPoint - attrib.intersection);
                ShadowPayload shadowPayload;
                shadowPayload.isVisible = true;
                Ray shadowRay = make_Ray(attrib.intersection + lightDir * cf.epsilon,
                    lightDir, 1, cf.epsilon, lightDist - (2 * cf.epsilon));
                rtTrace(root, shadowRay, shadowPayload);

                // check visibility first
                if (shadowPayload.isVisible)
                {
                    //calculate BRDF 
                    float3 reflected = reflect(-attrib.wo, attrib.normal);
                    float3 f = getBRDF(lightDir);

                    //calculate Geo term
                    float3 lightNormal = normalize(cross(light.ac, light.ab));
                    float nDotWi = fmaxf(dot(attrib.normal, lightDir), 0);
                    //if(nDotWi < 0) rtPrintf("%f\n", nDotWi);
                    float nlDotWi = fmaxf(dot(-lightNormal, lightDir), 0);
                    float power = pow(lightDist, 2.0f);
                    float g = nDotWi * nlDotWi / power;
                    //combine all terms for this sample and add it to the summed_samples
                    summed_samples += (f * g);
                }
            }
            //now we have all the sample, find the combined light
            float area = length(light.ab) * length(light.ac);
            result += light.intensity * area * summed_samples / light.sampleCount;
        }

        //emission term
        if (payload.depth == 0) {//first intersection
            result += mv.emission;
        }
        if (mv.emission.x != 0 || mv.emission.y != 0 || mv.emission.z != 0) {
            payload.radiance = result;
            payload.done = true;
            return;
        }
    }

    payload.radiance = result * payload.throughput;
    payload.origin = attrib.intersection;

    //indirect lighting
    //generate spherical coordinates theta and phi
    float theta = 0, phi = 0, t = 0;
    unsigned int thetaIsSpecular = 0;
    if (cf.importanceSamplingMode == 0) {
        theta = acos(rnd(payload.seed));
        phi = 2 * M_PIf * rnd(payload.seed);
    }
    else if (cf.importanceSamplingMode == 1) {//cosine
        theta = acos(sqrtf(rnd(payload.seed)));
        phi = 2 * M_PIf * rnd(payload.seed);
    }
    else if (cf.importanceSamplingMode == 2) {//brdf
        float diffuseAvg = (mv.diffuse.x + mv.diffuse.y + mv.diffuse.z) / 3;
        float specularAvg = (mv.specular.x + mv.specular.y + mv.specular.z) / 3;
        phi = 2 * M_PIf * rnd(payload.seed);

        if (mv.BRDFMode == 0) {
            t = specularAvg / (diffuseAvg + specularAvg);
        }
        else if (mv.BRDFMode == 1) {
            t = fmaxf(0.25, specularAvg / (diffuseAvg + specularAvg));
        }
        float xi_0 = rnd(payload.seed);
        if (xi_0 <= t) { //specular
            if (mv.BRDFMode == 0) {//phong
                theta = acos(pow(rnd(payload.seed), (1 / (mv.shininess + 1))));
                thetaIsSpecular = 1;
            }
            else if (mv.BRDFMode == 1) {//ggx
                float rand1 = rnd(payload.seed);
                theta = atan(mv.roughness * sqrtf(rand1) / sqrtf(1 - rand1));
            }
        }
        else {//diffuse
            theta = acos(sqrtf(rnd(payload.seed)));
        }
    }

    float3 omega_i = generateWi(theta, phi, thetaIsSpecular);
    //now calculate the BRDF
    float3 h = make_float3(0,0,0);
    if (cf.importanceSamplingMode == 2 && mv.BRDFMode == 1) {
        //ggx, the omega_i we get here is actually h, compute real wi
        h = omega_i;
        omega_i = reflect(-attrib.wo, omega_i);
    }
    float3 reflected = reflect(-attrib.wo, attrib.normal);
    float3 f = getBRDF(omega_i);
    float nDotwi = fmaxf(dot(attrib.normal, omega_i), 0);
    float rDotwi = fmaxf(dot(reflected, omega_i), 0);

    //then find the pdf and then update throughput
    float pdf = getPDF(omega_i, t, h);
    payload.throughput *= f * nDotwi / pdf;

    float q = 1 - fminf(fmaxf(payload.throughput.x, fmaxf(payload.throughput.y, payload.throughput.z)), 1);
    if (rnd(payload.seed) <= q) {
        //terminate this path
        payload.throughput = make_float3(0.f);
        payload.done = true;
        return;
    }
    else {
        //boost throughput
        payload.throughput = payload.throughput * (1 / (1 - q));
    }

    payload.dir = omega_i;
    payload.depth++;

}