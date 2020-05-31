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

RT_PROGRAM void closestHit()
{
    MaterialValue mv = attrib.mv;
    Config cf = config[0];

    float3 result = mv.ambient + mv.emission;
    //loop all lights
    for (int i = 0; i < qlights.size(); i++) 
    {
        QuadLight light = qlights[i];
        //generate random samples points to the light and add them up
        float3 summed_samples = make_float3(0);
        
        if (cf.useStratify == 1) {
            //use stratification
            float N = sqrtf(light.sampleCount);
            float3 delta_ab = light.ab / N;
            float3 delta_ac = light.ac / N;
            for (int m = 0; m < N; m++) {
                for (int n = 0; n < N; n++) {
                    float3 new_a = light.a + (m * delta_ab) + (n * delta_ac);
                    float ran1 = rnd(payload.seed);
                    float ran2 = rnd(payload.seed);
                    float3 lightPoint = new_a + ran1 * delta_ab + ran2 * delta_ac;
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
                        float3 f = (mv.diffuse / M_PIf) + (mv.specular * ((mv.shininess + 2) / (2 * M_PIf)) * pow(fmaxf(dot(reflected, lightDir), 0), mv.shininess));
                        float3 t = (mv.diffuse / M_PIf);


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
            }
        }
        else {
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
                    float3 f = (mv.diffuse / M_PIf) + (mv.specular * ((mv.shininess + 2) / (2 * M_PIf)) * pow(fmaxf(dot(reflected, lightDir), 0), mv.shininess));
                    float3 t = (mv.diffuse / M_PIf);


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
        }
        //now we have all the sample, find the combined light
        float area = length(light.ab) * length(light.ac);
        //rtPrintf("%f, %f, %f\n", light.intensity.x, light.intensity.y, light.intensity.z);
        result += light.intensity * area * summed_samples / light.sampleCount;
    }

    // Compute the final radiance
    payload.radiance = result;

    payload.done = true;
}