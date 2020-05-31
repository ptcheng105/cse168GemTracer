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
        float nDotwi = fmaxf(dot(attrib.normal, omega_i),0);
        float nDotwo = fmaxf(dot(attrib.normal, attrib.wo),0);
        //if(nDotwi > 1.0f || nDotwo > 1.0f)rtPrintf("%f, %f\n", nDotwi, nDotwo);
        if (nDotwi <= 0 || nDotwo <= 0) {
            f = make_float3(0, 0, 0);
        }
        else {
            float3 h = normalize(omega_i + attrib.wo);
            float hDotn = fmaxf(dot(h, attrib.normal), 0);
            if (hDotn >= 1.0f) hDotn = 1.0f;
            float theta_h = acos(hDotn);
            
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

    //if (!isfinite(f.x) || !isfinite(f.y) || !isfinite(f.z) )rtPrintf("%f,%f,%f\n", f.x, f.y, f.z);
    return f;
}

float3 generateWi(float theta, float phi, unsigned int thetaIsSpecular) {
    //if (!isfinite(cos(theta)))rtPrintf("%f\n", cos(theta));
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
    float3 result = normalize(s.x * u + s.y * v + s.z * w);
    //if (!isfinite(result.x) || !isfinite(result.y) || !isfinite(result.z))rtPrintf("%f,%f,%f\n", result.x, result.y, result.z);
    return result;
}

float getPDF_BRDF(float3 Wi, float t, float3 h) {
    float3 reflected = reflect(-attrib.wo, attrib.normal);
    float nDotwi = fmaxf(dot(attrib.normal, Wi), 0);
    float rDotwi = fmaxf(dot(reflected, Wi), 0);
    
    MaterialValue mv = attrib.mv;
    Config cf = config[0];
    float result = 0.0f;
    if (cf.importanceSamplingMode == 0) {
        result = 1 / (2 * M_PIf);
    }
    else if (cf.importanceSamplingMode == 1) {//cosine
        result =  (nDotwi / M_PIf);
    }
    else if (cf.importanceSamplingMode == 2) {//brdf
        if (mv.BRDFMode == 0) { // phong
            result =  ((1 - t) * nDotwi / M_PIf) + (t * ((mv.shininess + 1) / (2 * M_PIf)) * pow(rDotwi, mv.shininess));
        }
        else {
            float hDotn = fmaxf(dot(h, attrib.normal), 0);
            float hDotWi = fmaxf(dot(h, Wi), 0);
            if(hDotn >= 1.0f) hDotn = 1.0f;
            float theta_h = acos(hDotn);
            float D = pow(mv.roughness, 2.0f) / (M_PIf * pow(cos(theta_h), 4.0f) * pow((pow(mv.roughness, 2.0f) + pow(tan(theta_h), 2.0f)), 2.0f));
            result =  ((1 - t) * nDotwi / M_PIf) + (t * D * hDotn / (4 * hDotWi));
            
        }
    }
    if (!isfinite(result)) { rtPrintf("%f\n", result); }
    return result;
}

float3 directLightBRDF() {
    MaterialValue mv = attrib.mv;
    Config cf = config[0];

    //direction lighting term using BRDF important sampling
    float theta = 0, phi = 0, t = 0;
    unsigned int thetaIsSpecular = 0;
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
    float3 omega_i = generateWi(theta, phi, thetaIsSpecular);
    //now calculate the BRDF
    float3 h = make_float3(0, 0, 0);
    if (cf.importanceSamplingMode == 2 && mv.BRDFMode == 1) {
        //ggx, the omega_i we get here is actually h, compute real wi
        h = omega_i;
        omega_i = reflect(-attrib.wo, omega_i);
    }
    float3 reflected = reflect(-attrib.wo, attrib.normal);
    float3 f = getBRDF(omega_i);
    float nDotwi = fmaxf(dot(attrib.normal, omega_i), 0);

    //then find the pdf and then update throughput
    float pdf = getPDF(omega_i, t, h);

    //shoot a ray to omega_i and get the emission of the intersection
    // Prepare a payload
    Payload new_payload;
    new_payload.radiance = make_float3(0.f);
    new_payload.throughput = make_float3(1.f) * f * nDotwi / pdf;
    new_payload.depth = 0;
    new_payload.done = true;
    new_payload.seed = payload.seed;
    Ray ray = make_Ray(attrib.intersection + omega_i * cf.epsilon, omega_i, 0, cf.epsilon, RT_DEFAULT_MAX);
    rtTrace(root, ray, new_payload);
    return new_payload.radiance;

}

float3 directLightNEE() {
    MaterialValue mv = attrib.mv;
    Config cf = config[0];
    float3 result=make_float3(0,0,0);
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
    return result;
}

float3 DLBRDF(float& pdf_brdf) {
    MaterialValue mv = attrib.mv;
    Config cf = config[0];

    //direction lighting term using BRDF important sampling
    float theta = 0, phi = 0, t = 0;
    unsigned int thetaIsSpecular = 0;
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
    float3 omega_i = generateWi(theta, phi, thetaIsSpecular);
    //now calculate the BRDF
    float3 h = make_float3(0, 0, 0);
    if (cf.importanceSamplingMode == 2 && mv.BRDFMode == 1) {
        //ggx, the omega_i we get here is actually h, compute real wi
        h = omega_i;
        omega_i = reflect(-attrib.wo, omega_i);
    }
    float3 f = getBRDF(omega_i);
    float nDotwi = fmaxf(dot(attrib.normal, omega_i), 0);

    //then find the pdf and then update throughput
    float pdf = getPDF(omega_i, t, h);
    pdf_brdf = pdf;
    //shoot a ray to omega_i and get the emission of the intersection
    // Prepare a payload
    Payload new_payload;
    new_payload.radiance = make_float3(0.f);
    new_payload.throughput = make_float3(1.f) * f * nDotwi/ pdf;
    new_payload.depth = 0;
    new_payload.done = true;
    new_payload.seed = payload.seed;
    Ray ray = make_Ray(attrib.intersection + omega_i * cf.epsilon, omega_i, 0, cf.epsilon, RT_DEFAULT_MAX);
    rtTrace(root, ray, new_payload);
    return new_payload.radiance;
}
float3 DLNEE(float& pdf_nee) {
    MaterialValue mv = attrib.mv;
    Config cf = config[0];
    float3 result = make_float3(0, 0, 0);
    float3 summed_f = make_float3(0, 0, 0);
    float summed_pdf = 0;
    //generate ray to each light
    for (int i = 0; i < qlights.size(); i++)
    {
        QuadLight light = qlights[i];
        //generate random samples points to the light and add them up
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

        float pdf_i;
        float3 f_i;
        if (shadowPayload.isVisible)
        {
            f_i = getBRDF(lightDir);
            float area = length(light.ab) * length(light.ac);
            float3 lightNormal = normalize(cross(light.ac, light.ab));
            float nlDotWi = fmaxf(dot(-lightNormal, lightDir), 0);
            if (nlDotWi == 0) {
                pdf_i = 0;
            }
            else {
                pdf_i = pow(lightDist, 2.0f) / (area * nlDotWi);
                summed_pdf += pdf_i;
                float nDotwi = fmaxf(dot(attrib.normal, lightDir), 0);
                result += light.intensity * f_i * nDotwi / pdf_i;
            }            
        }
    }
    pdf_nee = summed_pdf / qlights.size();
    
    return result;
}

float3 EvalNee(float3 wi) {

}

float3 SampleNee(QuadLight light) {

}

float PdfNee(float3 wi){

}

float3 EvalBRDF(float3 wi) {

}

float3 SampleBRDF()
float PdfBRDF(float3 wi){

}
RT_PROGRAM void closestHit()
{
    MaterialValue mv = attrib.mv;
    Config cf = config[0];

    float3 result = make_float3(0, 0, 0);

    if (cf.useNEE == 0) {//not using NEE
        //add emission term to result if it is the bottom side of the light geometry
        float nDotwo = dot(attrib.normal, attrib.wo);
        if (nDotwo > 0) {
            result += make_float3(0, 0, 0);
        }
        else {
            result += mv.emission;
        }
        if (mv.emission.x != 0 || mv.emission.y != 0 || mv.emission.z != 0) {
            payload.radiance = result * payload.throughput;
            payload.done = true;
            return;
        }
    }
    else if (cf.useNEE == 1) { //using NEE
        //emission term
        if (payload.depth == 0) {//first intersection
            //add emission term to result if it is the bottom side of the light geometry
            float nDotwo = dot(attrib.normal, attrib.wo);
            if (nDotwo > 0) {
                result += make_float3(0, 0, 0);
            }
            else {
                result += mv.emission;
            }
        }
        //stop tracing path if the it hit a light source
        if (mv.emission.x != 0 || mv.emission.y != 0 || mv.emission.z != 0) {
            payload.radiance = result;
            payload.done = true;
            return;
        }

        result += directLightNEE();
    }
    else if (cf.useNEE == 2) {//using MIS
        if (payload.done == true) {//terminate
            payload.radiance = mv.emission * payload.throughput;
            return;
        }
        //add emission term to result if it is the bottom side of the light geometry
        float nDotwo = dot(attrib.normal, attrib.wo);
        if (nDotwo > 0) {
            result += make_float3(0, 0, 0);
        }
        else {
            result += mv.emission;
        }
        if (mv.emission.x != 0 || mv.emission.y != 0 || mv.emission.z != 0) {
            payload.radiance = result * payload.throughput;
            payload.done = true;
            return;
        }
        //get value and pdf of NEE
        float pdf_NEE = 0;
        float3 NEE_value = DLNEE(pdf_NEE);

        //get value and pdf of BRDF
        float pdf_BRDF = 0;
        float3 BRDF_value = DLBRDF(pdf_BRDF);

        if (!isfinite(pdf_BRDF)) {
            //result += make_float3(1000, 0, 0);
            //rtPrintf("%f, %f, %f\n", BRDF_value.x, BRDF_value.y, BRDF_value.z);
            //result += NEE_value;
        }
        else {
            float weight_nee = pow(pdf_NEE, 2.0f) / (pow(pdf_NEE, 2.0f) + pow(pdf_BRDF, 2.0f));
            float weight_brdf = pow(pdf_BRDF, 2.0f) / (pow(pdf_NEE, 2.0f) + pow(pdf_BRDF, 2.0f));
            if (weight_nee == 1.0f || weight_brdf == 0.0f) {
                //rtPrintf("%f, %f\n", pdf_NEE, pdf_BRDF);
            }
            //rtPrintf("%f, %f\n", weight_nee, weight_brdf);
            //calculate weighted value
            float3 weighted_nee = weight_nee * NEE_value;
            float3 weighted_brdf = weight_brdf * BRDF_value;


            result += weighted_nee+ weighted_brdf;
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

    if (cf.useRR == 1) {
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
    }
    
    payload.dir = omega_i;
    payload.depth++;

}