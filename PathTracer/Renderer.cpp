#include "Renderer.h"

using namespace optix;

Renderer::Renderer(std::shared_ptr<Scene> scene) : scene(scene)
{
    // Create the Optix context
    context = Context::create();

    // Configure the context 
    context->setRayTypeCount(3); // three types of rays: normal, shadow and ray-cast rays
    context->setEntryPointCount(1); // only one entry point
    context->setPrintEnabled(true); // enable the use of rtPrintf in programs
    context->setPrintBufferSize(2048);

    // Create the resultBuffer
    resultBuffer = context->createBuffer(RT_BUFFER_OUTPUT); // only device can write
    resultBuffer->setFormat(RT_FORMAT_FLOAT3); // each entry is of type float3
    accumBuffer = context->createBuffer(RT_BUFFER_OUTPUT); // only device can write
    accumBuffer->setFormat(RT_FORMAT_FLOAT3); // each entry is of type float3

    // Initialize Optix programs
    initPrograms();

    // Build the scene by constructing an Optix graph
    buildScene();
}

void Renderer::run(bool progressive)
{
    while (currentFrame != numFrames)
    {
        // Render a frame.
        context["frameID"]->setInt(++currentFrame);
        context->launch(0, width, height);

        // Only render a frame in progressive mode
        if (progressive) break;
    }
}

void Renderer::initPrograms()
{
    // Ray generation program
    programs["rayGen"] = createProgram("PinholeCamera.cu", "generateRays");
    context->setRayGenerationProgram(0, programs["rayGen"]);

    // Miss progarm
    programs["miss"] = createProgram("Common.cu", "miss");
    programs["miss"]["backgroundColor"]->setFloat(0.f, 0.f, 0.f);
    context->setMissProgram(0, programs["miss"]);

    // Exception program
    programs["exc"] = createProgram("Common.cu", "exception");
    context->setExceptionEnabled(RT_EXCEPTION_ALL, true);
    context->setExceptionProgram(0, programs["exc"]);

    // Triangle programs
    programs["triInt"] = createProgram("Triangle.cu", "intersect");
    programs["triBound"] = createProgram("Triangle.cu", "bound");

    // Sphere programs 
    programs["sphereInt"] = createProgram("Sphere.cu", "intersect");
    programs["sphereBound"] = createProgram("Sphere.cu", "bound");

    // Shadow Caster
    programs["shadowCaster"] = createProgram("Common.cu", "anyHit");

    // Integrators 
    programs["raytracer"] = createProgram("RayTracer.cu", "closestHit");
    programs["analyticdirect"] = createProgram("AnalyticDirect.cu", "closestHit");
    programs["direct"] = createProgram("Direct.cu", "closestHit");
    programs["pathtracer"] = createProgram("PathTracer.cu", "closestHit");
    programs["gemtracer"] = createProgram("GemTracer.cu", "closestHit");
    integrators = { "raytracer","analyticdirect", "direct", "pathtracer", "gemtracer" };
}

std::vector<unsigned char> Renderer::getResult()
{
    // Cast a float number (0 to 1) to a byte (0 to 255)
    auto cast = [](float v)
    {
        v = v > 1.f ? 1.f : v < 0.f ? 0.f : v;
        return static_cast<unsigned char>(v * 255);
    };

    float3* bufferData = (float3*)resultBuffer->map();

    // Store the data into a byte vector
    std::vector<unsigned char> imageData(width * height * 4);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int index = (i * width + j) * 4;
            float3 pixel = bufferData[(height - i - 1) * width + j];
            imageData[index + 0] = cast(pixel.x);
            imageData[index + 1] = cast(pixel.y);
            imageData[index + 2] = cast(pixel.z);
            imageData[index + 3] = 255; // alpha channel
        }
    }

    resultBuffer->unmap();

    return imageData;
}

Program Renderer::createProgram(const std::string& filename, 
    const std::string& programName)
{
    const char* ptx = sutil::getPtxString("PathTracer", filename.c_str());
    return context->createProgramFromPTXString(ptx, programName);
}

template <class T>
Buffer Renderer::createBuffer(std::vector<T> data)
{
    Buffer buffer = context->createBuffer(RT_BUFFER_INPUT); // only host can write
    buffer->setFormat(RT_FORMAT_USER); // use user-defined format
    buffer->setElementSize(sizeof(T)); // size of an element
    buffer->setSize(data.size()); // number of elements
    std::memcpy(buffer->map(), data.data(), sizeof(T) * data.size());
    buffer->unmap();
    return buffer;
}

void Renderer::buildScene()
{
    Config& config = scene->config;

    // Record some important info
    width = scene->width;
    height = scene->height;
    outputFilename = scene->outputFilename;
    currentFrame = 0;
    numFrames = config.spp;

    // Set width and height
    resultBuffer->setSize(width, height);
    programs["rayGen"]["resultBuffer"]->set(resultBuffer);
    accumBuffer->setSize(width, height);
    programs["rayGen"]["accumBuffer"]->set(accumBuffer);
    context["width"]->setFloat(width);
    context["height"]->setFloat(height);

    // Set config
    std::vector<Config> configs = { config };
    Buffer configBuffer = createBuffer(configs);
    context["config"]->set(configBuffer);

    // Set material programs based on integrator type.
    if (std::find(integrators.begin(), integrators.end(), scene->integratorName)
        == integrators.end())
    {
        throw std::runtime_error("Doesn't support the integrator " + 
            scene->integratorName);
    }
    programs["integrator"] = programs[scene->integratorName];

    Material material = context->createMaterial();
    material->setClosestHitProgram(0, programs["integrator"]);
    material->setAnyHitProgram(1, programs["shadowCaster"]);

    // Create buffers and pass them to Optix programs that the buffers
    Buffer triBuffer = createBuffer(scene->triangles);
    programs["triInt"]["triangles"]->set(triBuffer);
    programs["triBound"]["triangles"]->set(triBuffer);

    Buffer sphereBuffer = createBuffer(scene->spheres);
    programs["sphereInt"]["spheres"]->set(sphereBuffer);
    programs["sphereBound"]["spheres"]->set(sphereBuffer);

    Geometry triGeo = context->createGeometry();
    triGeo->setPrimitiveCount(scene->triangles.size());
    triGeo->setIntersectionProgram(programs["triInt"]);
    triGeo->setBoundingBoxProgram(programs["triBound"]);

    Geometry sphereGeo = context->createGeometry();
    sphereGeo->setPrimitiveCount(scene->spheres.size());
    sphereGeo->setIntersectionProgram(programs["sphereInt"]);
    sphereGeo->setBoundingBoxProgram(programs["sphereBound"]);

    GeometryInstance triGI = context->createGeometryInstance();
    triGI->setGeometry(triGeo);
    triGI->setMaterialCount(1);
    triGI->setMaterial(0, material);

    GeometryInstance sphereGI = context->createGeometryInstance();
    sphereGI->setGeometry(sphereGeo);
    sphereGI->setMaterialCount(1);
    sphereGI->setMaterial(0, material);

    GeometryGroup GG = context->createGeometryGroup();
    GG->setAcceleration(context->createAcceleration("Trbvh"));
    GG->setChildCount(1);
    GG->setChild(0, sphereGI);

    GeometryGroup triGG = context->createGeometryGroup();
    triGG->setAcceleration(context->createAcceleration("Trbvh"));
    triGG->setChildCount(1);
    triGG->setChild(0, triGI);

    Group root = context->createGroup();
    root->setAcceleration(context->createAcceleration("Trbvh"));
    root->setChildCount(2);
    root->setChild(0, triGG);
    root->setChild(1, GG);
    programs["rayGen"]["root"]->set(root);
    programs["integrator"]["root"]->set(root);

    // Create buffers for lights
    programs["integrator"]["plights"]->set(createBuffer(scene->plights));
    programs["integrator"]["dlights"]->set(createBuffer(scene->dlights));
    programs["integrator"]["qlights"]->set(createBuffer(scene->qlights));
    // Validate everything before running 
    context->validate();
}