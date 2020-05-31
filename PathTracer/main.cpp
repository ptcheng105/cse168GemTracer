#include "main.h"

int main(int argc, char** argv)
{
    try
    {
        // Check for scene file 
        if (argc == 1) throw
            std::runtime_error("Missing path to a scene file as the first argument.");

        // Load the scene file 
        SceneLoader loader;
        std::shared_ptr<Scene> scene = loader.load(argv[1]);

        // Create the optix app by passing in the path to the scene file
        Renderer renderer(scene);
        bool display = true;
         
        if (display)
        {
            // Display the result. Rendered image will be saved automatically when you 
            // close the viewer. 
            sutil::initGlut(&argc, argv);
            Viewer::setRenderer(&renderer);
            Viewer::run();
        }
        else
        {
            auto begin = std::chrono::steady_clock::now();

            renderer.run(false);

            auto end = std::chrono::steady_clock::now();

            std::cout << "Elapsed time: " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end - 
                begin).count() * float(1e-6) 
                << " seconds" 
                << std::endl;

            resultToPNG(renderer.getOutputFilename(),
                renderer.getWidth(),
                renderer.getHeight(),
                renderer.getResult());
        }
    }
    catch (const std::exception & ex)
    {
        // Print out the exception for debugging
        std::cerr << ex.what() << std::endl;
    }
    return 0;
}