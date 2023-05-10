#pragma once

#include "SDL.h"
#include <type_traits>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cudaRasterization.cuh"


class DisplayManager {
public:
    //The screen size is fixed and set at compile time along with other important
    //Display related values.
    int SCREEN_WIDTH = 400; //640 1280
    int SCREEN_HEIGHT = 400; //480 720
    //const static int SCREEN_PITCH = SCREEN_HEIGHT * sizeof(Uint32);
    //constexpr static float SCREEN_ASPECT_RATIO = SCREEN_WIDTH / (float)SCREEN_HEIGHT;

    //Dummy Constructor / Destructor
    DisplayManager();
    ~DisplayManager();

    //Initializes SDL context and creates window according to above values
    bool startUp();
    void shutDown();

    //Swaps the pixel buffer with the window surface buffer and draws to screen
    void swapBuffers(colorBuffer * cb);

    void pullEvents();

    bool onLoop{ true };
private:
    //Wrappers for SDL init functions
    bool startSDL();
    bool createWindow();
    bool createSDLTexture();
    void onEvent();

    //Pointers to the SDL window and surface
    //SDL_Surface* mSurface;
    SDL_Window* mWindow{nullptr};
    SDL_Renderer* mRender{nullptr};
    SDL_Texture* mTexture{nullptr};
    SDL_Event mEvent;

};
