#include "displayManager.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// #include "cudaRasterization.h"
//Dummy constructors/destructors

DisplayManager::DisplayManager() {}
DisplayManager::~DisplayManager() {}

//Initializes the window and obtains the surface on which we draw.
//Also first place where SDL is initialized. 
bool DisplayManager::startUp() {
    bool success = true;
    if (!startSDL()) {
        success = false;
    }
    else {
        if (!createWindow()) {
            success = false;
        }
        else {
            if (!createSDLTexture()) {
                success = false;
            }
        }
    }
    return success;
}

//Closes down sdl and destroys window.
//SDL surface is also destroyed in the call to destroy window
void DisplayManager::shutDown() {
    SDL_DestroyWindow(mWindow);
    mWindow = nullptr;
    SDL_Quit();
}

//Applies the rendering results to the window screen by copying the pixelbuffer values
//to the screen surface.
void DisplayManager::swapBuffers(colorBuffer * cb) {

    unsigned char* pixels{nullptr};
    int pitch;

    SDL_LockTexture(mTexture, NULL, (void**)&pixels, &pitch);
    memcpy(pixels, cb->pixels, cb->size * sizeof(UINT32));
    SDL_UnlockTexture(mTexture);
    SDL_RenderCopy(mRender, mTexture, NULL, NULL);
    SDL_RenderPresent(mRender);

}

void DisplayManager::pullEvents()
{
    while (SDL_PollEvent(&mEvent)) {
        onEvent();

    }
}

//Entry point to SDL
bool DisplayManager::startSDL() {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("Failed to initialize SDL. Error: %s\n", SDL_GetError());
        return  false;
    }
    return true;
}

//Inits window with the display values crated at compile time
bool DisplayManager::createWindow() {
    mWindow = SDL_CreateWindow("SoftwareRenderer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_ALLOW_HIGHDPI);
    if (mWindow == nullptr) {
        printf("Could not create window. Error: %s\n", SDL_GetError());
        return false;
    }
    return true;
}

//Gets the screen surface
//I know this is "Old" SDL and it's not really recommended anymore
//But given that I am doing 100% cpu based rendering it makes sense
//After all I'm not usin any of the new functionality
bool DisplayManager::createSDLTexture() {
    mRender = SDL_CreateRenderer(mWindow, -1, SDL_RENDERER_ACCELERATED);
    if (!mRender) {
        std::cout << "Error creating renderer: " << SDL_GetError() << std::endl;
        return false;
    }
    mTexture = SDL_CreateTexture(mRender, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, SCREEN_WIDTH, SCREEN_HEIGHT);
    if (!mTexture) {
        std::cout << "Error creating texture: " << SDL_GetError() << std::endl;
        return false;
    }
    return true;
}

void DisplayManager::onEvent()
{
    if (mEvent.type == SDL_QUIT) {
        onLoop = false;
    }
    else if (mEvent.type == SDL_MOUSEMOTION) {
        int x, y;
        SDL_GetMouseState(&x, &y);
    }
}
