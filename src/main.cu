#include <iostream>
#include <cuda_runtime.h>
// #include "cudaVec.cuh"
#include "vertexBuffer.cuh"
#include "DisplayManager.cuh"
#include "cudaTexture.cuh"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

int main(int argc, char* argv[]) {
    DisplayManager dspMangaer;
    dspMangaer.startUp();
    

    dim3 grid(1, 1, 1);
    dim3 block(32, 32, 1);
    cudaModel cube;
    cube.load("./model/test.obj");

    auto zb = createZbuffer(dspMangaer.SCREEN_WIDTH, dspMangaer.SCREEN_HEIGHT);
    auto cb = createColorBuffer(dspMangaer.SCREEN_WIDTH, dspMangaer.SCREEN_HEIGHT);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // cudaEventRecord(start)
    
    float milliseconds = 0;
    float cur_time = 0;
    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 50.0f);
    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    auto view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    auto projection = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 400.0f);

    cudaTextureObject_t tex = loadImg2Texture("./img/awesomeface.png");
    glm::mat4 tmat;
    while (dspMangaer.onLoop) {
        cudaEventRecord(start);
        clearColorBuffer(cb);
        clearZBuffer(zb);
        
        tmat = glm::mat4(1);
        tmat = glm::scale(tmat, glm::vec3(0.1, 0.1, 0.1));
        tmat = glm::rotate(tmat, -cur_time, glm::vec3(1, 0, 1));

        tmat = projection * view * tmat;

        

        ApplyTransToVertexBuffer(grid, block, 0, cube.vb, tmat, cube.transVb);
        Rasterization(grid, block, cube.transVb, zb, cb, tex);
        //printDB(zb);
        //printCB(cb);
        dspMangaer.swapBuffers(cb);
        dspMangaer.pullEvents();
        cudaEventRecord(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        cur_time += 20 / 1000.0f;
        //std::cout << "Rasterization (ms) :" << milliseconds << std::endl;
        SDL_Delay(20);
        //break;
        
    }
    dspMangaer.shutDown();
    

    freeZbuffer(zb);
    freeColorBuffer(cb);
    return 0;
}
