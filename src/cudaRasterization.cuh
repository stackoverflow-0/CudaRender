#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <iostream>

#include <Windows.h>

#include <cuda_fp16.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "vertexBuffer.cuh"
#include "colorBuffer.cuh"

// #include "cudaTexture.cuh"
#ifndef __CUDACC__

#define __CUDACC__
#include <cuda_texture_types.h>
#include <texture_types.h>
#include <texture_indirect_functions.h>
#include <texture_fetch_functions.h>
#include <device_atomic_functions.hpp>
#endif // !__CUDACC__

class mySemaphore {
public:
	__device__
		bool try_acquire();
	__device__
		void release();
private:
	int sem;
};

struct zbuffer {
		float* depth;
		int width;
		int height;
		int size;
		mySemaphore*  sems;
};

zbuffer* createZbuffer(int w, int h);

void Rasterization(dim3 grid, dim3 block, vertexBuffer* vb, zbuffer* zb, colorBuffer* cb, cudaTextureObject_t tex);

void printDB(zbuffer* zb);

void freeZbuffer(zbuffer* zb);

void clearZBuffer(zbuffer* zb);


