#pragma once

#define   PRINT_MACRO_HELPER(x)   #x  
#define   PRINT_MACRO(x)   #x"="PRINT_MACRO_HELPER(x)  

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
	#pragma message(PRINT_MACRO(__CUDA_ARCH__))
	//#pragma message(PRINT_MACRO(__USE_CUDA_SEMAPHORE__))
	#include <cuda/std/semaphore>
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
	#pragma message(PRINT_MACRO(__CUDA_ARCH__))
	//#pragma message(PRINT_MACRO(__USE_MY_SEMAPHORE__))
#endif



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
		//float * depth;
		float* depth;
		int width;
		int height;
		int size;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#pragma message("use cuda sema")
		cuda::binary_semaphore<cuda::thread_scope_device>* sems;
#else
#pragma message("use my sema")
		mySemaphore* sems;
#endif
};

zbuffer* createZbuffer(int w, int h);

void Rasterization(dim3 grid, dim3 block, vertexBuffer* vb, zbuffer* zb, colorBuffer* cb, cudaTextureObject_t tex);

void printDB(zbuffer* zb);

void freeZbuffer(zbuffer* zb);

void clearZBuffer(zbuffer* zb);

// bool __PointinTriangle(glm::vec3 A, glm::vec3 B, glm::vec3 C, glm::vec3& P);

