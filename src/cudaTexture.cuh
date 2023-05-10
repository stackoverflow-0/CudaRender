#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef __CUDACC__

#define __CUDACC__
#include <cuda_texture_types.h>
#include <texture_types.h>
#include <texture_indirect_functions.h>
#include <texture_fetch_functions.h>

#endif // !__CUDACC__

#include <Windows.h>
#include <iostream>

//#ifndef STB_IMAGE_IMPLEMENTATION



//#endif // !STB_IMAGE_IMPLEMENTATION



cudaTextureObject_t createTexture(int width, int height, UINT32 * src);

cudaTextureObject_t loadImg2Texture(const char* imgPath);