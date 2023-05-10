#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Windows.h>
#include <iostream>
struct colorBuffer {
	UINT32* pixels;
	int width;
	int height;
	int size;
};

colorBuffer* createColorBuffer(int w,int h);

void freeColorBuffer(colorBuffer* cb);

void clearColorBuffer(colorBuffer* cb);

void printCB(colorBuffer* cb);