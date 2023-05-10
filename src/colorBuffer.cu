#include "colorBuffer.cuh"

colorBuffer* createColorBuffer(int w, int h)
{
	colorBuffer* cb;
	cudaMallocManaged(&cb, sizeof(colorBuffer));
	cb->width = w;
	cb->height = h;
	cb->size = w * h;
	cudaMallocManaged(&cb->pixels, cb->size * sizeof(UINT32));
	return cb;
}

void freeColorBuffer(colorBuffer* cb)
{
	cudaFree(cb->pixels);
	cudaFree(cb);
}
__global__
void setColorBufferZero(colorBuffer* cb) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int step = blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;
	int size = cb->size;
	for (; idx < size; idx += step) {
		cb->pixels[idx] = 0;
	}
}

void clearColorBuffer(colorBuffer* cb)
{
	setColorBufferZero << <16, 16 >> > (cb);
	cudaDeviceSynchronize();
}


void printCB(colorBuffer* cb) {
	int cnt = 0;
	for (int i = 0; i < cb->height; ++i) {
		for (int j = 0; j < cb->width; ++j) {
			if (cb->pixels[i * cb->width + j] > 0) {
				cnt++;
			}
		}
	}
	std::cout << "CB count : " << cnt << std::endl;
}