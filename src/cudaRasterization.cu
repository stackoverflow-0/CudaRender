#include "cudaRasterization.cuh"

__device__
bool PointinTriangle(const face * f, glm::vec3 & P, glm::vec2 & uvP)
{
	glm::vec3 A = f->vertex[0];
	glm::vec3 B = f->vertex[1];
	glm::vec3 C = f->vertex[2];
	glm::vec2 uvA = f->uv[0];
	glm::vec2 uvB = f->uv[1];
	glm::vec2 uvC = f->uv[2];
	glm::vec2 v0(C - A);
	glm::vec2 v1(B - A);
	glm::vec2 v2(P - A);
	float dot00 = glm::dot(v0, v0);
	float dot01 = glm::dot(v0, v1);
	float dot02 = glm::dot(v0, v2);
	float dot11 = glm::dot(v1, v1);
	float dot12 = glm::dot(v1, v2);

	float inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);

	float u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
	float v = (dot00 * dot12 - dot01 * dot02) * inverDeno;
	P = A + u * (C - A) + v * (B - A);
	uvP = uvA + u * (uvC - uvA) + v * (uvB - uvA);
	if (u < 0 || u > 1) // if u out of range, return directly
	{
		return false;
	}
	if (v < 0 || v > 1) // if v out of range, return directly
	{
		return false;
	}
	return u + v <= 1;
	return true;
	//return !((u < 0 || u > 1) || (v < 0 || v > 1));
}


__device__ 
void faceRasterization(face * f,const float top,const float bottom,const float left,const float right, zbuffer * zb, colorBuffer* cb, cudaTextureObject_t tex) {
	
	const int scr_w = zb->width;
	const int scr_h = zb->height;
	const int hfscr_w = scr_w >> 1;
	const int hfscr_h = scr_h >> 1;

	const int height = (top - bottom) * hfscr_h + 4;
	const int width  = (right - left) * hfscr_w + 4;

	const int total_n = width * height;

	for (int idx  = 0;idx < total_n; ) {
		int l_idx = (left  + 1.0f) * float(hfscr_w) - 2;
		int t_idx = (-top  + 1.0f) * float(hfscr_h) - 2;
		// get index position in zbuffer mat
		glm::ivec2 z_idx (l_idx + idx % width , t_idx + idx / width);
		bool in_range = z_idx.x >= 0 && z_idx.x < scr_w && z_idx.y >= 0 && z_idx.y < scr_h;

		// screen pixel position vec2 in clip space
		glm::vec3 zbuffer_pos(
			float(z_idx.x) / float(hfscr_w) - 1.0f,
			-float(z_idx.y) / float(hfscr_h) + 1.0f, 0);
		glm::vec2 zbuffer_uv;
		int pixelIdx = z_idx.x + scr_w * z_idx.y;
		bool getSem = zb->sems[pixelIdx].try_acquire();
		if (getSem) {
			if (in_range && PointinTriangle(f, zbuffer_pos, zbuffer_uv)) {

				//std::cout << "uv pos :\t" << zbuffer_uv.x << "\t" << zbuffer_uv.y << std::endl;
				
				UINT32 UVcolor = tex2D<UINT32>(tex, zbuffer_uv.x, zbuffer_uv.y);
				//printf("idx - %d uv pos : \t %f\t%f\n", blockIdx.x * blockDim.x + threadIdx.x, zbuffer_uv.x, zbuffer_uv.y);

				if (zbuffer_pos.z > zb->depth[pixelIdx]) {
					cb->pixels[pixelIdx] = UVcolor;
					zb->depth[pixelIdx] = zbuffer_pos.z;
				}
				
			}
			idx++;
			zb->sems[pixelIdx].release();
			
		}
		zb->sems[pixelIdx].release();
		//printf("here Threadidx %d with loop idx %d \n", threadIdx.x, idx);
		
	}
}

__global__ 
void cudaRasterization(vertexBuffer * vb, zbuffer * zb, colorBuffer* cb, cudaTextureObject_t tex) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int step = blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;
	const int face_n = vb->face_num;
	for (; idx < face_n; idx += step) {
		face* f = &(vb->faces[idx]);
		float top	 = fmaxf(f->vertex[0].y, fmaxf(f->vertex[1].y, f->vertex[2].y));
		top = fminf(top, 1);
		float bottom = fminf(f->vertex[0].y, fminf(f->vertex[1].y, f->vertex[2].y));
		bottom = fmaxf(bottom, -1);
		float left   = fminf(f->vertex[0].x, fminf(f->vertex[1].x, f->vertex[2].x));
		left = fmaxf(left, -1);
		float right  = fmaxf(f->vertex[0].x, fmaxf(f->vertex[1].x, f->vertex[2].x));
		right = fminf(right, 1);
		dim3 grid(4, 1, 1);
		dim3 block(16, 16, 1);
		faceRasterization (f, top, bottom, left, right, zb, cb, tex);
	}
}

void printDB(zbuffer* zb) {
	int cnt = 0;
	for (int i = 0; i < zb->height; ++i) {
		for (int j = 0; j < zb->width; ++j) {
			if (zb->depth[i * zb->width + j] > 0) {
				cnt++;
			}
		}
	}
	std::cout << "DB count : " << cnt << std::endl;
}


void Rasterization(dim3 grid, dim3 block, vertexBuffer * vb, zbuffer* zb, colorBuffer* cb, cudaTextureObject_t tex) {
	cudaRasterization << <grid, block >> > (vb, zb, cb, tex);
	cudaDeviceSynchronize();
	// printDB();

}

zbuffer*  createZbuffer(int w, int h) {
	zbuffer* zb;
	cudaMallocManaged(&zb, sizeof(zbuffer));
	zb->width = w;
	zb->height = h;
	zb->size = w * h;
	cudaMallocManaged(&zb->depth, w * h * sizeof(float));
#if defined(__USE_CUDA_SEMAPHORE__)
	cudaMallocManaged(&zb->sems, w * h * sizeof(cuda::binary_semaphore<cuda::thread_scope_device>));
#elif defined(__USE_MY_SEMAPHORE__)
	cudaMallocManaged(&zb->sems, w * h * sizeof(mySemaphore));
#endif
	
	//cuda::binary_semaphore<cuda::thread_scope_system> sem[400 * 400];
	//cudaMemcpy(zb->sems, sem, 400 * 400 * sizeof(cuda::binary_semaphore<cuda::thread_scope_system>), cudaMemcpyHostToHost);
	//zb->sems = new cuda::binary_semaphore<cuda::thread_scope_system>[w * h]();
	return zb;
}

void freeZbuffer(zbuffer* zb) {
	cudaFree(zb->depth);
	cudaFree(zb->sems);
	cudaFree(zb);
}

__global__
void setZbufferZero(zbuffer* zb) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int step = blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;
	int size = zb->size;
	for (; idx < size; idx += step) {
		zb->depth[idx] = 0;
	}
}

void clearZBuffer(zbuffer* zb)
{
	setZbufferZero << <16, 16 >> > (zb);
	cudaDeviceSynchronize();
}

#if defined(__USE_MY_SEMAPHORE__)
__device__
bool mySemaphore::try_acquire()
{
	if (atomicCAS(&sem, 1, 0)) {
		return true;
	}
	return false;
}
__device__
void mySemaphore::release()
{
	atomicCAS(&sem, 0, 1);
}
#endif
