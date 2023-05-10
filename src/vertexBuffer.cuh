#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>


#include <iostream>
#include <stack>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

struct face {
	glm::vec3 vertex[3];
	glm::vec2 uv[3];
};
struct vertexBuffer {
	face* faces;
	int face_num{ 0 };
	//cuda::binary_semaphore<cuda::thread_scope_device> * sems;
};

extern void ApplyTransToVertexBuffer(dim3 grid, dim3 block,
	cudaStream_t stream,const vertexBuffer* vb, const glm::mat4 transMat,vertexBuffer* transed_VB);

extern void freeVertexBuffer(vertexBuffer* vb);

vertexBuffer* allocVertexBuffer(int faces_n);
class cudaModel {
public:
	~cudaModel();
	void load(std::string model_path);
	vertexBuffer* vb{ nullptr };
	vertexBuffer* transVb{ nullptr };
private:
	void processNode(aiNode* node, const aiScene* scene);
	std::stack<face> face_stk;
};
