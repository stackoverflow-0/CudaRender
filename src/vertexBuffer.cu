#include "vertexBuffer.cuh"

vertexBuffer* allocVertexBuffer(int vbSize) {
	vertexBuffer* vb;
	cudaMallocManaged(&vb, sizeof(vertexBuffer));

	vb->face_num = vbSize;

	cudaMallocManaged(&vb->faces, vbSize * sizeof(face));
	/*
		+-------> x
		|
		|
		y
	
	vb->faces[0].vertex[0] = glm::vec3(0, 0,  0);
	vb->faces[0].vertex[1] = glm::vec3(0.3, 0, 0);
	vb->faces[0].vertex[2] = glm::vec3(0, 0.3, 0);
	vb->faces[0].uv[0] = glm::vec2(0, 0);
	vb->faces[0].uv[1] = glm::vec2(1, 0);
	vb->faces[0].uv[2] = glm::vec2(0, 1);
	*/

	return vb;
}



void freeVertexBuffer(vertexBuffer * vb) {
	cudaFree(vb->faces);
	cudaFree(vb);
}

__global__ void cudaApplyTrans(const vertexBuffer * vb,const glm::mat4 * transMat, vertexBuffer* tmpVB) {
	int idx = blockIdx.z * blockDim.x * blockDim.y * blockDim.z + blockIdx.y * blockDim.x * blockDim.y +  blockIdx.x * blockDim.x + threadIdx.x;
	int step = blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;
	const int n = vb->face_num;
	for (; idx < n; idx+=step) {
		face* f = &(vb->faces[idx]);
		face* trans_f = &(tmpVB->faces[idx]);
		trans_f->vertex[0] = (*transMat) * glm::vec4(f->vertex[0], 1);
		trans_f->vertex[1] = (*transMat) * glm::vec4(f->vertex[1], 1);
		trans_f->vertex[2] = (*transMat) * glm::vec4(f->vertex[2], 1);
	}
}

void ApplyTransToVertexBuffer(dim3 grid, dim3 block, cudaStream_t stream,const vertexBuffer* vb,const glm::mat4 transMat, vertexBuffer* transed_VB) {
	// 变换矩阵的 cuda mem 副本
	glm::mat4 * cudaTransmat;
	cudaMallocManaged(&cudaTransmat, sizeof(glm::mat4));
	*cudaTransmat = transMat;
	cudaApplyTrans << <grid, block, 0, stream >> > (vb, cudaTransmat, transed_VB);
	cudaDeviceSynchronize();
	cudaFree(cudaTransmat);
}

void cudaModel::processNode(aiNode* node, const aiScene* scene)
{
	// process all the node's meshes (if any)
	for (unsigned int i = 0; i < node->mNumMeshes; i++)
	{
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
		for (unsigned int fidx = 0; fidx < mesh->mNumFaces; fidx++) {
			face f;
			aiVector3D v0 = mesh->mVertices[mesh->mFaces[fidx].mIndices[0]];
			aiVector3D v1 = mesh->mVertices[mesh->mFaces[fidx].mIndices[1]];
			aiVector3D v2 = mesh->mVertices[mesh->mFaces[fidx].mIndices[2]];
			aiVector3D uv0 = mesh->mTextureCoords[0][mesh->mFaces[fidx].mIndices[0]];
			aiVector3D uv1 = mesh->mTextureCoords[0][mesh->mFaces[fidx].mIndices[1]];
			aiVector3D uv2 = mesh->mTextureCoords[0][mesh->mFaces[fidx].mIndices[2]];

			printf("v ; %f %f %f \n", v0.x, v0.y, v0.z);
			printf("v ; %f %f %f \n", v1.x, v1.y, v1.z);
			printf("v ; %f %f %f \n", v2.x, v2.y, v2.z);

			f.vertex[0] = glm::vec3(v0.x, v0.y, v0.z);
			f.vertex[1] = glm::vec3(v1.x, v1.y, v1.z);
			f.vertex[2] = glm::vec3(v2.x, v2.y, v2.z);

			printf("uv ; %f %f \n", uv0.x, uv0.y);
			printf("uv ; %f %f \n", uv1.x, uv1.y);
			printf("uv ; %f %f \n", uv2.x, uv2.y);

			f.uv[0] = glm::vec2(uv0.x, uv0.y);
			f.uv[1] = glm::vec2(uv1.x, uv1.y);
			f.uv[2] = glm::vec2(uv2.x, uv2.y);

			face_stk.push(f);
		}
	}
	// then do the same for each of its children
	for (unsigned int i = 0; i < node->mNumChildren; i++)
	{
		processNode(node->mChildren[i], scene);
	}
}

cudaModel::~cudaModel()
{
	freeVertexBuffer(vb);
	freeVertexBuffer(transVb);
}

void cudaModel::load(std::string model_path)
{
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(model_path, aiProcess_Triangulate | aiProcess_FlipUVs);
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
	{
		std::cout << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
		return;
	}
	std::string directory = model_path.substr(0, model_path.find_last_of('/'));
	processNode(scene->mRootNode, scene);
	
	vb = allocVertexBuffer(face_stk.size());
	transVb = allocVertexBuffer(face_stk.size());
	std::cout << "total faces : " << face_stk.size() << std::endl;
	face* fp = vb->faces;
	face* tfp = transVb->faces;
	///*
	while (! face_stk.empty())
	{
		*(fp++) = face_stk.top();
		*(tfp++) = face_stk.top();
		face_stk.pop();
	}
	//*/

}