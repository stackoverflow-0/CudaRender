#include "cudaTexture.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.cuh"

cudaTextureObject_t createTexture(int width, int height, UINT32* dataIn)
{
	cudaTextureObject_t tex;
	UINT32* dataDev;
	size_t pitch;
	cudaMallocPitch(&dataDev, &pitch, width * sizeof(UINT32), height);
	cudaMemcpy2D(dataDev, pitch, dataIn, width * sizeof(UINT32), width * sizeof(UINT32), height, cudaMemcpyHostToDevice);

	cudaResourceDesc resDesc;
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = dataDev;
	resDesc.res.pitch2D.width = width;
	resDesc.res.pitch2D.height = height;
	resDesc.res.pitch2D.desc = cudaCreateChannelDesc<UINT32>(); 
	//resDesc.res.pitch2D.desc.f = cudaChannelFormatKindUnsignedNormalized8X4;
	resDesc.res.pitch2D.pitchInBytes = pitch;
	
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeMirror;
	texDesc.addressMode[1] = cudaAddressModeMirror;
	texDesc.addressMode[2] = cudaAddressModeMirror;
	texDesc.normalizedCoords = 1;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
	return tex;
}



cudaTextureObject_t loadImg2Texture(const char* imgPath)
{
	int w, h;
	int channel;
	stbi_uc* data = stbi_load(imgPath, &w, &h, &channel, STBI_rgb_alpha);
	if (data == nullptr) {
		std::cout << " load error" << std::endl;
		return 0;
	}
	std::cout << "channel : " << channel << std::endl;
	UINT32* cvt_TexColor = (UINT32*)malloc(w * h * sizeof(UINT32));
	if (cvt_TexColor == nullptr) {
		std::cout << "alloc error\n";
	}
	stbi_uc* d = data;
	UINT32* cvt = cvt_TexColor;
	for (int i = 0; i < w * h; i++) {
		*cvt = *(d++) << 24;
		*cvt |= *(d++) << 16;
		*cvt |= *(d++) << 8;
		*cvt |= *(d++);
		++cvt;

		//cvt_TexColor[i] = (87 << 24);
		//cvt_TexColor[i] |= (255 << 16);
		//cvt_TexColor[i] |= (143 << 8);
		//cvt_TexColor[i] |= 255;
		
		//cvt_TexColor[i] = (254 << 24) | (222 << 16) | (87 << 8) | 255;
	}
	int pos = 512 * 256 * 4 + 256 * 4;
	//printf("rgba at 0,0 : %x  \n", ((UINT32*)data)[pos]);
	cudaTextureObject_t tex = createTexture(w, h, cvt_TexColor);
	stbi_image_free(data);
	free(cvt_TexColor);
	return tex;
}
