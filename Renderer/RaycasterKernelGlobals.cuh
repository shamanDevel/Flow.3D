
extern texture<float4, cudaTextureType3D, cudaReadModeElementType> g_texVolume1;
extern texture<float4, cudaTextureType3D, cudaReadModeElementType> g_texVolume2;
extern texture<float4, cudaTextureType3D, cudaReadModeElementType> g_texVolume3;

extern texture<float1, cudaTextureType3D, cudaReadModeElementType> g_texFeatureVolume;

extern texture<float4, cudaTextureType1D, cudaReadModeElementType> g_texTransferFunction;

extern surface<void,   cudaSurfaceType2D>                          g_surfTarget;
extern surface<void,   cudaSurfaceType2D>                          g_surfDepth;


extern __constant__ ProjectionParamsGPU c_projParams;

extern __constant__ RaycastParamsGPU c_raycastParams;
