
extern cudaTextureObject_t g_texVolume1;
extern cudaTextureObject_t g_texVolume2;
extern cudaTextureObject_t g_texVolume3;

extern cudaTextureObject_t g_texFeatureVolume;

extern cudaTextureObject_t g_texTransferFunction;

extern surface<void,   cudaSurfaceType2D> g_surfTarget;
extern surface<void,   cudaSurfaceType2D> g_surfDepth;


extern __constant__ ProjectionParamsGPU c_projParams;

extern __constant__ RaycastParamsGPU c_raycastParams;
