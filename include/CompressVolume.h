#ifndef __TUM3D__COMPRESS_VOLUME_H__
#define __TUM3D__COMPRESS_VOLUME_H__


#include <vector>

#include <cuda_runtime.h>

#include <cudaCompress/global.h>
#include <cudaCompress/Instance.h>
#include <cudaCompress/util/CudaTimer.h>

#include "GPUResources.h"

struct CompressVolumeResources
{
	CompressVolumeResources()
		: pUpload(nullptr), syncEventUpload(0) {}

	static GPUResources::Config getRequiredResources(cudaCompress::uint sizeX, cudaCompress::uint sizeY, cudaCompress::uint sizeZ, cudaCompress::uint channelCount, cudaCompress::uint log2HuffmanDistinctSymbolCountMax = 0);

	bool create(const GPUResources::Config& config);
	void destroy();

	cudaCompress::byte* pUpload;
	cudaEvent_t syncEventUpload;

	cudaCompress::util::CudaTimerResources timerEncode;
	cudaCompress::util::CudaTimerResources timerDecode;
};


// helper struct for multi-channel compression functions
struct VolumeChannel
{
	float*      dpImage;
	const cudaCompress::uint* pBits;
	cudaCompress::uint        bitCount;
	float       quantizationStepLevel0;
};


// Compress one level of a scalar volume (lossless):
// - perform integer (reversible) DWT
// - encode highpass coefficients into bitstream
// - return lowpass coefficients
// The input is assumed to be roughly zero-centered.
// Decompress works analogously.
bool compressVolumeLosslessOneLevel(GPUResources& shared, CompressVolumeResources& resources, const short* dpImage, cudaCompress::uint sizeX, cudaCompress::uint sizeY, cudaCompress::uint sizeZ, short* dpLowpass, std::vector<cudaCompress::uint>& highpassBitStream);
void decompressVolumeLosslessOneLevel(GPUResources& shared, CompressVolumeResources& resources, short* dpImage, cudaCompress::uint sizeX, cudaCompress::uint sizeY, cudaCompress::uint sizeZ, const short* dpLowpass, const std::vector<cudaCompress::uint>& highpassBitStream);

// Convenience functions for multi-level lossless compression
bool compressVolumeLossless(GPUResources& shared, CompressVolumeResources& resources, const short* dpImage, cudaCompress::uint sizeX, cudaCompress::uint sizeY, cudaCompress::uint sizeZ, cudaCompress::uint numLevels, std::vector<cudaCompress::uint>& bits);
void decompressVolumeLossless(GPUResources& shared, CompressVolumeResources& resources, short* dpImage, cudaCompress::uint sizeX, cudaCompress::uint sizeY, cudaCompress::uint sizeZ, cudaCompress::uint numLevels, const std::vector<cudaCompress::uint>& bits);


// Compress a volume (lossy):
// - perform numLevels DWT
// - quantize coefficients and encode into bitstream
// The input is assumed to be roughly zero-centered.
// Decompress works analogously.
bool compressVolumeFloat(GPUResources& shared, CompressVolumeResources& resources, const float* dpImage, cudaCompress::uint sizeX, cudaCompress::uint sizeY, cudaCompress::uint sizeZ, cudaCompress::uint numLevels, std::vector<cudaCompress::uint>& bitStream, float quantizationStepLevel0, bool doRLEOnlyOnLvl0 = false);
void decompressVolumeFloat(GPUResources& shared, CompressVolumeResources& resources, float* dpImage, cudaCompress::uint sizeX, cudaCompress::uint sizeY, cudaCompress::uint sizeZ, cudaCompress::uint numLevels, const std::vector<cudaCompress::uint>& bitStream, float quantizationStepLevel0, bool doRLEOnlyOnLvl0 = false);
void decompressVolumeFloat(GPUResources& shared, CompressVolumeResources& resources, float* dpImage, cudaCompress::uint sizeX, cudaCompress::uint sizeY, cudaCompress::uint sizeZ, cudaCompress::uint numLevels, const cudaCompress::uint* pBits, cudaCompress::uint bitCount, float quantizationStepLevel0, bool doRLEOnlyOnLvl0 = false);
void decompressVolumeFloatMultiChannel(GPUResources& shared, CompressVolumeResources& resources, const VolumeChannel* pChannels, cudaCompress::uint channelCount, cudaCompress::uint sizeX, cudaCompress::uint sizeY, cudaCompress::uint sizeZ, cudaCompress::uint numLevels, bool doRLEOnlyOnLvl0 = false);


// Compress a volume (lossy):
// - quantize first
// - perform numLevels integers DWT
// - encode coefficients into bitstream
// This ensures a maximum error <= quantStep / 2
bool compressVolumeFloatQuantFirst(GPUResources& shared, CompressVolumeResources& resources, const float* dpImage, cudaCompress::uint sizeX, cudaCompress::uint sizeY, cudaCompress::uint sizeZ, cudaCompress::uint numLevels, std::vector<cudaCompress::uint>& bitStream, float quantizationStep, bool doRLEOnlyOnLvl0 = false);
bool decompressVolumeFloatQuantFirst(GPUResources& shared, CompressVolumeResources& resources, float* dpImage, cudaCompress::uint sizeX, cudaCompress::uint sizeY, cudaCompress::uint sizeZ, cudaCompress::uint numLevels, const std::vector<cudaCompress::uint>& bitStream, float quantizationStep, bool doRLEOnlyOnLvl0 = false);
bool decompressVolumeFloatQuantFirst(GPUResources& shared, CompressVolumeResources& resources, float* dpImage, cudaCompress::uint sizeX, cudaCompress::uint sizeY, cudaCompress::uint sizeZ, cudaCompress::uint numLevels, const cudaCompress::uint* pBits, cudaCompress::uint bitCount, float quantizationStep, bool doRLEOnlyOnLvl0 = false);
bool decompressVolumeFloatQuantFirstMultiChannel(GPUResources& shared, CompressVolumeResources& resources, const VolumeChannel* pChannels, cudaCompress::uint channelCount, cudaCompress::uint sizeX, cudaCompress::uint sizeY, cudaCompress::uint sizeZ, cudaCompress::uint numLevels, bool doRLEOnlyOnLvl0 = false);


#endif
