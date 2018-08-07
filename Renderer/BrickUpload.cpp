#include "BrickUpload.h"

#include <cassert>

#include <cuda_runtime.h>
#include <cudaCompress/Instance.h>

#include <cudaUtil.h>


void UploadBrick(GPUResources* pShared, CompressVolumeResources* pRes,
				 const TimeVolumeInfo& volumeInfo, const TimeVolumeIO::Brick& brick,
				 float** pdpBuffer,
				 BrickSlot* pBrickSlot, const tum3D::Vec3ui& slotIndex,
				 MultiTimerGPU* pTimer)
{
	if(pTimer)
		pTimer->StartNextTimer();

	// actually, brick.GetDataSize() (non-padded) should be fine as well, but apparently the non-padded size is set to 0 in some data sets?
	cudaSafeCall(cudaHostRegister(const_cast<void*>(brick.GetData()), brick.GetPaddedDataSize(), cudaHostRegisterDefault));

	// upload/decompress
	const Vec3ui& size = brick.GetSize();

	if(volumeInfo.GetCompressionType() == COMPRESSION_NONE)
	{
		for(int channel = 0; channel < volumeInfo.GetChannelCount(); channel++)
		{
			// uncompressed data, upload directly
			cudaMemcpyAsync(pdpBuffer[channel], brick.GetChannelData(channel), brick.GetChannelDataSize(channel), cudaMemcpyHostToDevice);
			//TODO: before unloading a brick, we'd have to sync on the uploads to complete...
		}
	}
	else if(volumeInfo.GetCompressionType() == COMPRESSION_FIXEDQUANT || volumeInfo.GetCompressionType() == COMPRESSION_FIXEDQUANT_QF)
	{
		const int subbandCount = 7;
		int channelsPerPass = cudaCompress::getInstanceStreamCountMax(pShared->m_pCuCompInstance) / subbandCount;

		for(int channel0 = 0; channel0 < volumeInfo.GetChannelCount(); channel0 += channelsPerPass)
		{
			int channelCountThisPass = min(channelsPerPass, volumeInfo.GetChannelCount() - channel0);
			std::vector<VolumeChannel> channels(channelCountThisPass);
			for(int c = 0; c < channelCountThisPass; c++)
			{
				int channel = channel0 + c;
				channels[c].dpImage = pdpBuffer[channel];
				channels[c].pBits = static_cast<const uint*>(brick.GetChannelData(channel));
				channels[c].bitCount = uint(brick.GetChannelDataSize(channel) * 8);
				channels[c].quantizationStepLevel0 = volumeInfo.GetQuantStep(channel);
			}
			switch(volumeInfo.GetCompressionType())
			{
				case COMPRESSION_FIXEDQUANT:
					decompressVolumeFloatMultiChannel(*pShared, *pRes, channels.data(), channelCountThisPass, size.x(), size.y(), size.z(), 2, volumeInfo.GetUseLessRLE());
					break;

				case COMPRESSION_FIXEDQUANT_QF:
					decompressVolumeFloatQuantFirstMultiChannel(*pShared, *pRes, channels.data(), channelCountThisPass, size.x(), size.y(), size.z(), 2, volumeInfo.GetUseLessRLE());
					break;

				default: assert(false);
			}
		}
	}
	else if(volumeInfo.GetCompressionType() == COMPRESSION_ADAPTIVEQUANT)
	{
		printf("COMPRESSION_ADAPTIVEQUANT is not supported anymore!\n");
		exit(42);
	}
	else
	{
		printf("Unknown/unsupported compression mode %u\n", uint(volumeInfo.GetCompressionType()));
		exit(42);
	}
	//if(false) {
	//	std::vector<float> data(size.volume());
	//	cudaSafeCall(cudaMemcpy(data.data(), pdpBuffer[0], size.volume()*sizeof(float), cudaMemcpyDeviceToHost));
	//	FILE*file=fopen("E:\\volx.raw", "wb");
	//	fwrite(data.data(), sizeof(float), data.size(), file);
	//	fclose(file);
	//	cudaSafeCall(cudaMemcpy(data.data(), pdpBuffer[1], size.volume()*sizeof(float), cudaMemcpyDeviceToHost));
	//	file=fopen("E:\\voly.raw", "wb");
	//	fwrite(data.data(), sizeof(float), data.size(), file);
	//	fclose(file);
	//	cudaSafeCall(cudaMemcpy(data.data(), pdpBuffer[2], size.volume()*sizeof(float), cudaMemcpyDeviceToHost));
	//	file=fopen("E:\\volz.raw", "wb");
	//	fwrite(data.data(), sizeof(float), data.size(), file);
	//	fclose(file);
	//}
	// fill brick slot (ie texture)
	if(pBrickSlot)
	{
		pBrickSlot->FillFromGPUChannels(const_cast<const float**>(pdpBuffer), size, slotIndex);
	}

	cudaSafeCall(cudaHostUnregister(const_cast<void*>(brick.GetData())));

	if(pTimer)
		pTimer->StopCurrentTimer();
}

void UploadBrick(GPUResources* pShared, CompressVolumeResources* pRes,
				 const std::vector<uint*>& data, const std::vector<uint>& dataSize,
				 const Vec3ui& brickSize,
				 eCompressionType compressionType, const std::vector<float>& quantSteps, bool lessRLE,
				 float** pdpBuffer,
				 BrickSlot* pBrickSlot, const tum3D::Vec3ui& slotIndex, int textureIndex,
				 MultiTimerGPU* pTimer)
{
	if(pTimer)
		pTimer->StartNextTimer();

	uint channelCount = uint(data.size());

	//TODO multi-channel decoding

	// upload/decompress
	for(uint channel = 0; channel < channelCount; channel++)
	{
		const uint* pChannelData = data[channel];
		uint channelDataSize = dataSize[channel];

		cudaSafeCall(cudaHostRegister(const_cast<uint*>(pChannelData), channelDataSize, cudaHostRegisterDefault));

		switch(compressionType)
		{
			case COMPRESSION_NONE:
				// uncompressed data, upload directly
				cudaMemcpyAsync(pdpBuffer[channel], pChannelData, channelDataSize, cudaMemcpyHostToDevice);
				break;

			case COMPRESSION_FIXEDQUANT:
				decompressVolumeFloat(*pShared, *pRes, pdpBuffer[channel], brickSize.x(), brickSize.y(), brickSize.z(), 2, pChannelData, channelDataSize * 8, quantSteps[channel], lessRLE);
				break;

			case COMPRESSION_FIXEDQUANT_QF:
				decompressVolumeFloatQuantFirst(*pShared, *pRes, pdpBuffer[channel], brickSize.x(), brickSize.y(), brickSize.z(), 2, pChannelData, channelDataSize * 8, quantSteps[channel], lessRLE);
				break;

			default: assert(false);
		}

		cudaSafeCall(cudaHostUnregister(const_cast<uint*>(pChannelData)));
	}
	// fill brick slot (ie texture)
	if(pBrickSlot)
	{
		if(textureIndex < 0)
		{
			assert(channelCount >= pBrickSlot->GetChannelCount());
			pBrickSlot->FillFromGPUChannels(const_cast<const float**>(pdpBuffer), brickSize, slotIndex);
		}
		else
		{
			pBrickSlot->FillTextureFromGPUChannels(textureIndex, const_cast<const float**>(pdpBuffer), brickSize, slotIndex);
		}
	}

	if(pTimer)
		pTimer->StopCurrentTimer();
}

void UploadBrickChannel(GPUResources* pShared, CompressVolumeResources* pRes,
						const TimeVolumeInfo& volumeInfo, const TimeVolumeIO::Brick& brick,
						int channel,
						float* dpBuffer,
						MultiTimerGPU* pTimer)
{
	if(pTimer)
		pTimer->StartNextTimer();

	cudaSafeCall(cudaHostRegister(const_cast<void*>(brick.GetChannelData(channel)), brick.GetChannelDataSize(channel), cudaHostRegisterDefault));

	// upload/decompress
	const Vec3ui& size = brick.GetSize();

	switch(volumeInfo.GetCompressionType())
	{
		case COMPRESSION_NONE:
		{
			// uncompressed data, upload directly
			cudaMemcpyAsync(dpBuffer, brick.GetChannelData(channel), brick.GetChannelDataSize(channel), cudaMemcpyHostToDevice);
			//TODO: before unloading a brick, we'd have to sync on the uploads to complete...
		} break;

		case COMPRESSION_FIXEDQUANT:
		{
			const uint* pBits = static_cast<const uint*>(brick.GetChannelData(channel));
			uint bitCount = uint(brick.GetChannelDataSize(channel) * 8);
			float quantStep = volumeInfo.GetQuantStep(channel);
			decompressVolumeFloat(*pShared, *pRes, dpBuffer, size.x(), size.y(), size.z(), 2, pBits, bitCount, quantStep, volumeInfo.GetUseLessRLE());
		} break;

		case COMPRESSION_FIXEDQUANT_QF:
		{
			const uint* pBits = static_cast<const uint*>(brick.GetChannelData(channel));
			uint bitCount = uint(brick.GetChannelDataSize(channel) * 8);
			float quantStep = volumeInfo.GetQuantStep(channel);
			decompressVolumeFloatQuantFirst(*pShared, *pRes, dpBuffer, size.x(), size.y(), size.z(), 2, pBits, bitCount, quantStep, volumeInfo.GetUseLessRLE());
		} break;

		default: assert(false);
	}

	cudaSafeCall(cudaHostUnregister(const_cast<void*>(brick.GetChannelData(channel))));

	if(pTimer)
		pTimer->StopCurrentTimer();
}

void UploadBrickChannel(GPUResources* pShared, CompressVolumeResources* pRes,
						const uint* pData, uint dataSize,
						const Vec3ui& brickSize,
						eCompressionType compressionType, float quantStep, bool lessRLE,
						int channel,
						float* dpBuffer,
						MultiTimerGPU* pTimer)
{
	if(pTimer)
		pTimer->StartNextTimer();

	cudaSafeCall(cudaHostRegister(const_cast<uint*>(pData), dataSize, cudaHostRegisterDefault));

	// upload/decompress
	switch(compressionType)
	{
		case COMPRESSION_NONE:
			// uncompressed data, upload directly
			cudaMemcpyAsync(dpBuffer, pData, dataSize, cudaMemcpyHostToDevice);
			break;

		case COMPRESSION_FIXEDQUANT:
			decompressVolumeFloat(*pShared, *pRes, dpBuffer, brickSize.x(), brickSize.y(), brickSize.z(), 2, pData, dataSize * 8, quantStep, lessRLE);
			break;

		case COMPRESSION_FIXEDQUANT_QF:
			decompressVolumeFloatQuantFirst(*pShared, *pRes, dpBuffer, brickSize.x(), brickSize.y(), brickSize.z(), 2, pData, dataSize * 8, quantStep, lessRLE);
			break;

		default: assert(false);
	}

	cudaSafeCall(cudaHostUnregister(const_cast<uint*>(pData)));

	if(pTimer)
		pTimer->StopCurrentTimer();
}
