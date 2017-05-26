#include "WorkerThread.h"

#include <cassert>

#include <cuda_runtime.h>

#include <cudaCompress/Init.h>

#include "cudaUtil.h"


WorkerThread::WorkerThread(int cudaDevice, const TimeVolume& volume)
	: m_cudaDevice(cudaDevice)
	, m_pVolume(&volume)
	, m_pThread(nullptr), m_threadStarted(false)
	, m_isWorking(false), m_progress(0.0f)
	, m_cancelCurrentTask(false)
{
	m_pResultImage[0] = nullptr;
	m_pResultImage[1] = nullptr;
	m_resultImageTimestamp[0] = 0;
	m_resultImageTimestamp[1] = 0;
	m_resultImageIndex = 0;
	m_resultImageDownloadSyncEvent = 0;
}

WorkerThread::~WorkerThread()
{
	Stop();
}


bool WorkerThread::Start()
{
	if(m_pThread != nullptr)
	{
		// already started
		return false;
	}

	// wait until thread is actually started and has grabbed m_mutex
	// (otherwise the first task can be queued before the thread is ready and may be skipped)
	{ tthread::lock_guard<tthread::mutex> guard(m_mutexThreadStarted);
		m_pThread = new tthread::thread(ThreadFunc, this);
		while(!m_threadStarted)
		{
			m_condThreadStarted.wait(m_mutexThreadStarted);
		}
	}

	return true;
}

bool WorkerThread::Stop()
{
	if(m_pThread == nullptr)
	{
		return false;
	}

	// set "please close" flag for thread
	{ tthread::lock_guard<tthread::mutex> guard(m_mutex);
		m_shared.task = TASK_STOP_THREAD;

		m_condNewTask.notify_all();
	}

	// wait for thread to actually finish
	m_pThread->join();
	delete m_pThread;
	m_pThread = nullptr;

	return true;
}


void WorkerThread::CreateVolumeDependentResources()
{
	{ tthread::lock_guard<tthread::mutex> guard(m_mutex);
		assert(m_threadStarted);
		assert(m_shared.task == TASK_NONE);
		m_cancelCurrentTask = false;

		m_shared.task = TASK_CREATE_VOLUME_DEPENDENT_RESOURCES;

		m_condNewTask.notify_all();
	}
}

void WorkerThread::ReleaseVolumeDependentResources()
{
	{ tthread::lock_guard<tthread::mutex> guard(m_mutex);
		assert(m_threadStarted);
		assert(m_shared.task == TASK_NONE);
		m_cancelCurrentTask = false;

		m_shared.task = TASK_RELEASE_VOLUME_DEPENDENT_RESOURCES;

		m_condNewTask.notify_all();
	}
}


void WorkerThread::SetProjectionParams(const ProjectionParams& projParams, const Range1D& range)
{
	{ tthread::lock_guard<tthread::mutex> guard(m_mutex);
		assert(m_threadStarted);
		assert(m_shared.task == TASK_NONE);
		m_cancelCurrentTask = false;

		m_shared.task = TASK_SET_PROJ_PARAMS;
		m_shared.projParams = projParams;
		m_shared.range = range;

		m_condNewTask.notify_all();
	}
}


void WorkerThread::StartRendering(const std::vector<FilteredVolume>& filteredVolumes, ViewParams viewParams, StereoParams stereoParams, RaycastParams raycastParams, cudaArray* pTransferFunction, int transferFunctionDevice)
{
	{ tthread::lock_guard<tthread::mutex> guard(m_mutex);
		assert(m_threadStarted);
		assert(m_shared.task == TASK_NONE);
		m_cancelCurrentTask = false;

		m_shared.task = TASK_RENDER;
		m_shared.pFilteredVolumes = &filteredVolumes;
		m_shared.viewParams = viewParams;
		m_shared.stereoParams = stereoParams;
		m_shared.raycastParams = raycastParams;
		m_shared.pTransferFunction = pTransferFunction;
		m_shared.transferFunctionDevice = transferFunctionDevice;

		m_condNewTask.notify_all();
	}
}

void WorkerThread::CancelRendering()
{
	// this "should" be threadsafe (32-bit reads/writes are atomic,
	//  and the thread will only change m_shared.task back to TASK_NONE,
	//  in which case setting the cancel flag is harmless)
	if(m_shared.task == TASK_RENDER)
	{
		CancelCurrentTask();
	}
}


void WorkerThread::CancelCurrentTask()
{
	// don't lock, just set the flag
	m_cancelCurrentTask = true;
}


uint WorkerThread::LockResultImage(byte*& pData)
{
	//TODO check if the other image is ready (cudaEventQuery)
	m_mutexResultImage.lock();
	pData = m_pResultImage[m_resultImageIndex];
	return m_resultImageTimestamp[m_resultImageIndex];
}

void  WorkerThread::UnlockResultImage()
{
	m_mutexResultImage.unlock();
}


void WorkerThread::Run()
{
	tthread::lock_guard<tthread::mutex> guard(m_mutex);

	m_threadStarted = true;
	m_condThreadStarted.notify_all();

	cudaSafeCall(cudaSetDevice(m_cudaDevice));

	cudaSafeCall(cudaEventCreate(&m_resultImageDownloadSyncEvent, cudaEventDisableTiming));
	cudaSafeCall(cudaEventRecord(m_resultImageDownloadSyncEvent));

	bool stop = false;
	while(!stop)
	{
		// wait until we're notified there's something to do
		m_condNewTask.wait(m_mutex);
		m_isWorking = true;

		// check what we're supposed to do
		switch(m_shared.task)
		{
			case TASK_NONE:
				assert(false);
				break;

			case TASK_STOP_THREAD:
				stop = true;
				break;

			case TASK_CREATE_VOLUME_DEPENDENT_RESOURCES:
				TaskCreateVolumeDependentResources();
				break;

			case TASK_RELEASE_VOLUME_DEPENDENT_RESOURCES:
				TaskReleaseVolumeDependentResources();
				break;

			case TASK_SET_PROJ_PARAMS:
				TaskSetProjParams();
				break;

			case TASK_RENDER:
				TaskRender();
				break;
		}

		// reset task
		m_shared.task = TASK_NONE;
		m_cancelCurrentTask = false;
		m_isWorking = false;
	}

	TaskReleaseVolumeDependentResources();

	cudaSafeCall(cudaFreeHost(m_pResultImage[1]));
	m_pResultImage[1] = nullptr;
	cudaSafeCall(cudaFreeHost(m_pResultImage[0]));
	m_pResultImage[0] = nullptr;

	cudaSafeCall(cudaEventDestroy(m_resultImageDownloadSyncEvent));
	m_resultImageDownloadSyncEvent = 0;

	cudaSafeCall(cudaDeviceReset());
}


void WorkerThread::ThreadFunc(void* pParam)
{
	WorkerThread* pThread = (WorkerThread*)pParam;
	pThread->Run();
}


void WorkerThread::TaskCreateVolumeDependentResources()
{
	TaskReleaseVolumeDependentResources();

	if(m_pVolume->IsCompressed())
	{
		uint brickSize = m_pVolume->GetBrickSizeWithOverlap();
		// do multi-channel decoding only for small bricks; for large bricks, mem usage gets too high
		uint channelCount = (brickSize <= 128) ? m_pVolume->GetChannelCount() : 1;
		uint huffmanBits = m_pVolume->GetHuffmanBitsMax();
		m_private.compressShared.create(CompressVolumeResources::getRequiredResources(brickSize, brickSize, brickSize, channelCount, huffmanBits));
		m_private.compressVolume.create(m_private.compressShared.getConfig());
	}
	m_private.renderingManager.Create(&m_private.compressShared, &m_private.compressVolume, nullptr);
}

void WorkerThread::TaskReleaseVolumeDependentResources()
{
	m_private.renderingManager.Release();
	m_private.compressVolume.destroy();
	m_private.compressShared.destroy();
}


void WorkerThread::TaskSetProjParams()
{
	m_private.renderingManager.SetProjectionParams(m_shared.projParams, m_shared.range);

	uint width  = m_shared.projParams.GetImageWidth (m_shared.range);
	uint height = m_shared.projParams.GetImageHeight(m_shared.range);

	cudaSafeCall(cudaFreeHost(m_pResultImage[0]));
	cudaSafeCall(cudaFreeHost(m_pResultImage[1]));
	cudaSafeCall(cudaMallocHost(&m_pResultImage[0], width * height * sizeof(uchar4)));
	cudaSafeCall(cudaMallocHost(&m_pResultImage[1], width * height * sizeof(uchar4)));
}


void WorkerThread::TaskRender()
{
	std::vector<LineBuffers*> lineBuffers;
	std::vector<BallBuffers*> ballBuffers;
	RenderingManager::eRenderState state = m_private.renderingManager.StartRendering(
		*m_pVolume, *m_shared.pFilteredVolumes,
		m_shared.viewParams, m_shared.stereoParams,
		false, false, false, false,
		m_shared.particleTraceParams, m_shared.particleRenderParams, lineBuffers, false, ballBuffers, 0.0f, NULL,
		m_shared.raycastParams, m_shared.pTransferFunction, m_shared.transferFunctionDevice);
	if(state != RenderingManager::STATE_RENDERING) //FIXME ?
	{
		return;
	}

	m_progress = 0.0f;

	uint width  = m_shared.projParams.GetImageWidth (m_shared.range);
	uint height = m_shared.projParams.GetImageHeight(m_shared.range);

	assert(m_pResultImage[0] != nullptr);
	memset(m_pResultImage[0], 0, width * height * sizeof(uchar4));
	memset(m_pResultImage[1], 0, width * height * sizeof(uchar4));
	m_resultImageTimestamp[0] = 0;
	m_resultImageTimestamp[1] = 0;

	bool finished = false;
	do
	{
		if(m_cancelCurrentTask)
		{
			m_private.renderingManager.CancelRendering();
			break;
		}

		finished = (m_private.renderingManager.Render() == RenderingManager::STATE_DONE);
		m_progress = m_private.renderingManager.GetRenderingProgress();

		// download result
		// if previous download isn't finished yet, don't bother starting the next one
		//TODO one event per buffer?
		bool previousDownloadFinished = (cudaSuccess == cudaEventQuery(m_resultImageDownloadSyncEvent));
		if(previousDownloadFinished || finished)
		{
			// first make sure previous download is done
			cudaSafeCall(cudaEventSynchronize(m_resultImageDownloadSyncEvent));
			// start download and swap buffers
			{ tthread::lock_guard<tthread::mutex> guard(m_mutexResultImage);
				cudaSafeCall(cudaMemcpyFromArrayAsync(m_pResultImage[m_resultImageIndex], m_private.renderingManager.GetRaycastArray(), 0, 0, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));
				m_resultImageTimestamp[m_resultImageIndex] = m_resultImageTimestamp[1 - m_resultImageIndex] + 1;
				m_resultImageIndex = 1 - m_resultImageIndex;
			}
			// record event to check for download completion
			cudaSafeCall(cudaEventRecord(m_resultImageDownloadSyncEvent));
		}
	}
	while(!finished);

	if(finished)
	{
		// wait for last download to finish
		cudaSafeCall(cudaEventSynchronize(m_resultImageDownloadSyncEvent));
		{ tthread::lock_guard<tthread::mutex> guard(m_mutexResultImage);
			m_resultImageIndex = 1 - m_resultImageIndex;
		}
	}


	//if(finished) {
	//	printf("Device %i: Writing result to file...", m_cudaDevice);
	//	cudaArray* pArray = m_private.renderingManager.GetRaycastArray();
	//	cudaExtent extent;
	//	cudaArrayGetInfo(nullptr, &extent, nullptr, pArray);
	//	std::vector<uchar4> data(extent.width * extent.height);
	//	cudaMemcpyFromArray(data.data(), pArray, 0, 0, extent.width * extent.height * 4, cudaMemcpyDeviceToHost);
	//	FILE* file = fopen("snapshot.raw", "wb");
	//	if(!file) {
	//		printf(" failed!\n");
	//	} else {
	//		fwrite(data.data(), extent.width * extent.height * 4, 1, file);
	//		fclose(file);
	//		printf(" done.\n");
	//	}
	//}
}
