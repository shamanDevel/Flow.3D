#ifndef __TUM3D__WORKER_THREAD_H__
#define __TUM3D__WORKER_THREAD_H__


#include "global.h"

#include <tinythread.h>

#include "Range.h"
#include "RenderingManager.h"

#include "CompressVolume.h"
#include "GPUResources.h"


class WorkerThread
{
public:
	WorkerThread(int cudaDevice, const TimeVolume& volume);
	~WorkerThread();

	// start worker thread
	bool Start();
	// wait until all pending work is done and then exit thread
	bool Stop();

	void CreateVolumeDependentResources();
	void ReleaseVolumeDependentResources();
	void SetProjectionParams(const ProjectionParams& projParams, const Range1D& range);

	void StartRendering(const std::vector<FilteredVolume>& filteredVolumes, ViewParams viewParams, StereoParams stereoParams, RaycastParams raycastParams, cudaArray* pTransferFunction, int transferFunctionDevice);
	void CancelRendering();
	// tell the thread that the current task may be canceled as soon as possible
	void CancelCurrentTask();
	bool IsWorking() const { return m_isWorking; }
	float GetProgress() const { return m_progress; }

	// returns timestamp of last change
	uint LockResultImage(byte*& pData);
	void UnlockResultImage();

private:
	void Run();
	static void ThreadFunc(void* pParam);


	void TaskCreateVolumeDependentResources();
	void TaskReleaseVolumeDependentResources();

	void TaskSetProjParams();

	void TaskRender();


	const int m_cudaDevice;

	const TimeVolume* m_pVolume;


	tthread::thread*            m_pThread;
	bool                        m_threadStarted;
	tthread::mutex              m_mutexThreadStarted;
	tthread::condition_variable m_condThreadStarted;

	tthread::mutex              m_mutex;
	tthread::condition_variable m_condNewTask;


	bool m_isWorking;
	float m_progress;

	enum ETask
	{
		TASK_NONE,
		TASK_STOP_THREAD,
		TASK_CREATE_VOLUME_DEPENDENT_RESOURCES,
		TASK_RELEASE_VOLUME_DEPENDENT_RESOURCES,
		TASK_SET_PROJ_PARAMS,
		TASK_RENDER
	};
	bool m_cancelCurrentTask;


	// access to this must be protected by the mutex
	struct Shared
	{
		Shared() : task(TASK_NONE), pFilteredVolumes(nullptr), pTransferFunction(nullptr), transferFunctionDevice(-1) {}

		ETask task;

		ProjectionParams projParams;
		Range1D range;

		const std::vector<FilteredVolume>* pFilteredVolumes;
		ViewParams viewParams;
		StereoParams stereoParams;
		RaycastParams raycastParams;
		ParticleTraceParams particleTraceParams;
		ParticleRenderParams particleRenderParams;
		cudaArray* pTransferFunction;
		int transferFunctionDevice;
	} m_shared;

	// this must *only* be accessed by the thread
	struct Private
	{
		GPUResources            compressShared;
		CompressVolumeResources compressVolume;
		RenderingManager        renderingManager;
	} m_private;


	byte*          m_pResultImage[2];
	uint           m_resultImageTimestamp[2];
	int            m_resultImageIndex;
	cudaEvent_t    m_resultImageDownloadSyncEvent;
	tthread::mutex m_mutexResultImage;


	// disallow copy and assignment
	WorkerThread(const WorkerThread&);
	WorkerThread& operator=(const WorkerThread&);
};


#endif
