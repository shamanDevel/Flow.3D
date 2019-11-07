#pragma once

#include <global.h>

#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <Windows.h>

#include <Utils.h>
#include <MemUtils.h>
#include <TimerCPU.h>
#include <Vec.h>
#include <SysTools.h>

#include "FilePool.h"
#include "TimeVolumeInfo.h"
#include "../Renderer/Measure.h"

using tum3D::Vec3i;
using tum3D::Vec3ui;


// ***************************************
// TimeVolumeIO
// ***************************************

/** Async I/O for time volume files

KNOWN BUGS
	- A timestep file is always recreated when timesteps are written out of order

USAGE
	- Create a new file with TimeVolumeIO::Create()
	- Write all bricks of all timesteps in sequential order using TimeVolumeIO::AddBrick()
		-> Might also be written out of order, but this is bugged right now and why would you
		want to do that?
	- Call TimeVolumeIO::Close() to write the index file

	- Open an existing file with TimeVolumeIO::Open()
	- Schedule bricks for loading using TimeVolumeIO::EnqueueLoadBrickData()
	- A brick has finished loading when Brick::GetData() does not return nullptr
	- Use TimeVolumeIO::WaitForAllIO() to execute blocking read
	- Call TimeVolumeIO::UpdateAysncIO() once per frame to populate the loading slots

USAGE HINTS
	- Only a single timestep can be read in parallel, so try to batch loads by timestep-index
		Might be fixed in the future, but I don't really see a need for this right now

WARNING This class uses fseek to navigate through the index file
This will fail if the index file is > 2 GB. However, I cannot see this happening...
*/

class TimeVolumeIO
{
public:
	static const uint ms_channelCountMax = 4;

	// ***** Inner structs ****

	class Brick
	{
		friend class TimeVolumeIO;

	public:
		enum LoadingState
		{
			LOADING_STATE_UNLOADED,
			LOADING_STATE_QUEUED,
			LOADING_STATE_LOADING,
			LOADING_STATE_FINISHED
		};

		Brick()
			: m_exists(false)
			, m_state(LOADING_STATE_UNLOADED)
			, m_pData(nullptr)
		{}

		bool Exists() const { return m_exists; }

		const Vec3i& GetSpatialIndex() const { return m_spatialIndex; }
		const Vec3ui& GetSize() const { return m_size; }
		int32 GetTimestep() const { return m_timestep; }

		const void* GetData() const  { return IsLoaded() ? m_pData : nullptr; }
		const void* GetChannelData(int32 channel) const  {  return IsLoaded() ? m_ppChannelData[channel] : nullptr; }

		int64 GetDataSize() const  { return m_bytesize; }
		int64 GetPaddedDataSize() const  { return m_bytesizePadded; }
		int64 GetChannelDataSize(int32 channel) const  { return m_pChannelSize[channel]; }

		LoadingState GetLoadingState() const { return m_state; }
		bool IsLoaded() const { return m_state == LOADING_STATE_FINISHED; }

		const float* GetMinMeasuresInBrick() const { return m_pMinMeasuresInBrick; }
		const float* GetMaxMeasuresInBrick() const  { return m_pMaxMeasuresInBrick; }

	private:
		bool		m_exists;

		int32		m_timestep;
		Vec3i		m_spatialIndex;
		Vec3ui		m_size;

		int64		m_fileOffset;

		void*		m_pData;				///< nullptr as long as no data is present
		//int64*	m_pChannelSize;
		//void**	m_ppChannelData;
		int64		m_pChannelSize[ms_channelCountMax];
		void*		m_ppChannelData[ms_channelCountMax];
		LoadingState	m_state;

		// The minimum and maximum values for all measures in our brick.
		float		m_pMinMeasuresInBrick[NUM_MEASURES];
		float		m_pMaxMeasuresInBrick[NUM_MEASURES];

		int64		m_bytesize;			///< Usable bytesize of this brick
		int64		m_bytesizePadded;

		std::list<Brick*>::const_iterator	m_lruIt;
	};


	struct Timestep
	{
		int32				index;
		std::vector<Brick>	bricks; // in xyz order

		Timestep(): index(-1) {}
	};



	// ***** Construction / Destruction *****

	TimeVolumeIO(float memoryUsageLimit = 0.7f);
	~TimeVolumeIO();

	bool Open(const std::string& indexFileName, bool writeAccess = false);
	void Create(const std::string& indexFileName, const Vec3i& volumeSize, bool periodic, int32 brickSize, int32 overlap,
		int32 channels, const std::string& timestepFileNamePrefix = "");

	const std::string& GetFilename() const { return m_strFilename; }
	const std::string& GetName() const { return m_strName; }

	void Close();

	bool IsOpen() const { return m_bIsOpen; }







	// ***** Data management *****

	/// Purge bricks from LRU cache if we're over the memory budget, call once per frame or so
	void UnloadLRUBricks();

	bool AllIOIsFinished();
	/// Wait until all IO has finished
	void WaitForAllIO(bool discardQueued = false);
	/// Discard all IO that has not been started yet
	void DiscardQueuedIO();
	/// Discard all IO, even operations that have been started already
	void DiscardAllIO();

	/** Adds a brick with given data to the file.
	NOTE: The brick's data will _not_ be set to the passed pointer; the brick is still
	considered as "not loaded"
	*/
	template<typename T>
	inline Brick& AddBrick(int32 timestep, const Vec3i& spatialIndex, const Vec3ui& size, const std::vector<std::vector<T>>& channelData,
			const std::vector<float>& minMeasuresInBrick, const std::vector<float>& maxMeasuresInBrick);

	/** Load a brick's data
	This load will be performed asynchronously, but can be blocked using WaitForAllIO()
	Returns true if brick was successfully enqueued, or was already loaded/queued
	*/
	bool EnqueueLoadBrickData(Brick& brick);
	/// Unload a brick's data
	void UnloadBrickData(Brick& brick);


	/// Updates a brick in the lru queue
	void TouchBrick(Brick* brick);



	bool GetPageLockBrickData() const { return m_bPageLockBricks; }
	void SetPageLockBrickData(bool lock);



	// ***** Setter / Getter *****
	const TimeVolumeInfo& GetInfo() const { return m_info; }

	const Vec3i& GetVolumeSize() const { return m_info.GetVolumeSize(); }
	int32 GetChannelCount() const { return m_info.GetChannelCount(); }

	tum3D::Vec3f GetGridSpacing() const { return m_info.GetGridSpacing(); }
	void SetGridSpacing(const tum3D::Vec3f& value) { m_info.m_fGridSpacing = value; }
	float GetTimeSpacing() const { return m_info.GetTimeSpacing(); }
	void SetTimeSpacing(float value) { m_info.m_fTimeSpacing = value; }

	bool IsPeriodic() const { return m_info.IsPeriodic(); }

	int GetFloorTimestepIndex  (float time) const { return m_info.GetFloorTimestepIndex(time); }
	int GetCeilTimestepIndex   (float time) const { return m_info.GetCeilTimestepIndex(time); }
	int GetNearestTimestepIndex(float time) const { return m_info.GetNearestTimestepIndex(time); }

	int32 GetBrickSizeWithOverlap() const { return m_info.GetBrickSizeWithOverlap(); }
	int32 GetBrickSizeWithoutOverlap() const { return m_info.GetBrickSizeWithoutOverlap(); }
	int32 GetBrickOverlap() const { return m_info.GetBrickOverlap(); }
	Vec3i GetBrickCount() const { return m_info.GetBrickCount(); }

	uint GetBrickLinearIndex(const Vec3i& spatialIndex) const { return m_info.GetBrickLinearIndex(spatialIndex); }
	Vec3i GetBrickSpatialIndex(uint linearIndex) const { return m_info.GetBrickSpatialIndex(linearIndex); }

	tum3D::Vec3f GetVolumeHalfSizeWorld() const { return m_info.GetVolumeHalfSizeWorld(); }
	tum3D::Vec3f GetBrickSizeWorld() const { return m_info.GetBrickSizeWorld(); }
	tum3D::Vec3f GetBrickOverlapWorld() const { return m_info.GetBrickOverlapWorld(); }
	bool GetBrickBoxWorld(const tum3D::Vec3i& spatialIndex, tum3D::Vec3f& boxMin, tum3D::Vec3f& boxMax) const { return m_info.GetBrickBoxWorld(spatialIndex, boxMin, boxMax); }

	const Brick& GetBrick(uint timestep, uint linearIndex) const { return m_vTimesteps[timestep].bricks[linearIndex]; }
	const Brick& GetBrick(uint timestep, const Vec3i& spatialIndex) const { return GetBrick(timestep, GetBrickLinearIndex(spatialIndex)); }

	bool IsCompressed() const { return m_info.IsCompressed(); }
	eCompressionType GetCompressionType() const { return m_info.GetCompressionType(); }
	void SetCompressionType(eCompressionType value) 
	{
		E_ASSERT("Cannot change compression property in read access", m_openMode == OPEN_MODE_WRITE); 
		m_info.m_compression = value; 
	}
	const std::vector<float>& GetQuantSteps() const { return m_info.GetQuantSteps(); }
	float GetQuantStep(int32 channel) const { return m_info.GetQuantStep(channel); }
	void SetQuantStep(int32 channel, float value)
	{
		E_ASSERT("Cannot change quantization step property in read access", m_openMode == OPEN_MODE_WRITE);
		m_info.m_vQuantSteps[channel] = value;
	}
	bool GetUseLessRLE() const { return m_info.GetUseLessRLE(); }
	void SetUseLessRLE(bool value)
	{
		E_ASSERT("Cannot change lessrle property in read access", m_openMode == OPEN_MODE_WRITE);
		m_info.m_lessRLE = value;
	}
	uint GetHuffmanBitsMax() const { return m_info.GetHuffmanBitsMax(); }
	void SetHuffmanBitsMax(uint value)
	{
		E_ASSERT("Cannot change huffmanbits property in read access", m_openMode == OPEN_MODE_WRITE);
		m_info.m_huffmanBitsMax = value;
	}


	Timestep& GetTimestep(int32 index) { return m_vTimesteps[index]; }
	const Timestep& GetTimestep(int32 index) const { return m_vTimesteps[index]; }
	int32 GetTimestepCount() const { return (int32)m_vTimesteps.size(); }


	void SetLoadingQueueLimit(int32 value) { m_iLoadingQueueLimit = value; }
	int32 GetLoadingQueueLimit() const { return m_iLoadingQueueLimit; }

	int32 GetLoadingQueueSize() const { return (int32)m_loadingQueue.size(); }


	float GetTotalDiskBusyTimeMS() const { return m_diskBusyTimeMS; }

	size_t GetTotalLoadedBytes() const { return m_dataLoadedBytesTotal; }

	SystemMemoryUsage& GetSystemMemoryUsage() { return m_memUsage; }
	const SystemMemoryUsage& GetSystemMemoryUsage() const { return m_memUsage; }


protected:
	static const int32 ms_loadingSlotCount = 1;

private:
	#pragma pack(push, 1)
	// note: these two structs are dumped straight into the file, so don't change!
	struct FileTimestepHeader
	{
		int32		index;				///< Absolute index of this timestep
		int32		brickListOffset;	///< Byteoffset of the first brick desc in the index file
		int32		numBricks;			///< Number of bricks in this timestep
	};

	struct BrickDesc
	{
		Vec3i		spatialIndex;		///< Spatial brick id x/y/z
		int64		dataOffset;			///< Byteoffset in the timestep-file
		int64		bytesize;			///< Usable bytesize of this brick
		int64		paddedBytesize;		///< On-disk size of this brick
		Vec3ui		bricksize;			///< Real bricksize, might differ from default for border bricks

		// The minimum and maximum values for all measures in our brick.
		float		m_pMinMeasuresInBrick[NUM_MEASURES];
		float		m_pMaxMeasuresInBrick[NUM_MEASURES];
	};
	#pragma pack(pop)


	enum OpenMode
	{
		OPEN_MODE_READ,
		OPEN_MODE_WRITE
	};

	struct LoadingSlot
	{
		OVERLAPPED	overlapped;
		HANDLE		hEvent;

		HANDLE		hFile;
		Brick*		pBrick;

		LoadingSlot(): hFile(INVALID_HANDLE_VALUE), pBrick(nullptr)
		{
			hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
			ZeroMemory(&overlapped, sizeof(overlapped));
			overlapped.hEvent = hEvent;
		}
		~LoadingSlot()
		{
			assert(!InUse());
			CloseHandle(hEvent);
		}

		bool InUse() const { return pBrick != nullptr; }
	};


	void LoadingThreadFunc();
	void UpdateAsyncIO(bool blockingRead = false);
	void ReserveBrickData(Brick* brick);


	void ReadHeaders(FILE* indexFile);
	void ReadSettings(FILE* indexFile);
	void ReadTimesteps(FILE* indexFile);

	void WriteHeaders(FILE* indexFile);
	void WriteSettings(FILE* indexFile);
	void WriteTimesteps(FILE* indexFile);


	bool StartLoadingThread();
	bool ShutdownLoadingThread();
	void WakeupLoadingThread();


	OpenMode			m_openMode;
	bool				m_bIsOpen;


	std::string			m_strBaseFilePath;
	std::string			m_strFilename;
	std::string			m_strName;
	std::string			m_strTimestepFileNamePrefix;
	int32				m_iPadding;


	TimeVolumeInfo		m_info;

	std::vector<Timestep>	m_vTimesteps;
	std::list<Brick*>		m_lruCache;
	std::recursive_mutex	m_lruMutex;

	SystemMemoryUsage	m_memUsage;

	bool				m_bPageLockBricks;

	// output only:
	HANDLE				m_hCurrentOutputFile;
	int32				m_iCurrentOutputTimestep;

	// input only:
	FilePool			m_filePool;

	std::thread			m_loadingThread;
	bool				m_shutdownLoadingThread;

	LoadingSlot			m_loadingSlots[ms_loadingSlotCount];
	HANDLE				m_wakeupEvents[ms_loadingSlotCount + 1]; // one per loading slot ("IO finished"), plus one generic wakeup event
	uint32				m_iBricksLoading; // ie occupied loading slots

	std::mutex			m_loadingQueueMutex;
	std::queue<Brick*>	m_loadingQueue;
	int32				m_iLoadingQueueLimit;


	TimerCPU			m_diskBusyTimer;
	float				m_diskBusyTimeMS;

	size_t				m_dataLoadedBytesTotal;


	// disallow copy and assignment
	TimeVolumeIO(const TimeVolumeIO&);
	TimeVolumeIO& operator=(const TimeVolumeIO&);
};






const char TIMESTEP_FILE_MASK[] = "%05d";



// ************** Implementation ****************************
template<typename T>
TimeVolumeIO::Brick& TimeVolumeIO::AddBrick(int32 timestep, const Vec3i& spatialIndex, const Vec3ui& size, const std::vector<std::vector<T>>& channelData,
		const std::vector<float>& minMeasuresInBrick, const std::vector<float>& maxMeasuresInBrick)
{
	assert("Invalid timestep" && (timestep >= 0));

	assert("channelData element count does not match #channels" && (channelData.size() == m_info.m_iChannels));

	// Create timestep if neccessary
	if (timestep >= m_vTimesteps.size())
	{
		int32 oldTsCount = (int32)m_vTimesteps.size();
		m_vTimesteps.resize(timestep + 1);

		for (int32 i = oldTsCount; i < m_vTimesteps.size(); ++i)
		{
			m_vTimesteps[i].index = i;
			m_vTimesteps[i].bricks.resize(GetBrickCount().volume());
		}
	}

	int brickIndex = GetBrickLinearIndex(spatialIndex);
	Brick& brick = m_vTimesteps[timestep].bricks[brickIndex];

	brick.m_exists = true;
	brick.m_spatialIndex = spatialIndex;
	brick.m_size = size;
	brick.m_bytesize = 0;
	brick.m_lruIt = m_lruCache.cend();

	//brick.m_ppChannelData = new void*[m_info.m_iChannels];
	//brick.m_pChannelSize = new int64[m_info.m_iChannels];

	for (int32 i = 0; i < m_info.m_iChannels; ++i)
	{
		brick.m_pChannelSize[i] = channelData[i].size() * sizeof(T);
		brick.m_bytesize += brick.m_pChannelSize[i];
	}

	memcpy(brick.m_pMinMeasuresInBrick, minMeasuresInBrick.data(), sizeof(float)*NUM_MEASURES);
	memcpy(brick.m_pMaxMeasuresInBrick, maxMeasuresInBrick.data(), sizeof(float)*NUM_MEASURES);

	if (!m_info.IsCompressed())
	{
		assert("Total brick bytesize does not match #channels * brickSize" && 
			(brick.m_bytesize == m_info.m_iChannels * brick.m_size.volume() * sizeof(T)));
	}

	brick.m_bytesizePadded = RoundToNextMultiple(brick.m_bytesize, (int64)m_iPadding);

	// Open timestep file if not already open
	if (timestep != m_iCurrentOutputTimestep)
	{
		if (m_hCurrentOutputFile != INVALID_HANDLE_VALUE)
		{
			CloseHandle(m_hCurrentOutputFile);
		}

		char currentFile[1024];
		sprintf_s(currentFile, TIMESTEP_FILE_MASK, timestep);
		std::string fileName = m_strBaseFilePath + m_strTimestepFileNamePrefix + currentFile;

		DWORD fileFlags = 0;
		if (tum3d::FileExists(m_strBaseFilePath + currentFile))
		{
			fileFlags = OPEN_EXISTING;
		}
		else
		{
			//fileFlags = CREATE_NEW;
			fileFlags = CREATE_ALWAYS;
		}

		m_hCurrentOutputFile = CreateFileA(fileName.c_str(), GENERIC_WRITE, 0, 0, fileFlags, 0, 0);
		DWORD errorCode = GetLastError();
		E_ASSERT("Could not create file " << fileName + " for writing, error code: " << errorCode, m_hCurrentOutputFile != INVALID_HANDLE_VALUE);
		SetEndOfFile(m_hCurrentOutputFile);
	}

	// Append to end
	LARGE_INTEGER li;
	li.QuadPart = 0;

	m_iCurrentOutputTimestep = timestep;

	SetFilePointerEx(m_hCurrentOutputFile, li, &li, FILE_END);
	brick.m_fileOffset = (int64)li.QuadPart;
	DWORD dwBytesWritten;

	for (int i = 0; i < m_info.m_iChannels; ++i)
	{
		WriteFile(m_hCurrentOutputFile, channelData[i].data(), (DWORD)channelData[i].size() * sizeof(T), &dwBytesWritten, nullptr);
		E_ASSERT("Could not write brick data", dwBytesWritten == (DWORD)channelData[i].size() * sizeof(T));
	}

	// Set end of file to the next multiple of the padding
	GetFileSizeEx(m_hCurrentOutputFile, &li);
	li.QuadPart = RoundToNextMultiple(li.QuadPart, (int64)m_iPadding);
	SetFilePointerEx(m_hCurrentOutputFile, li, &li, FILE_BEGIN);
	SetEndOfFile(m_hCurrentOutputFile);

	/*cout << "Added brick [" << brick.m_spatialIndex[0] << ", " << brick.m_spatialIndex[1] << ", " <<
		brick.m_spatialIndex[2] << "] to timestep " << timestep << "\n";*/

	return brick;
}