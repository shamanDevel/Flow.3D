#include "TimeVolumeIO.h"

#include <cassert>
#include <algorithm>

#include <cuda_runtime.h>

#include <SysTools.h>

using namespace tum3D;

/*
* Get File Name from a Path with or without extension
*/
std::string getFileName(std::string filePath, bool withExtension = true, char seperator = '/')
{
	// Get last dot position
	std::size_t dotPos = filePath.rfind('.');
	std::size_t sepPos = filePath.rfind(seperator);

	if (sepPos != std::string::npos)
	{
		return filePath.substr(sepPos + 1, filePath.size() - (withExtension || dotPos != std::string::npos ? 1 : dotPos));
	}
	return "";
}


//#define DONT_USE_LOADING_THREAD


const char IDENT[] = "TUM3D_TIMEVOLUME";
const int32 CURRENT_VERSION = 3;
const int32 DEFAULT_PADDING = 1024;


TimeVolumeIO::TimeVolumeIO(float memoryUsageLimit):
	m_bIsOpen(false), m_iBricksLoading(0), m_iLoadingQueueLimit(std::numeric_limits<int32>::max()),
	m_memUsage(memoryUsageLimit),
	m_bPageLockBricks(false),
	m_shutdownLoadingThread(false),
	m_diskBusyTimeMS(0.0f), m_dataLoadedBytesTotal(0)
{
	float limitMB = float(m_memUsage.GetSystemMemoryLimitBytes() / (1024.0 * 1024.0));
	printf("TimeVolumeIO MemUsageLimit: %.2f MB\n", limitMB);

	for(int i = 0; i < ms_loadingSlotCount; i++)
	{
		m_wakeupEvents[i] = m_loadingSlots[i].hEvent;
	}
	m_wakeupEvents[ms_loadingSlotCount] = CreateEvent(NULL, FALSE, FALSE, NULL);
}



TimeVolumeIO::~TimeVolumeIO()
{
	Close();

	CloseHandle(m_wakeupEvents[ms_loadingSlotCount]);
}



bool TimeVolumeIO::Open(const string& indexFileName, bool writeAccess)
{
	Close();

	cout << "Opening file " << indexFileName << "...\n";

	if (writeAccess)
	{
		m_openMode = OPEN_MODE_WRITE;
	}
	else
	{
		m_openMode = OPEN_MODE_READ;
	}

	m_strBaseFilePath = tum3d::GetPath(indexFileName);
	m_strFilename = indexFileName;
	m_strName = getFileName(m_strFilename, true, '\\');
	FILE* indexFile = nullptr;

	fopen_s(&indexFile, indexFileName.c_str(), "rb");
	if(!indexFile)
	{
		cout << "Error opening index file " << indexFileName << "\n";
		return false;
	}
	//E_ASSERT("Could not open file \"" << indexFileName << "\"", indexFile != nullptr);


	// Read headers
	ReadHeaders(indexFile);

	// Read settings
	ReadSettings(indexFile);

	// Read timesteps
	ReadTimesteps(indexFile);

	fclose(indexFile);

	// register all timestep files with the file pool
	for(uint timestep = 0; timestep < m_vTimesteps.size(); timestep++)
	{
		char currentFile[1024];
		sprintf_s(currentFile, TIMESTEP_FILE_MASK, timestep);
		string fileName = m_strBaseFilePath + m_strTimestepFileNamePrefix + currentFile;

		m_filePool.RegisterFile(fileName, timestep);
	}

	StartLoadingThread();

	m_bIsOpen = true;

	return true;
}



void TimeVolumeIO::Create(const string& indexFileName, const Vec3i& volumeSize, bool periodic, int32 brickSize, int32 overlap,
	int32 channels, const string& timestepFileNamePrefix)
{
	Close();

	cout << "Creating file " << indexFileName << "...\n";
	m_openMode = OPEN_MODE_WRITE;

	m_strFilename = indexFileName;
	m_strBaseFilePath = tum3d::GetPath(indexFileName) + "\\";
	FILE* indexFile;

	fopen_s(&indexFile, indexFileName.c_str(), "wb+");
	E_ASSERT("Could not open file \"" << indexFileName << "\"", indexFile != nullptr);

	// Close file again, writing will occur later on
	fclose(indexFile);

	m_info.m_volumeSize = volumeSize;
	m_info.m_periodic = periodic;
	m_info.m_iBrickSize = brickSize;
	m_info.m_iOverlap = overlap;
	m_info.m_iChannels = channels;
	E_ASSERT("Channel count must be <= " << ms_channelCountMax << " at the moment, but is " << m_info.m_iChannels, m_info.m_iChannels <= ms_channelCountMax);
	m_info.m_vQuantSteps.resize(m_info.m_iChannels);
	std::for_each(m_info.m_vQuantSteps.begin(), m_info.m_vQuantSteps.end(), [](float& val)
	{
		val = 1.0f / 256.0f;
	});

	m_vTimesteps.clear();
	m_iPadding = DEFAULT_PADDING;
	m_strTimestepFileNamePrefix = timestepFileNamePrefix;

	m_bIsOpen = true;
}



void TimeVolumeIO::ReadHeaders(FILE* indexFile)
{
	// Read magic string
	std::unique_ptr<char[]> ident(new char[strlen(IDENT)]);
	fread_s(ident.get(), strlen(IDENT), 1, strlen(IDENT), indexFile);

	E_ASSERT("File is not a valid TimeVolume file (magic string mismatch)", !memcmp(ident.get(), IDENT, strlen(IDENT)));

	cout << "\tMagic string valid\n";


	// Read version
	int32 version;
	fread_s(&version, sizeof(version), sizeof(int32), 1, indexFile);
	E_ASSERT("Version mismatch: found " << version << ", need " << CURRENT_VERSION, version == CURRENT_VERSION);
	cout << "\tVersion valid\n";
}



void TimeVolumeIO::Close()
{
	if (m_bIsOpen)
	{
		ShutdownLoadingThread();

		// note: this affects only reads, not writes, because writes are blocking
		DiscardAllIO();

		cout << "Closing file " << m_strFilename << "\n";

		if (m_hCurrentOutputFile != INVALID_HANDLE_VALUE)
		{
			CloseHandle(m_hCurrentOutputFile);
			m_hCurrentOutputFile = INVALID_HANDLE_VALUE;
		}

		m_filePool.UnregisterAllFiles();


		if (m_openMode == OPEN_MODE_WRITE)
		{
			cout << "\tWriting index file...\n";

			FILE* indexFile;
			fopen_s(&indexFile, m_strFilename.c_str(), "wb+");
			E_ASSERT("Could not open file \"" << m_strFilename << "\"", indexFile != nullptr);

			// Write data to index file
			WriteHeaders(indexFile);
			WriteSettings(indexFile);
			WriteTimesteps(indexFile);
			fclose(indexFile);
		}

		m_bIsOpen = false;

		// all loaded bricks are in the LRU cache
		while(!m_lruCache.empty())
		{
			Brick* pBrick = *m_lruCache.begin();
			UnloadBrickData(*pBrick);
		}
		m_lruCache.clear();

		m_vTimesteps.clear();
	}

	m_info.m_volumeSize.clear();
	m_info.m_iTimestepCount = 0;
	m_info.m_iBrickSize = 0;
	m_info.m_iOverlap = 0;
	m_info.m_compression = COMPRESSION_NONE;
	m_info.m_vQuantSteps.clear();
	m_info.m_iChannels = 0;
	m_vTimesteps.resize(0);
	m_iPadding = DEFAULT_PADDING;
	m_strTimestepFileNamePrefix = "";

	m_strFilename = "";
	m_strName = "";

	m_iCurrentOutputTimestep = -1;
	m_hCurrentOutputFile = INVALID_HANDLE_VALUE;
}



void TimeVolumeIO::ReadSettings(FILE* indexFile)
{
	static const int32 MAX_LEN = 4096;

	cout << "\tReading settings\n";
	assert(indexFile != nullptr);

	char line[MAX_LEN];
	char value[MAX_LEN];
	char name[MAX_LEN];

	fgets(line, MAX_LEN, indexFile);

	bool haveGridSpacing = false;
	bool haveTimeSpacing = false;
	bool havePeriodic = false;
	bool haveLessRLE = false;
	bool haveHuffmanBits = false;

	while(strlen(line) > 0)
	{
		E_ASSERT("Invalid settings line: " << line, sscanf_s(line, "%s\t=\t%[^\n]", name, MAX_LEN, value, MAX_LEN) == 2);

		if (!strcmp(name, "volumesize"))
		{
			E_ASSERT("Invalid setting \"volumesize\"", sscanf_s(value, "[%d, %d, %d]", &m_info.m_volumeSize[0], &m_info.m_volumeSize[1], &m_info.m_volumeSize[2]) == 3);
		}

		else if (!strcmp(name, "gridspacing"))
		{
			float val;
			E_ASSERT("Invalid setting \"gridspacing\"", sscanf_s(value, "%f", &val) == 1);
			m_info.m_fGridSpacing = Vec3f(val, val, val);
			haveGridSpacing = true;
		}
		else if (!strcmp(name, "gridspacing3"))
		{
			E_ASSERT("Invalid setting \"gridspacing\"", sscanf_s(value, "[%f, %f, %f]", &m_info.m_fGridSpacing[0], &m_info.m_fGridSpacing[1], &m_info.m_fGridSpacing[2]) == 3);
			haveGridSpacing = true;
		}

		else if (!strcmp(name, "timespacing"))
		{
			E_ASSERT("Invalid setting \"timespacing\"", sscanf_s(value, "%f", &m_info.m_fTimeSpacing) == 1);
			haveTimeSpacing = true;
		}

		else if (!strcmp(name, "periodic"))
		{
			char c[255];
			E_ASSERT("Invalid setting \"periodic\"", sscanf_s(value, "%[^\n]", c, 255) == 1);
			if(!strcmp(c, "true")) {
				m_info.m_periodic = true;
			} else if(!strcmp(c, "false")) {
				m_info.m_periodic = false;
			} else {
				THROW("Invalid periodic specification " << c);
			}
			havePeriodic = true;
		}

		else if (!strcmp(name, "bricksize"))
		{
			E_ASSERT("Invalid setting \"bricksize\"", sscanf_s(value, "%d", &m_info.m_iBrickSize) == 1);
		}

		else if (!strcmp(name, "overlap"))
		{
			E_ASSERT("Invalid setting \"overlap\"", sscanf_s(value, "%d", &m_info.m_iOverlap) == 1);
		}

		else if (!strcmp(name, "padding"))
		{
			E_ASSERT("Invalid setting \"padding\"", sscanf_s(value, "%d", &m_iPadding) == 1);
		}

		else if (!strcmp(name, "channels"))
		{
			E_ASSERT("Invalid setting \"channels\"", sscanf_s(value, "%d", &m_info.m_iChannels) == 1);
			m_info.m_vQuantSteps.resize(m_info.m_iChannels);
		}

		else if (!strcmp(name, "prefix"))
		{
			m_strTimestepFileNamePrefix = value;
		}

		else if (!strcmp(name, "compression"))
		{
			char c[255];
			E_ASSERT("Invalid setting \"compression\"", sscanf_s(value, "%[^\n]", c, 255) == 1);
			// legacy: "true" means COMPRESSION_FIXEDQUANT
			if(!strcmp(c, "true")) {
				m_info.m_compression = COMPRESSION_FIXEDQUANT;
			} else if(!strcmp(c, "false")) {
				m_info.m_compression = COMPRESSION_NONE;
			} else {
				m_info.m_compression = GetCompressionTypeFromName(c);
				E_ASSERT("Invalid compression specification " << c, m_info.m_compression != COMPRESSION_TYPE_COUNT);
			}
		}

		else if (!strcmp(name, "lessrle"))
		{
			char c[255];
			E_ASSERT("Invalid setting \"lessrle\"", sscanf_s(value, "%[^\n]", c, 255) == 1);
			if(!strcmp(c, "true")) {
				m_info.m_lessRLE = true;
			} else if(!strcmp(c, "false")) {
				m_info.m_lessRLE = false;
			} else {
				THROW("Invalid lessrle specification " << c);
			}
			haveLessRLE = true;
		}

		else if (!strcmp(name, "huffmanbits"))
		{
			E_ASSERT("Invalid setting \"huffmanbits\"", sscanf_s(value, "%d", &m_info.m_huffmanBitsMax) == 1);
			haveHuffmanBits = true;
		}

		// Legacy: Single quant step
		else if (!strcmp(name, "quantstep"))
		{
			float qs;
			E_ASSERT("Invalid setting \"quantstep\"", sscanf_s(value, "%f", &qs) == 1);
			for (auto it = m_info.m_vQuantSteps.begin(); it != m_info.m_vQuantSteps.end(); ++it)
			{
				*it = qs;
			}
		}

		else if (!memcmp(name, "quantstep_", strlen("quantstep_") - 1))
		{
			char c[255];
			for (int32 i = 0; i < m_info.m_iChannels; ++i)
			{
				sprintf_s(c, "quantstep_%d", i);
				if (!strcmp(name, c))
				{
					E_ASSERT("Invalid setting \"quantstep_" << i << "\"", sscanf_s(value, "%f", &m_info.m_vQuantSteps[i]) == 1);
					break;
				}
			}
		}

		else if (!strcmp(name, "qualityfactor"))
		{
			E_ASSERT("qualityfactor is not supported anymore", false);
		}

		else if (!strcmp(name, "timesteps"))
		{
			E_ASSERT("Invalid setting \"timesteps\"", sscanf_s(value, "%d", &m_info.m_iTimestepCount) == 1);
			m_vTimesteps.resize(m_info.m_iTimestepCount);
		}

		else
		{
			// unknown setting, warn
			string setting(name);
			setting = setting.substr(0, setting.find_first_of(" \t\r\n"));
			cout << "\t\tWarning: Unknown setting \"" << setting << "\" ignored\n";
		}

		char c = fgetc(indexFile);

		if (c != 0)
		{
			ungetc(c, indexFile);
			fgets(line, MAX_LEN, indexFile);
		}
		else
		{
			line[0] = 0;
		}
	}

	E_ASSERT("Invalid volume size", (m_info.m_volumeSize[0] > 0) && (m_info.m_volumeSize[1] > 0) && (m_info.m_volumeSize[2] > 0));
	E_ASSERT("Invalid brick size", m_info.m_iBrickSize > 0);

	// legacy: periodic defaults to true
	if(!havePeriodic)
	{
		m_info.m_periodic = true;
	}

	// legacy: if there was no grid/time spacing setting, default to values for isotropic turbulence data set
	if(!haveGridSpacing)
	{
		m_info.m_fGridSpacing = tum3D::Vec3f((2.0f * PI) / 1024.0f);
	}
	if(!haveTimeSpacing)
	{
		m_info.m_fTimeSpacing = 0.002f;
	}

	// legacy: default to false
	if(!haveLessRLE)
	{
		m_info.m_lessRLE = false;
	}
	// legacy: if huffmanbits isn't specified, set to 0 (which makes cudaCompress use the default)
	if(!haveHuffmanBits)
	{
		m_info.m_huffmanBitsMax = 0;
	}
}



void TimeVolumeIO::ReadTimesteps(FILE* indexFile)
{
	// NOTE This procedure performs pretty random file access, change this if
	// it really shows to be a problem

	FileTimestepHeader ts;
	int32 nextTimestepOffset = ftell(indexFile);

	for (auto itTimestep = m_vTimesteps.begin(); itTimestep != m_vTimesteps.end(); ++itTimestep)
	{
		// Read timestep header
		fseek(indexFile, nextTimestepOffset, SEEK_SET);
		assert(!feof(indexFile));
		fread_s(&ts, sizeof(ts), sizeof(ts), 1, indexFile);
		nextTimestepOffset = ftell(indexFile);

		itTimestep->index = ts.index;
		itTimestep->bricks.resize(GetBrickCount().volume());

		// Read timestep brick headers
		fseek(indexFile, ts.brickListOffset, SEEK_SET);

		for (int32 i = 0; i < ts.numBricks; i++)
		{
			BrickDesc bd;
			fread_s(&bd, sizeof(BrickDesc), sizeof(BrickDesc), 1, indexFile);

			int32 brickIndex = GetBrickLinearIndex(bd.spatialIndex);
			Brick& brick = itTimestep->bricks[brickIndex];

			brick.m_exists = true;

			brick.m_timestep = itTimestep->index;
			brick.m_spatialIndex = bd.spatialIndex;
			brick.m_size = bd.bricksize;

			brick.m_fileOffset = bd.dataOffset;

			brick.m_bytesize = bd.bytesize;
			brick.m_bytesizePadded = bd.paddedBytesize;
			//brick.m_ppChannelData = new void*[m_info.m_iChannels];
			//brick.m_pChannelSize = new int64[m_info.m_iChannels];

			memcpy(brick.m_pMinMeasuresInBrick, bd.m_pMinMeasuresInBrick, sizeof(float)*NUM_MEASURES);
			memcpy(brick.m_pMaxMeasuresInBrick, bd.m_pMaxMeasuresInBrick, sizeof(float)*NUM_MEASURES);

			brick.m_lruIt = m_lruCache.cend();

			fread_s(brick.m_pChannelSize, m_info.m_iChannels * sizeof(int64), sizeof(int64), m_info.m_iChannels, indexFile);
			for (int i = 0; i < m_info.m_iChannels; ++i)
			{
				brick.m_ppChannelData[i] = nullptr;
			}
		}
	}
}



void TimeVolumeIO::WriteHeaders(FILE* indexFile)
{
	fwrite(IDENT, 1, strlen(IDENT), indexFile);
	fwrite(&CURRENT_VERSION, sizeof(CURRENT_VERSION), 1, indexFile);
}



void TimeVolumeIO::WriteSettings(FILE* indexFile)
{
	fprintf(indexFile, "%s\t=\t[%d, %d, %d]\n", "volumesize", m_info.m_volumeSize[0], m_info.m_volumeSize[1], m_info.m_volumeSize[2]); 
	//fprintf(indexFile, "%s\t=\t%.32f\n", "gridspacing", m_info.m_fGridSpacing);
	fprintf(indexFile, "%s\t=\t[%.10f, %.10f, %.10f]\n", "gridspacing3", m_info.m_fGridSpacing[0], m_info.m_fGridSpacing[1], m_info.m_fGridSpacing[2]);
	fprintf(indexFile, "%s\t=\t%.32f\n", "timespacing", m_info.m_fTimeSpacing);
	fprintf(indexFile, "%s\t=\t%s\n", "periodic", m_info.m_periodic ? "true" : "false");
	fprintf(indexFile, "%s\t=\t%d\n", "bricksize", m_info.m_iBrickSize);
	fprintf(indexFile, "%s\t=\t%d\n", "overlap", m_info.m_iOverlap);
	fprintf(indexFile, "%s\t=\t%d\n", "padding", m_iPadding);
	fprintf(indexFile, "%s\t=\t%d\n", "timesteps", static_cast<int32>(m_vTimesteps.size()));
	fprintf(indexFile, "%s\t=\t%d\n", "channels", m_info.m_iChannels);
	if(!m_strTimestepFileNamePrefix.empty()) {
		fprintf(indexFile, "%s\t=\t%s\n", "prefix", m_strTimestepFileNamePrefix.c_str()); 
	}

	fprintf(indexFile, "%s\t=\t%s\n", "compression", GetCompressionTypeName(m_info.m_compression).c_str());
	if(m_info.m_compression == COMPRESSION_FIXEDQUANT || m_info.m_compression == COMPRESSION_FIXEDQUANT_QF)
	{
		for (int32 i = 0; i < m_info.m_iChannels; ++i)
		{
			fprintf(indexFile, "quantstep_%d\t=\t%.32f\n", i, m_info.m_vQuantSteps[i]);
		}
	}
	if(m_info.m_compression != COMPRESSION_NONE)
	{
		fprintf(indexFile, "%s\t=\t%s\n", "lessrle", m_info.m_lessRLE ? "true" : "false");
		fprintf(indexFile, "%s\t=\t%d\n", "huffmanbits", m_info.m_huffmanBitsMax);
	}

	// Write \0 to denote end of settings section
	fputc(0, indexFile);
}



void TimeVolumeIO::WriteTimesteps(FILE* indexFile)
{
	int32 nextTimestepOffset = ftell(indexFile);

	// First pass: Reserve space by writing dummy timesteps
	for (auto it = m_vTimesteps.cbegin(); it != m_vTimesteps.cend(); ++it)
	{
		FileTimestepHeader ts;
		fwrite(&ts, sizeof(ts), 1, indexFile);
	}

	// Second pass: Fill timesteps and write brick lists
	for (auto it = m_vTimesteps.cbegin(); it != m_vTimesteps.cend(); ++it)
	{
		FileTimestepHeader ts;
		fseek(indexFile, 0, SEEK_END);

		ts.brickListOffset = ftell(indexFile);
		ts.index = it->index;
		ts.numBricks = 0;

		// Write brick list
		for (auto brick = it->bricks.cbegin(); brick != it->bricks.cend(); ++brick)
		{
			if(!brick->m_exists) continue;

			++ts.numBricks;

			BrickDesc bd;
			bd.spatialIndex = brick->m_spatialIndex;
			bd.bricksize = brick->m_size;

			bd.dataOffset = brick->m_fileOffset;

			bd.bytesize = brick->m_bytesize;
			bd.paddedBytesize = brick->m_bytesizePadded;

			memcpy(bd.m_pMinMeasuresInBrick, brick->m_pMinMeasuresInBrick, sizeof(float)*NUM_MEASURES);
			memcpy(bd.m_pMaxMeasuresInBrick, brick->m_pMaxMeasuresInBrick, sizeof(float)*NUM_MEASURES);

			fwrite(&bd, sizeof(bd), 1, indexFile);
			fwrite(brick->m_pChannelSize, sizeof(int64), m_info.m_iChannels, indexFile);
		}

		// Write timestep header
		fseek(indexFile, nextTimestepOffset, SEEK_SET);
		fwrite(&ts, sizeof(ts), 1, indexFile);
		nextTimestepOffset = ftell(indexFile);
	}
}



bool TimeVolumeIO::StartLoadingThread()
{
	if(m_loadingThread.joinable()) return false;

	m_shutdownLoadingThread = false;
	m_loadingThread = std::thread([this] { LoadingThreadFunc(); });
	return true;
}

bool TimeVolumeIO::ShutdownLoadingThread()
{
	if(!m_loadingThread.joinable()) return false;

	m_shutdownLoadingThread = true;
	WakeupLoadingThread();
	m_loadingThread.join();
	return true;
}

void TimeVolumeIO::WakeupLoadingThread()
{
	SetEvent(m_wakeupEvents[ms_loadingSlotCount]);
}


bool TimeVolumeIO::EnqueueLoadBrickData(Brick& brick)
{
	if (brick.m_state != Brick::LOADING_STATE_UNLOADED)
	{
		return true;
	}

	std::lock_guard<std::mutex> lock(m_loadingQueueMutex);

	// is there a spot available in the loading queue?
	if(m_loadingQueue.size() >= m_iLoadingQueueLimit)
	{
		return false;
	}

	// Add brick to loading queue
	brick.m_state = Brick::LOADING_STATE_QUEUED;
	m_loadingQueue.push(&brick);

	WakeupLoadingThread();

	return true;
}



void TimeVolumeIO::UnloadBrickData(Brick& brick)
{
	E_ASSERT("Cannot unload a brick without data present", brick.m_pData != nullptr);

#ifdef DEBUG
	cout << "Unloading brick [" << brick.m_spatialIndex[0] << ", " << brick.m_spatialIndex[1] << ", " <<
		brick.m_spatialIndex[2] << "] from timestep " << brick.m_timestep << "\n";
#endif

	if(m_bPageLockBricks)
	{
		cudaError_t result = cudaHostUnregister(brick.m_pData);
		if(result != cudaSuccess)
		{
			printf("TimeVolumeIO::UnloadBrickData: cudaHostUnregister failed with error %u (%s)!\n", uint(result), cudaGetErrorString(result));
		}
	}
	VirtualFree(brick.m_pData, 0, MEM_RELEASE);
	brick.m_pData = nullptr;
	m_memUsage.MemoryDeallocated(brick.m_bytesizePadded);

	if(brick.m_state == Brick::LOADING_STATE_FINISHED)
	{
		std::lock_guard<std::recursive_mutex> lock(m_lruMutex);

		assert(brick.m_lruIt != m_lruCache.cend());

		m_lruCache.erase(brick.m_lruIt);
		brick.m_lruIt = m_lruCache.cend();
	}

	brick.m_state = Brick::LOADING_STATE_UNLOADED;
}


void TimeVolumeIO::LoadingThreadFunc()
{
	while(!m_shutdownLoadingThread)
	{
#ifndef DONT_USE_LOADING_THREAD
		// start IO timer if it's not running already - also count CPU overhead towards IO time!
		if(!m_diskBusyTimer.IsRunning()) {
			m_diskBusyTimer.Start();
		}

		UpdateAsyncIO();

		// update total accumulated time
		m_diskBusyTimer.Stop();
		m_diskBusyTimeMS += m_diskBusyTimer.GetElapsedTimeMS();

		// if anything is still loading, start timing again
		if(m_iBricksLoading > 0) {
			m_diskBusyTimer.Start();
		}
#endif
		// wait until an IO operation is finished, or we're woken up explicitly (happens when items are added to the loading queue, or on shutdown)
		WaitForMultipleObjects(ms_loadingSlotCount + 1, m_wakeupEvents, false, INFINITE);
	}
}


void TimeVolumeIO::UpdateAsyncIO(bool blockingRead)
{
	// Check for finished I/O
	for (int32 i = 0; i < ms_loadingSlotCount; ++i)
	{
		if (!m_loadingSlots[i].InUse()) continue;

		if (blockingRead || HasOverlappedIoCompleted(&m_loadingSlots[i].overlapped))
		{
			DWORD dwNumberOfBytesRead;
			if (GetOverlappedResult(m_loadingSlots[i].hFile, &m_loadingSlots[i].overlapped, &dwNumberOfBytesRead, blockingRead))
			{
#if defined(DEBUG) || defined(_DEBUG)
				cout << "Finished loading brick [" << m_loadingSlots[i].pBrick->m_spatialIndex[0] << ", " << 
					m_loadingSlots[i].pBrick->m_spatialIndex[1] << ", " <<
					m_loadingSlots[i].pBrick->m_spatialIndex[2] << "] from timestep " << 
					m_loadingSlots[i].pBrick->m_timestep << "\n";
#endif
				E_ASSERT("Error in async I/O: Too few bytes read", dwNumberOfBytesRead == m_loadingSlots[i].pBrick->m_bytesizePadded);

				Brick* pLoadedBrick = m_loadingSlots[i].pBrick;
				m_loadingSlots[i].pBrick = nullptr;

				pLoadedBrick->m_state = Brick::LOADING_STATE_FINISHED;
				TouchBrick(pLoadedBrick);

				m_filePool.ReleaseFileHandle(pLoadedBrick->m_timestep);
				m_loadingSlots[i].hFile = INVALID_HANDLE_VALUE;

				InterlockedAdd64((LONG64*)&m_dataLoadedBytesTotal, pLoadedBrick->GetPaddedDataSize());
				InterlockedDecrement(&m_iBricksLoading);
			}
			else
			{
				DWORD le = GetLastError();

				if (le != ERROR_IO_INCOMPLETE)
				{
					// Some error occured - usually, this should not happen very often, but our server begs to differ
					// For now, we just discard this brick and start over...
					cout << "\n*** Error in GetOverlappedResult: " << le << "\n\n";

					UnloadBrickData(*m_loadingSlots[i].pBrick);

					m_filePool.ReleaseFileHandle(m_loadingSlots[i].pBrick->m_timestep);
					m_loadingSlots[i].hFile = INVALID_HANDLE_VALUE;

					m_loadingSlots[i].pBrick = nullptr;

					// Flush queue to avoid prefetching when not all bricks of the current timestep have been loaded
					DiscardQueuedIO();
				}
			}
		}
	}

	
	// Start new I/O
	for (int32 i = 0; i < ms_loadingSlotCount; ++i)
	{
		if (m_loadingSlots[i].InUse()) continue;

		Brick* brick = nullptr;
		// get brick from loading queue
		{
			std::lock_guard<std::mutex> lock(m_loadingQueueMutex);
			if (!m_loadingQueue.empty())
			{
				brick = m_loadingQueue.front();
				m_loadingQueue.pop();
				InterlockedIncrement(&m_iBricksLoading);
			}
		}
		// if we couldn't get a brick, bail out
		if (brick == nullptr) break;


		// get file handle
		m_loadingSlots[i].hFile = m_filePool.GetFileHandle(brick->m_timestep);

		// Prepare loading
		m_loadingSlots[i].pBrick = brick;
		m_loadingSlots[i].overlapped.Offset = m_loadingSlots[i].pBrick->m_fileOffset & 0xFFFFFFFF;
		m_loadingSlots[i].overlapped.OffsetHigh = m_loadingSlots[i].pBrick->m_fileOffset >> 32;

		// Reserve brick data
		assert(m_loadingSlots[i].pBrick->m_pData == nullptr);
		m_loadingSlots[i].pBrick->m_state = Brick::LOADING_STATE_LOADING;
		ReserveBrickData(m_loadingSlots[i].pBrick);

		int64 offset = 0;
		for (int32 channel = 0; channel < m_info.m_iChannels; ++channel)
		{
			m_loadingSlots[i].pBrick->m_ppChannelData[channel] = static_cast<int8*>(m_loadingSlots[i].pBrick->m_pData) + offset;
			offset += m_loadingSlots[i].pBrick->m_pChannelSize[channel];
		}

		// Start loading
		DWORD dwNumberOfBytesRead;
		ReadFile(m_loadingSlots[i].hFile, m_loadingSlots[i].pBrick->m_pData, (DWORD)m_loadingSlots[i].pBrick->m_bytesizePadded,
			&dwNumberOfBytesRead, &m_loadingSlots[i].overlapped);

#if defined(DEBUG) || defined(_DEBUG)
		cout << "Starting loading of brick [" << m_loadingSlots[i].pBrick->m_spatialIndex[0] << ", " << 
				m_loadingSlots[i].pBrick->m_spatialIndex[1] << ", " <<
				m_loadingSlots[i].pBrick->m_spatialIndex[2] << "] from timestep " << 
				m_loadingSlots[i].pBrick->m_timestep << "\n";
#endif
	}
}



void TimeVolumeIO::DiscardQueuedIO()
{
	std::lock_guard<std::mutex> lock(m_loadingQueueMutex);

	while (!m_loadingQueue.empty())
	{
		Brick* brick = m_loadingQueue.front();
		m_loadingQueue.pop();
		assert(brick->m_state == Brick::LOADING_STATE_QUEUED);
		brick->m_state = Brick::LOADING_STATE_UNLOADED;
	}
}


void TimeVolumeIO::DiscardAllIO()
{
	DiscardQueuedIO();

	// HACK: kill the loading thread, re-start afterwards
	bool threadWasRunning = ShutdownLoadingThread();

	for (int32 i = 0; i < ms_loadingSlotCount; ++i)
	{
		if(m_loadingSlots[i].InUse())
		{
			BOOL result = CancelIoEx(m_loadingSlots[i].hFile, &m_loadingSlots[i].overlapped);
			if(result == 0) { 
				DWORD error = GetLastError();
				assert(error == ERROR_NOT_FOUND);
			}
			m_filePool.ReleaseFileHandle(m_loadingSlots[i].pBrick->m_timestep);
			m_loadingSlots[i].hFile = INVALID_HANDLE_VALUE;

			UnloadBrickData(*m_loadingSlots[i].pBrick);

#ifdef DEBUG
			cout << "Canceled loading brick [" << m_loadingSlots[i].pBrick->m_spatialIndex[0] << ", " << 
				m_loadingSlots[i].pBrick->m_spatialIndex[1] << ", " <<
				m_loadingSlots[i].pBrick->m_spatialIndex[2] << "] from timestep " << 
				m_loadingSlots[i].pBrick->m_timestep << "\n";
#endif

			m_loadingSlots[i].pBrick = nullptr;

			InterlockedDecrement(&m_iBricksLoading);
		}
	}

	if(threadWasRunning)
	{
		StartLoadingThread();
	}
}


void TimeVolumeIO::UnloadLRUBricks()
{
	// Page out lru bricks while not enough memory is available
	while (!m_memUsage.IsMemoryAvailable())
	{
		std::lock_guard<std::recursive_mutex> lock(m_lruMutex);
		if(m_lruCache.empty())
		{
			printf("WARNING TimeVolumeIO::UpdateLRUCache: No memory available, but LRU is empty?!?\n");
			break;
		}

		Brick* lruBrick = *m_lruCache.begin();
		E_ASSERT("Brick in m_lruCache is not loaded!", lruBrick->IsLoaded());

		UnloadBrickData(*lruBrick);
	}

#ifdef DONT_USE_LOADING_THREAD
	UpdateAsyncIO();
#endif
}


bool TimeVolumeIO::AllIOIsFinished()
{
	std::lock_guard<std::mutex> lock(m_loadingQueueMutex);

	return m_loadingQueue.empty() && m_iBricksLoading == 0;
}


void TimeVolumeIO::WaitForAllIO(bool discardQueued)
{
	if (discardQueued)
	{
		DiscardQueuedIO();
	}

	while (!AllIOIsFinished())
	{
#ifdef DONT_USE_LOADING_THREAD
		UpdateAsyncIO(true);
#else
		// just wait for the loading thread to do its thing
		Sleep(10); // yeah, i did that
#endif
	}
}


void TimeVolumeIO::ReserveBrickData(Brick* brick)
{
	brick->m_pData = VirtualAlloc(nullptr, brick->m_bytesizePadded, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
	E_ASSERT("VirtualAlloc failed with error code " << GetLastError(), brick->m_pData != nullptr);
	m_memUsage.MemoryAllocated(brick->m_bytesizePadded);
	if(m_bPageLockBricks)
	{
		cudaError_t result = cudaHostRegister(brick->m_pData, brick->m_bytesizePadded, cudaHostRegisterPortable);
		if(result != cudaSuccess)
		{
			printf("TimeVolumeIO::ReserveBrickData: cudaHostRegister failed with error %u (%s)!\n", uint(result), cudaGetErrorString(result));
		}
	}

	TouchBrick(brick);
}


void TimeVolumeIO::TouchBrick(Brick* brick)
{
	// move brick to the back of the LRU cache
	std::lock_guard<std::recursive_mutex> lock(m_lruMutex);

	if (brick->m_lruIt != m_lruCache.cend())
	{
		m_lruCache.erase(brick->m_lruIt);
	}

	if(brick->IsLoaded())
	{
		m_lruCache.push_back(brick);
		brick->m_lruIt = m_lruCache.cend();
		--brick->m_lruIt;
	}
}


void TimeVolumeIO::SetPageLockBrickData(bool lock)
{
	if(lock == m_bPageLockBricks) return;

	m_bPageLockBricks = lock;

	//TODO keep track of all allocated bricks, so we don't have to walk over all timesteps?
	for(int t = 0; t < GetTimestepCount(); t++)
	{
		Timestep& timestep = GetTimestep(t);
		for(auto brick = timestep.bricks.begin(); brick != timestep.bricks.end(); brick++)
		{
			if(brick->m_pData != nullptr)
			{
				if(m_bPageLockBricks)
				{
					cudaError_t result = cudaHostRegister(brick->m_pData, brick->m_bytesizePadded, cudaHostRegisterPortable);
					if(result != cudaSuccess)
					{
						printf("TimeVolumeIO::ReserveBrickData: cudaHostRegister failed with error %u (%s)!\n", uint(result), cudaGetErrorString(result));
					}
				}
				else
				{
					cudaError_t result = cudaHostUnregister(brick->m_pData);
					if(result != cudaSuccess)
					{
						printf("TimeVolumeIO::UnloadBrickData: cudaHostUnregister failed with error %u (%s)!\n", uint(result), cudaGetErrorString(result));
					}
				}
			}
		}
	}
}
