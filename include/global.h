#ifndef __TUM3D__GLOBAL_H__
#define __TUM3D__GLOBAL_H__


typedef unsigned char      byte;
typedef unsigned char      uchar;
typedef unsigned short     ushort;
typedef unsigned int       uint;

#include <cstdint>

typedef int8_t		int8;
typedef uint8_t		uint8;
typedef int16_t		int16;
typedef uint16_t	uint16;
typedef int32_t		int32;
typedef uint32_t	uint32;
typedef	int64_t		int64;
typedef uint64_t	uint64;


#ifndef __CUDACC__
#include <algorithm>

template<typename T>
T clamp(T val, T min, T max)
{
	uint32_t;

	return val < min ? min : (val > max ? max : val);
}

using std::min;
using std::max;
#endif


const float PI = 3.141592654f;


#ifndef SAFE_DELETE
#define SAFE_DELETE(x) delete x; x = nullptr;
#endif

#ifndef SAFE_DELETE_ARRAY
#define SAFE_DELETE_ARRAY(x) delete[] x; x = nullptr;
#endif


#endif
