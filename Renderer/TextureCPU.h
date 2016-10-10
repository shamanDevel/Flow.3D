#ifndef __TUM3D__TEXTURE_CPU_H__
#define __TUM3D__TEXTURE_CPU_H__


#include <vector>

#include <vector_types.h>


template<typename TexType>
struct TextureCPU
{
	uint3 size;
	std::vector<TexType> data;
};


#endif
