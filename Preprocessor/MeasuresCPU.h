#ifndef __TUM3D__MEASURESCPU_H__
#define __TUM3D__MEASURESCPU_H__

#include <vector>
#include <cassert>
#include "../Renderer/Measure.h"

/******************************************************************************
** Linear algebra classes
******************************************************************************/

struct vec4 {
	union {
		float v[4];
		struct {
			float x, y, z, w;
		};
	};

	friend vec4 operator+(vec4 lhs, const vec4& rhs) {
		for (int i = 0; i < 4; i++) {
			lhs.v[i] += rhs.v[i];
		}
		return lhs;
	}

	friend vec4 operator-(vec4 lhs, const vec4& rhs) {
		for (int i = 0; i < 4; i++) {
			lhs.v[i] -= rhs.v[i];
		}
		return lhs;
	}

	friend vec4 operator*(vec4 lhs, const vec4& rhs) {
		for (int i = 0; i < 4; i++) {
			lhs.v[i] *= rhs.v[i];
		}
		return lhs;
	}

	friend vec4 operator/(vec4 lhs, const vec4& rhs) {
		for (int i = 0; i < 4; i++) {
			lhs.v[i] /= rhs.v[i];
		}
		return lhs;
	}

	friend vec4 operator*(vec4 lhs, float a) {
		for (int i = 0; i < 4; i++) {
			lhs.v[i] *= a;
		}
		return lhs;
	}

	friend vec4 operator/(vec4 lhs, float a) {
		for (int i = 0; i < 4; i++) {
			lhs.v[i] /= a;
		}
		return lhs;
	}

	friend vec4 operator*(float a, vec4 lhs) {
		for (int i = 0; i < 4; i++) {
			lhs.v[i] *= a;
		}
		return lhs;
	}

	friend vec4 operator/(float a, vec4 lhs) {
		for (int i = 0; i < 4; i++) {
			lhs.v[i] /= a;
		}
		return lhs;
	}
};

struct vec3 {
	union {
		float v[4];
		struct {
			float x, y, z;
			float padding;
		};
	};

	friend vec3 operator+(vec3 lhs, const vec3& rhs) {
		for (int i = 0; i < 3; i++) {
			lhs.v[i] += rhs.v[i];
		}
		return lhs;
	}

	friend vec3 operator-(vec3 lhs, const vec3& rhs) {
		for (int i = 0; i < 3; i++) {
			lhs.v[i] -= rhs.v[i];
		}
		return lhs;
	}

	friend vec3 operator*(vec3 lhs, const vec3& rhs) {
		for (int i = 0; i < 3; i++) {
			lhs.v[i] *= rhs.v[i];
		}
		return lhs;
	}

	friend vec3 operator/(vec3 lhs, const vec3& rhs) {
		for (int i = 0; i < 3; i++) {
			lhs.v[i] /= rhs.v[i];
		}
		return lhs;
	}

	friend vec3 operator*(vec3 lhs, float a) {
		for (int i = 0; i < 3; i++) {
			lhs.v[i] *= a;
		}
		return lhs;
	}

	friend vec3 operator/(vec3 lhs, float a) {
		for (int i = 0; i < 3; i++) {
			lhs.v[i] /= a;
		}
		return lhs;
	}

	friend vec3 operator*(float a, vec3 lhs) {
		for (int i = 0; i < 3; i++) {
			lhs.v[i] *= a;
		}
		return lhs;
	}

	friend vec3 operator/(float a, vec3 lhs) {
		for (int i = 0; i < 3; i++) {
			lhs.v[i] /= a;
		}
		return lhs;
	}
};

struct mat3x3 {
	vec3 m[3];

	friend mat3x3 operator+(mat3x3 lhs, const mat3x3& rhs) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				lhs.m[i].v[j] += rhs.m[i].v[j];
			}
		}
		return lhs;
	}
	friend mat3x3 operator-(mat3x3 lhs, const mat3x3& rhs) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				lhs.m[i].v[j] -= rhs.m[i].v[j];
			}
		}
		return lhs;
	}
	friend mat3x3 operator*(mat3x3 lhs, const mat3x3& rhs) {
		mat3x3 result;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				result.m[i].v[j] = 0;
				for (int k = 0; k < 3; k++) {
					result.m[i].v[j] += lhs.m[i].v[k] * rhs.m[k].v[j];
				}
			}
		}
		return result;
	}
	friend vec3 operator*(mat3x3 lhs, const vec3& rhs) {
		vec3 result;
		for (int i = 0; i < 3; i++) {
			result.v[i] = 0;
			for (int j = 0; j < 3; j++) {
				result.v[i] += lhs.m[i].v[j] * rhs.v[j];
				for (int k = 0; k < 3; k++) {
				}
			}
		}
		return result;
	}
};

/******************************************************************************
** Class for encapsulating a volume texture on the CPU
******************************************************************************/

class VolumeTextureCPU {
public:
	VolumeTextureCPU(const std::vector<std::vector<float>>& channelData,
			size_t sizeX, size_t sizeY, size_t sizeZ, size_t brickOverlap)
		: channelData(channelData), sizeX(sizeX), sizeY(sizeY), sizeZ(sizeZ), brickOverlap(brickOverlap) {}

	inline const std::vector<std::vector<float>>& getChannelData() { return channelData; }
	inline size_t getSizeX() { return sizeX; }
	inline size_t getSizeY() { return sizeY; }
	inline size_t getSizeZ() { return sizeZ; }
	inline size_t getBrickOverlap() { return brickOverlap; }

	vec4 samplevec4(int x, int y, int z) const {
		assert(channelData.size() >= 4);
		x = clamp(x, 0, (int)sizeX - 1);
		y = clamp(y, 0, (int)sizeY - 1);
		z = clamp(z, 0, (int)sizeZ - 1);

		vec4 vec;
		for (size_t i = 0; i < 4; i++) {
			size_t idx = x + y * sizeX + z * sizeX * sizeY;
			vec.v[i] = channelData.at(i).at(idx);
		}
		return vec;
	}

	vec3 samplevec3(int x, int y, int z) const {
		assert(channelData.size() >= 3);
		x = clamp(x, 0, (int)sizeX - 1);
		y = clamp(y, 0, (int)sizeY - 1);
		z = clamp(z, 0, (int)sizeZ - 1);

		vec3 vec;
		for (size_t i = 0; i < 3; i++) {
			size_t idx = x + y * sizeX + z * sizeX * sizeY;
			vec.v[i] = channelData.at(i).at(idx);
		}
		return vec;
	}

private:
	size_t sizeX, sizeY, sizeZ;
	size_t brickOverlap;
	const std::vector<std::vector<float>>& channelData;

	template<typename T>
	T clamp(T val, T min, T max) const
	{
		return val < min ? min : (val > max ? max : val);
	}
};


/******************************************************************************
** Function for computing measures on the CPU
******************************************************************************/

float getMeasureFromVolume(
        const VolumeTextureCPU& tex, int x, int y, int z, const vec3& h,
        eMeasureSource measureSource, eMeasure measure);

#endif
