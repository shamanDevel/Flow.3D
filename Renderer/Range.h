#ifndef __TUM3D__RANGE_H__
#define __TUM3D__RANGE_H__


#include "global.h"


// [min,max) from [0,1)
class Range1D
{
public:
	Range1D() : m_min(0.0f), m_max(1.0f) {}
	Range1D(float min, float max) : m_min(min), m_max(max) {}

	void Set(float min, float max) { m_min = min; m_max = max; }

	float GetMin() const { return m_min; }
	float GetMax() const { return m_max; }
	float GetLength() const { return m_max - m_min; }

	// map to inclusive int range [0,intmax]
	int GetMinInt(int intmax) const;
	int GetMaxInt(int intmax) const;

	bool operator==(const Range1D& rhs) const;
	bool operator!=(const Range1D& rhs) const;

private:
	float m_min;
	float m_max;
};


#endif
