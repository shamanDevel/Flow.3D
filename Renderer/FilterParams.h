#ifndef __TUM3D__FILTERPARAMS_H__
#define __TUM3D__FILTERPARAMS_H__


#include <global.h>

#include <vector>

#include <ConfigFile.h>
#include <Vec.h>


struct FilterParams
{
	FilterParams();

	void Reset();

	void ApplyConfig(const ConfigFile& config);
	void WriteConfig(ConfigFile& config) const;

	std::vector<uint> m_radius;

	bool HasNonZeroRadius() const;

	bool operator==(const FilterParams& rhs) const;
	bool operator!=(const FilterParams& rhs) const;
};


#endif
