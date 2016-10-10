#include "Range.h"

#include <cmath>


int Range1D::GetMinInt(int intmax) const
{
	return int(floor(m_min * float(intmax + 1)));
}

int Range1D::GetMaxInt(int intmax) const
{
	return int(floor(m_max * float(intmax + 1))) - 1;
}

bool Range1D::operator==(const Range1D& rhs) const
{
	return m_min == rhs.m_min && m_max == rhs.m_max;
}

bool Range1D::operator!=(const Range1D& rhs) const
{
	return !(*this == rhs);
}
