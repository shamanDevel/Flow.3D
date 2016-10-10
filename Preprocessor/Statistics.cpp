#include "Statistics.h"


Statistics::Statistics()
{
	Clear();
}

void Statistics::Clear()
{
	m_elemCount = 0;
	m_sizeOriginal = 0;
	m_sizeCompressed = 0;

	m_valueMin = std::numeric_limits<double>::max();
	m_valueMax = -std::numeric_limits<double>::max();
	m_valueSum = 0.0;
	m_valueSquareSum = 0.0;

	m_reconstMin = std::numeric_limits<double>::max();
	m_reconstMax = -std::numeric_limits<double>::max();
	m_reconstSum = 0.0;
	m_reconstSquareSum = 0.0;

	m_errorAbsSum = 0.0;
	m_errorAbsMax = 0.0;
	m_errorSquareSum = 0.0;
	m_errorRelSum = 0.0;
	m_errorRelMax = 0.0;
	m_errorRelCount = 0;
}


Statistics::Stats Statistics::GetStats() const
{
	Stats stats;

	stats.ElemCount = GetElemCount();
	stats.OriginalSize = GetOriginalSize();
	stats.CompressedSize = GetCompressedSize();
	stats.BitsPerElem = GetBitsPerElem();
	stats.CompressionRate = GetCompressionRate();

	stats.Min = GetMin();
	stats.Max = GetMax();
	stats.Range = GetRange();
	stats.Average = GetAverage();
	stats.Variance = GetVariance();

	stats.ReconstMin = GetReconstMin();
	stats.ReconstMax = GetReconstMax();
	stats.ReconstRange = GetReconstRange();
	stats.ReconstAverage = GetReconstAverage();
	stats.ReconstVariance = GetReconstVariance();

	stats.AvgAbsError = GetAvgAbsError();
	stats.MaxAbsError = GetMaxAbsError();
	stats.RMSError = GetRMSError();
	stats.SNR = GetSNR();
	stats.PSNR = GetPSNR();
	stats.AvgRelError = GetAvgRelError();
	stats.MaxRelError = GetMaxRelError();
	stats.RelErrorCount = GetRelErrorCount();

	return stats;
}


size_t Statistics::GetElemCount() const
{
	return m_elemCount;
}

size_t Statistics::GetOriginalSize() const
{
	return m_sizeOriginal;
}

size_t Statistics::GetCompressedSize() const
{
	return m_sizeCompressed;
}

double Statistics::GetBitsPerElem() const
{
	return double(m_sizeCompressed) * 8.0 / double(m_elemCount);
}

double Statistics::GetCompressionRate() const
{
	return double(m_sizeOriginal) / double(m_sizeCompressed);
}


double Statistics::GetMin() const
{
	return m_valueMin;
}

double Statistics::GetMax() const
{
	return m_valueMax;
}

double Statistics::GetRange() const
{
	return m_valueMax - m_valueMin;
}

double Statistics::GetAverage() const
{
	return m_valueSum / double(m_elemCount);
}

double Statistics::GetVariance() const
{
	double avg = m_valueSum / double(m_elemCount);
	double avgSq = m_valueSquareSum / double(m_elemCount);

	return avgSq - avg * avg;
}


double Statistics::GetReconstMin() const
{
	return m_reconstMin;
}

double Statistics::GetReconstMax() const
{
	return m_reconstMax;
}

double Statistics::GetReconstRange() const
{
	return m_reconstMax - m_reconstMin;
}

double Statistics::GetReconstAverage() const
{
	return m_reconstSum / double(m_elemCount);
}

double Statistics::GetReconstVariance() const
{
	double avg = m_reconstSum / double(m_elemCount);
	double avgSq = m_reconstSquareSum / double(m_elemCount);

	return avgSq - avg * avg;
}


double Statistics::GetAvgAbsError() const
{
	return m_errorAbsSum / double(m_elemCount);
}

double Statistics::GetMaxAbsError() const
{
	return m_errorAbsMax;
}

double Statistics::GetRMSError() const
{
	double avgSqError = m_errorSquareSum / double(m_elemCount);
	return std::sqrt(avgSqError);
}

double Statistics::GetSNR() const
{
	return 20.0 * std::log10(sqrt(GetVariance()) / GetRMSError());
}

double Statistics::GetPSNR() const
{
	return 20.0 * std::log10(sqrt(GetRange()) / GetRMSError());
}

double Statistics::GetAvgRelError() const
{
	return m_errorRelSum / double(m_errorRelCount);
}

double Statistics::GetMaxRelError() const
{
	return m_errorRelMax;
}

size_t Statistics::GetRelErrorCount() const
{
	return m_errorRelCount;
}


Statistics& Statistics::operator+=(const Statistics& rhs)
{
	m_elemCount += rhs.m_elemCount;
	m_sizeOriginal += rhs.m_sizeOriginal;
	m_sizeCompressed += rhs.m_sizeCompressed;

	m_valueMin = std::min(m_valueMin, rhs.m_valueMin);
	m_valueMax = std::max(m_valueMax, rhs.m_valueMax);
	m_valueSum += rhs.m_valueSum;
	m_valueSquareSum += rhs.m_valueSquareSum;

	m_reconstMin = std::min(m_reconstMin, rhs.m_reconstMin);
	m_reconstMax = std::max(m_reconstMax, rhs.m_reconstMax);
	m_reconstSum += rhs.m_reconstSum;
	m_reconstSquareSum += rhs.m_reconstSquareSum;

	m_errorAbsSum += rhs.m_errorAbsSum;
	m_errorAbsMax = std::max(m_errorAbsMax, rhs.m_errorAbsMax);
	m_errorSquareSum += rhs.m_errorSquareSum;
	m_errorRelSum += rhs.m_errorRelSum;
	m_errorRelMax = std::max(m_errorRelMax, rhs.m_errorRelMax);
	m_errorRelCount += rhs.m_errorRelCount;

	return *this;
}
