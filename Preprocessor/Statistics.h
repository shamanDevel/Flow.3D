#ifndef __TUM3D__STATISTICS_H__
#define __TUM3D__STATISTICS_H__


#include <algorithm>
#include <cmath>


class Statistics
{
public:
	struct Stats {
		size_t ElemCount;
		size_t OriginalSize;
		size_t CompressedSize;
		double BitsPerElem;
		double CompressionRate;

		double Min;
		double Max;
		double Range;
		double Average;
		double Variance;

		double ReconstMin;
		double ReconstMax;
		double ReconstRange;
		double ReconstAverage;
		double ReconstVariance;

		double AvgAbsError;
		double MaxAbsError;
		double RMSError;
		double SNR;
		double PSNR;
		double AvgRelError;
		double MaxRelError;
		size_t RelErrorCount;
	};


	Statistics();


	// Add a block of data to the running total
	// pDataReconst may be null, but should consistently be null in either all or none of the AddData calls
	template<typename T>
	void AddData(const T* pDataOrig, const T* pDataReconst, size_t count, size_t compressedSize, const T& relErrorEpsilon = T(0), size_t stride = 1);

	// Clear running totals
	void Clear();

	// Query statistics on running totals
	Stats GetStats() const;

	size_t GetElemCount() const;
	size_t GetOriginalSize() const;
	size_t GetCompressedSize() const;
	double GetBitsPerElem() const;
	double GetCompressionRate() const;

	double GetMin() const;
	double GetMax() const;
	double GetRange() const;
	double GetAverage() const;
	double GetVariance() const;

	double GetReconstMin() const;
	double GetReconstMax() const;
	double GetReconstRange() const;
	double GetReconstAverage() const;
	double GetReconstVariance() const;

	double GetAvgAbsError() const;
	double GetMaxAbsError() const;
	double GetRMSError() const;
	double GetSNR() const;
	double GetPSNR() const;
	double GetAvgRelError() const;
	double GetMaxRelError() const;
	size_t GetRelErrorCount() const;

	// Add other instance to our running totals
	Statistics& operator+=(const Statistics& rhs);

private:
	size_t m_elemCount;
	size_t m_sizeOriginal;
	size_t m_sizeCompressed;

	double m_valueMin;
	double m_valueMax;
	double m_valueSum;
	double m_valueSquareSum;

	double m_reconstMin;
	double m_reconstMax;
	double m_reconstSum;
	double m_reconstSquareSum;

	double m_errorAbsSum;
	double m_errorAbsMax;
	double m_errorSquareSum;
	double m_errorRelSum;
	double m_errorRelMax;
	size_t m_errorRelCount;
};


template<typename T>
void Statistics::AddData(const T* pDataOrig, const T* pDataReconst, size_t count, size_t compressedSize, const T& relErrorEpsilon, size_t stride)
{
	m_elemCount += count;
	m_sizeOriginal += count * sizeof(T);
	m_sizeCompressed += compressedSize;

	for(unsigned int i = 0; i < count; i++) {
		const T& valueT = pDataOrig[i * stride];
		double value = double(valueT);

		m_valueMin = std::min(m_valueMin, value);
		m_valueMax = std::max(m_valueMax, value);
		m_valueSum += value;
		m_valueSquareSum += value * value;

		if(pDataReconst) {
			double valueReconst = double(pDataReconst[i * stride]);

			m_reconstMin = std::min(m_reconstMin, valueReconst);
			m_reconstMax = std::max(m_reconstMax, valueReconst);
			m_reconstSum += valueReconst;
			m_reconstSquareSum += valueReconst * valueReconst;

			double errorAbs = std::abs(valueReconst - value);
			m_errorAbsSum += errorAbs;
			m_errorAbsMax = std::max(m_errorAbsMax, errorAbs);
			m_errorSquareSum += errorAbs * errorAbs;
			if(std::abs(valueT) > relErrorEpsilon) {
				double errorRel = errorAbs / std::abs(value);
				m_errorRelSum += errorRel;
				m_errorRelMax = std::max(m_errorRelMax, errorRel);
				m_errorRelCount++;
			}
		}
	}
}


#endif
