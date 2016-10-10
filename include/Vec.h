#ifndef __VEC_H__
#define __VEC_H__

#define TUM3D_VEC_STRICT
#define TUM3D_MAT_STRICT


#include <cmath>
#include <iostream>
#include <cassert>
#include <algorithm>



/********************************************************************

TUM3D Vector/Matrix/Utils Header

copyright 2008- 2009

Joachim Georgii, georgii@tum.de
Roland Fraedrich, fraedrich@tum.de
Christian Dick, dick@tum.de

VERSION      1.01
DATE         29.05.2009
LAST_CHANGE  JG


CHANGES   ***********************************************************


VERSION 1.01:

- removed unnecessary const (JG)
- removed unnecessay normalize(const Vec) (JG)
- removed crucial min/max defines



NOTES   *************************************************************

Do not use *this as argument in any of the functions unless you
exactly know what to do ...


SUMMARY *************************************************************

Vec<N,T> members

	// General Constructors
	Vec<N,T>()
	Vec<N,T>(const Vec<N,T> &v)
	Vec<N,T>(const T *v)
	Vec<N,T>(const T &w)

	// Special Constructors from specialized Subclasses
	Vec2<T>(const T &t0, const T &t1)
	Vec3<T>(const T &t0, const T &t1, const T &t2)
	Vec4<T>(const T &t0, const T &t1, const T &t2, const T &t3)
	
	// Array subscripting and dereferencing operator
	T &operator[](int k)
	operator T *()

	// Assignment operator and arithmetic assignment operators
	Vec<N,T> &operator=(const Vec<N,T> &v)
	Vec<N,T> &operator+=(const Vec<N,T> &v)
	Vec<N,T> &operator-=(const Vec<N,T> &v)
	Vec<N,T> &operator*=(const T &w)
	Vec<N,T> &operator*=(const Vec<N,T> &v)
	Vec<N,T> &operator/=(const T &w)
	Vec<N,T> &operator/=(const Vec<N,T> &v)
	
	// Arithmetic operators
	// PLEASE NOTE:
	// - arithmetic operators include copying of a temporary vector
	// - for a better performance use the arithmetic assignment operators
	Vec<N,T> operator+(const Vec<N,T> &v) const
	Vec<N,T> operator-(const Vec<N,T> &v) const
	Vec<N,T> operator*(const T &w) const
	Vec<N,T> operator*(const Vec<N,T> &v) const
	Vec<N,T> operator/(const T &w) const
	Vec<N,T> operator/(const Vec<N,T> &v) const

	// Vector functions
	void clear()
	T norm() const
	T normSqr() const
	float normf() const
	T dot(const Vec<N,T> &v) const
	bool operator==(const Vec<N,T> &v) const
	T maximum() const
	T minimum() const

----------------------------------------------------

Mat<M,N,T> members

	// Constructors
	Mat<M,N,T>()
	Mat<M,N,T>(const Mat<M,N,T> &m)
	Mat<M,N,T>(const T *m)
	Mat<M,N,T>(const T &w)

	// Array subscripting, matrix access and dereferencing operators
	T &operator[](int k)
	T &get(int k, int l)
	Vec<N,T> getRow(int k)
	Vec<M,T> getCol(int l)
	operator T *()

	// Assignment operator and arithmetic assignment operators
	Mat<M,N,T> &operator=(const Mat<M,N,T> &m) 
	Mat<M,N,T> &operator+=(const Mat<M,N,T> &m)
	Mat<M,N,T> &operator-=(const Mat<M,N,T> &m)
	Mat<M,N,T> &operator*=(const T &w)
	Mat<M,N,T> &operator*=(const Mat<M,N,T> &m)
	Mat<M,N,T> &operator/=(const T &w)
	Mat<M,N,T> &operator/=(const Mat<M,N,T> &m)

	// additional matrix functions
	void clear()
	Mat<M,L,T> &multMat(const Mat<N,L,T> &B, Mat<M,L,T> &erg) const
	Mat<M,L,T> &multMatT(const Mat<L,N,T> &B, Mat<M,L,T> &erg) const
	Vec<M,T> &multVec(const Vec<N,T> &b, Vec<M,T> &erg) const
	Vec<N,T> &multTVec(const Vec<M,T> &b, Vec<N,T> &erg) const
	Mat<N,M,T> &transpose(Mat<N,M,T> &erg) const

----------------------------------------------------

tum3D static members
	
	Vec<3,T> &crossprod(const Vec<3,T> &a, const Vec<3,T> &b, Vec<3,T> &erg)

	float distance(const Vec<N,float> &v0, const Vec<N,float> &v1)
	double distance(const Vec<N,double> &v0, const Vec<N,double> &v1)
	Vec<N,double> &normalize(Vec<N,double> &v)
	Vec<N,float> &normalize(Vec<N,float> &v)

	Mat<2,2,T> &mat2x2(const T &v00, const T &v01, const T &v10, const T &v11, Mat<2,2,T> &m)
	Mat<2,2,T> &mat2x2(const Vec<2,T> &v0, const Vec<2,T> &v1, Mat<2,2,T> &m)
	Mat<3,3,T> &mat3x3(const T &v00, const T &v01, const T &v02, const T &v10, const T &v11, const T &v12, const T &v20, const T &v21, const T &v22, Mat<3,3,T> &m)
	Mat<3,3,T> &mat3x3(const Vec<3,T> &v0, const Vec<3,T> &v1, const Vec<3,T> &v2, Mat<3,3,T> &m)
	Mat<4,4,T> &mat4x4(const Vec<4,T> &v0, const Vec<4,T> &v1, const Vec<4,T> &v2, const Vec<4,T> &v3, Mat<4,4,T> &m)
	
	Mat<4,4,T> &homoMat4x4(const Mat<3,3,T> &mIn, Mat<4,4,T> &m)
	Mat<N,N,T> &diagMat(const Vec<N,T> &v, Mat<N,N,T> &m)

	Mat<3,3,T> &rotXMat(const double &angle, Mat<3,3,T> &m) 
	Mat<3,3,T> &rotYMat(const double &angle, Mat<3,3,T> &m) 
	Mat<3,3,T> &rotZMat(const double &angle, Mat<3,3,T> &m) 
	Mat<3,3,T> &rotMat(const Vec<3,T> &axis, const double &angle, Mat<3,3,T> &m) 

	void invert2x2(const Mat<2,2,T> &A, Mat<2,2,T> &I)
	void invert3x3(const Mat<3,3,T> &A, Mat<3,3,T> &I)
	void invert4x4(const Mat<4,4,T> &A, Mat<4,4,T> &I)

	T determinant2x2(Mat<2,2,T> &A)
	T determinant3x3(Mat<3,3,T> &A)
	T determinant(const Mat<N,N,T>& A)						[js, O(N!)]

	/// Matrix Analysis for square Matrices
	class MatrixProperty;									[js, can be ostream'ed]
	MatrixProperty symmetry(const Mat<M,N,T>& A)			[js, checks symmetries]
	MatrixProperty definiteness(const Mat<N,N,T>& A)		[js, O(N!) definiteness and non-singularity]
	MatrixProperty orthogonaliy(const Mat<N,N,T>& A)		[js, checks orthogonality]
	MatrixProperty diagonalDominance(const Mat<N,N,T>& A)	[js, checks diagonal dominance]
	Vec<N,T> leadingMinors(const Mat<N,N,T>& A)				[js, leading minors, sorted with descending dimensionality]
	MatrixProperty analysis(const Mat<N,N,T>& A)			[js, performs full analysis of A]
	// some of the above functions have additional, optional parameters.
	
*********************************************************************/



/// Simple vector class including common operators and functions. 
/** If TUM3D_VEC_STRICT is defined, the arithmetic operators do only accept vectors
 *  of the same type as operands. Otherwise arrays of type T are valid as well. 
 *  @see tum3D for further functions.
 */
template<int N, class T>
class Vec {
	T val[N];

public:

	/// @name Constructors
	//@{
	/// Default constructor without initialization
	Vec<N,T>()
	{}

	/// Copy constructor
	Vec<N,T>(const Vec<N,T> &v)
	{
		for(int k=0; k<N; ++k) val[k] = v[k];
	}

	/// Cast constructor
	template<int M, class S>
	explicit Vec<N,T>(const Vec<M,S> &v)
	{
		for(int k=0; k<std::min(M,N); ++k) val[k] = T(v[k]);
		for(int k=std::min(M,N); k<N; ++k) val[k] = T(0);
	}

	/// Copy constructor with a given array (must have a length >= N)
	Vec<N,T>(const T *v)
	{
		for(int k=0; k<N; ++k) val[k] = v[k];
	}

	/// Constructor with an initial value for all elements
	explicit Vec<N,T>(const T &w)
	{
		for(int k=0; k<N; ++k) val[k] = w;
	}
	//@}


	/// @name Array subscripting and dereferencing operator
	// @{
	/// Array subscripting operator
	T &operator[](int k)
	{
		assert( (k>=0) && (k<N) );
		return val[k]; 
	}

	/// Constant array subscripting operator
	const T &operator[](int k) const
	{
		assert( (k>=0) && (k<N) );
		return val[k]; 
	}

	/// Dereferencing operator
	operator T *()
	{
		return val;
	}

	/// Constant Dereferencing operator
	operator const T *() const
	{
		return val;
	}
	// @}

	/// @name Assignment operator and arithmetic assignment operators
	// @{
	/// Assignment operator
#ifdef TUM3D_VEC_STRICT
	Vec<N,T> &operator=(const Vec<N,T> &v) 
#else
	Vec<N,T> &operator=(const T *v) 
#endif
	{
		for(int k=0; k<N; ++k) val[k] = v[k]; 
		return (*this);		
	}

	/// Add and assign
#ifdef TUM3D_VEC_STRICT
	Vec<N,T> &operator+=(const Vec<N,T> &v)
#else
	Vec<N,T> &operator+=(const T *v)
#endif
	{
		for(int k=0; k<N; ++k) val[k] += v[k]; 
		return (*this);	
	}

	/// Add and assign scalar (expand)
	Vec<N,T> &operator+=(const T &v)
	{
		for(int k=0; k<N; ++k) val[k] += v; 
		return (*this);	
	}

	/// Subtract and assign
#ifdef TUM3D_VEC_STRICT
	Vec<N,T> &operator-=(const Vec<N,T> &v)
#else
	Vec<N,T> &operator-=(const T *v)
#endif
	{
		for(int k=0; k<N; ++k) val[k] -= v[k]; 
		return (*this);	
	}

	/// Subtract and assign scalar (expand)
	Vec<N,T> &operator-=(const T &v)
	{
		for(int k=0; k<N; ++k) val[k] -= v; 
		return (*this);	
	}

	/// Multiply scalar and assign
	Vec<N,T> &operator*=(const T &w)
	{
		for(int k=0; k<N; ++k) val[k] *= w; 
		return (*this);	
	}

	/// Multiply vector elementwise and assign
#ifdef TUM3D_VEC_STRICT
	Vec<N,T> &operator*=(const Vec<N,T> &v)
#else
	Vec<N,T> &operator*=(const T *v)
#endif
	{
		for(int k=0; k<N; ++k) val[k] *= v[k]; 
		return (*this);	
	}

	/// Divide by scalar and assign
	Vec<N,T> &operator/=(const T &w)
	{
		for(int k=0; k<N; ++k) val[k] /= w; 
		return (*this);	
	}

	/// Modulo by scalar and assign
	Vec<N,T> &operator%=(const T &w)
	{
		for(int k=0; k<N; ++k) val[k] %= w; 
		return (*this);	
	}

	/// Divide elementwise by vector and assign
#ifdef TUM3D_VEC_STRICT
	Vec<N,T> &operator/=(const Vec<N,T> &v)
#else
	Vec<N,T> &operator/=(const T *v)
#endif
	{
		for(int k=0; k<N; ++k) val[k] /= v[k]; 
		return (*this);	
	}

	/// Modulo elementwise by vector and assign
#ifdef TUM3D_VEC_STRICT
	Vec<N,T> &operator%=(const Vec<N,T> &v)
#else
	Vec<N,T> &operator%=(const T *v)
#endif
	{
		for(int k=0; k<N; ++k) val[k] %= v[k]; 
		return (*this);	
	}
	// @}

	/// @name Arithmetic operators
	// @{
	/// Add
	const Vec<N,T> operator+(const Vec<N,T> &v) const
	{
		Vec<N,T> res;
		for(int k=0; k<N; ++k) res[k] = val[k] + v[k]; 
		return res;	
	}

	/// Add scalar (expand)
	const Vec<N,T> operator+(const T &v) const
	{
		Vec<N,T> res;
		for(int k=0; k<N; ++k) res[k] = val[k] + v; 
		return res;	
	}

	/// Subtract
	const Vec<N,T> operator-(const Vec<N,T> &v) const
	{
		Vec<N,T> res;
		for(int k=0; k<N; ++k) res[k] = val[k] - v[k]; 
		return res;	
	}

	/// Subtract scalar (expand)
	const Vec<N,T> operator-(const T &v) const
	{
		Vec<N,T> res;
		for(int k=0; k<N; ++k) res[k] = val[k] - v; 
		return res;	
	}

	/// Multiply with scalar
	const Vec<N,T> operator*(const T &w) const
	{
		Vec<N,T> res;
		for(int k=0; k<N; ++k) res[k] = val[k] * w; 
		return res;	
	}

	/// Multiply with scalar
	friend Vec<N,T> operator*(const T &left, const Vec<N,T> &right)
	{
		return right * left;	
	}	
	
	/// Multiply elementwise with vector
	const Vec<N,T> operator*(const Vec<N,T> &v) const
	{
		Vec<N,T> res;
		for(int k=0; k<N; ++k) res[k] = val[k] * v[k]; 
		return res;	
	}

	/// Divide by scalar
	const Vec<N,T> operator/(const T &w) const
	{
		Vec<N,T> res;
		for(int k=0; k<N; ++k) res[k] = val[k] / w; 
		return res;	
	}

	/// Modulo by scalar
	const Vec<N,T> operator%(const T &w) const
	{
		Vec<N,T> res;
		for(int k=0; k<N; ++k) res[k] = val[k] % w; 
		return res;	
	}

	/// Divide elementwise by vector
	const Vec<N,T> operator/(const Vec<N,T> &v) const
	{
		Vec<N,T> res;
		for(int k=0; k<N; ++k) res[k] = val[k] / v[k]; 
		return res;	
	}
	
	/// Modulo elementwise by vector
	const Vec<N,T> operator%(const Vec<N,T> &v) const
	{
		Vec<N,T> res;
		for(int k=0; k<N; ++k) res[k] = val[k] % v[k]; 
		return res;	
	}

	/// Unary -
	const Vec<N,T> operator-() const
	{
		Vec<N,T> res;
		for(int k=0; k<N; ++k) res[k] = -(*this)[k]; 
		return res;	
	}
	// @}


	/// @name Vector functions
	// @{
	/// Clear the vector to zero
	void clear()
	{
		for(int k=0; k<N; ++k) 
			val[k] = T(0);
	}
	
	/// Returns the norm of the vector.
	T norm() const
	{
		return T(sqrt(normSqr()));
	}

	/// Returns the squared norm of the vector.
	T normSqr() const
	{
		T sum(0);
		for(int k=0; k<N; ++k) sum += val[k]*val[k];
		return sum;
	}

	/// Returns the norm of the vector as float
	float normf() const
	{
		float sum = 0.0f;
		for(int k=0; k<N; ++k) sum += float(val[k]) * float(val[k]);
		return sqrt(sum);
	}

	/// Returns the dot product.
#ifdef TUM3D_VEC_STRICT
	T dot(const Vec<N,T> &v) const
#else
	T dot(const T *v) const
#endif	
	{
		T sum(0);
		for(int k=0; k<N; ++k) sum += val[k]*v[k];	
		return sum;
	}

	/// Comparison operator
#ifdef TUM3D_VEC_STRICT
	bool operator==(const Vec<N,T> &v) const
#else
	bool operator==(const T *v) const
#endif
	{
		bool res = true;
		for(int k=0; k<N; ++k) res = res && (val[k] == v[k]);	
		return res;
	}

#ifdef TUM3D_VEC_STRICT
	bool operator!=(const Vec<N,T> &v) const
#else
	bool operator!=(const T *v) const
#endif
	{
		return !(*this == v);
	}

#ifdef TUM3D_VEC_STRICT
	bool operator<(const Vec<N,T> &v) const
#else
	bool operator<(const T *v) const
#endif
	{
		bool res = true;
		for(int k=0; k<N; ++k) res = res && (val[k] < v[k]);	
		return res;
	}

#ifdef TUM3D_VEC_STRICT
	bool operator>(const Vec<N,T> &v) const
#else
	bool operator>(const T *v) const
#endif
	{
		bool res = true;
		for(int k=0; k<N; ++k) res = res && (val[k] > v[k]);	
		return res;
	}

#ifdef TUM3D_VEC_STRICT
	bool operator<=(const Vec<N,T> &v) const
#else
	bool operator<=(const T *v) const
#endif
	{
		bool res = true;
		for(int k=0; k<N; ++k) res = res && (val[k] <= v[k]);	
		return res;
	}

#ifdef TUM3D_VEC_STRICT
	bool operator>=(const Vec<N,T> &v) const
#else
	bool operator>=(const T *v) const
#endif
	{
		bool res = true;
		for(int k=0; k<N; ++k) res = res && (val[k] >= v[k]);	
		return res;
	}

	/// Component-wise comparison
	const Vec<N,bool> compLess(const Vec<N,T> &v) const
	{
		Vec<N,bool> res;
		for(int k=0; k<N; ++k) res[k] = val[k] < v[k];
		return res;
	}

	const Vec<N,bool> compLessEqual(const Vec<N,T> &v) const
	{
		Vec<N,bool> res;
		for(int k=0; k<N; ++k) res[k] = val[k] <= v[k];
		return res;
	}

	const Vec<N,bool> compGreater(const Vec<N,T> &v) const
	{
		Vec<N,bool> res;
		for(int k=0; k<N; ++k) res[k] = val[k] > v[k];
		return res;
	}

	const Vec<N,bool> compGreaterEqual(const Vec<N,T> &v) const
	{
		Vec<N,bool> res;
		for(int k=0; k<N; ++k) res[k] = val[k] >= v[k];
		return res;
	}

	const Vec<N,bool> compEqual(const Vec<N,T> &v) const
	{
		Vec<N,bool> res;
		for(int k=0; k<N; ++k) res[k] = val[k] == v[k];
		return res;
	}

	const Vec<N,bool> compNEqual(const Vec<N,T> &v) const
	{
		Vec<N,bool> res;
		for(int k=0; k<N; ++k) res[k] = val[k] != v[k];
		return res;
	}

	Vec<N,T> abs() const {
		Vec<N,T> r;

		for(int k=0; k<N; ++k)
			r.val[k] = std::abs(val[k]);

		return r;
	}

	/// Returns the maximal value of the vector's elements
	T maximum() const
	{
		T max = val[0];
		for(int k=1; k<N; ++k)
			if (max < val[k])
				max = val[k];
		return max;
	}

	/// Returns the minimal value of the vector's elements
	T minimum() const
	{
		T min = val[0];
		for(int k=1; k<N; ++k)
			if (min > val[k])
				min = val[k];
		return min;
	}

	/// Returns the index of the maximal of the vector's elements
	int maximumIndex() const
	{
		T max = val[0];
		int maxIndex = 0;
		for(int k=1; k<N; ++k) {
			if (max < val[k]) {
				max = val[k];
				maxIndex = k;
			}
		}
		return maxIndex;
	}

	/// Returns the index of the minimal of the vector's elements
	int minimumIndex() const
	{
		T min = val[0];
		int minIndex = 0;
		for(int k=1; k<N; ++k) {
			if (min > val[k]) {
				min = val[k];
				minIndex = k;
			}
		}
		return minIndex;
	}
	// @}
};

/// Divide scalar by vector
template<int N, class T>
const Vec<N,T> operator/(const T &s, const Vec<N,T>& vec)
{
	Vec<N,T> res;
	for(int k=0; k<N; ++k) res[k] = s / vec[k]; 
	return res;	
}


/// Simple vector class specialized for 2D. 
/** Adds common Constructors and swizzle operators
 *  @see tum3D for further functions.
 */
template<class T>
class Vec2 : public Vec<2,T>
{
public:

	/// @name Constructors
	//@{
	/// Default constructor without initialization
	Vec2<T>()
	{}

	/// Copy constructor
	Vec2<T>(const Vec<2,T> &v)
	{
		(*this)[0] = v[0]; 
		(*this)[1] = v[1];
	}
	
	/// Cast constructor
	template<int M, class S>
	explicit Vec2<T>(const Vec<M,S> &v)
	{
		for(int k=0; k<std::min(M,2); ++k) (*this)[k] = T(v[k]);
		for(int k=std::min(M,2); k<2; ++k) (*this)[k] = T(0);
	}

	/// Copy constructor with a given array (must have a length >= N)
	Vec2<T>(const T *v)
	{
		(*this)[0] = v[0]; 
		(*this)[1] = v[1];
	}

	/// Constructor with an initial value for all elements
	explicit Vec2<T>(const T &w)
	{
		(*this)[0] = w; 
		(*this)[1] = w;
	}
		
	/// Constructor with an initial value for each component
	Vec2<T>(const T &x, const T &y)
	{
		(*this)[0] = x; 
		(*this)[1] = y;
	}
	
	//@}

	void set(const T &x, const T &y)
	{
		(*this)[0] = x; 
		(*this)[1] = y;
	}
	
	/// @name Swizzle operators
	//@{
	/// Get x
	T &x() { return (*this)[0]; }
	const T &x() const { return (*this)[0]; }
	
	/// Get y
	T &y() { return (*this)[1]; }
	const T &y() const { return (*this)[1]; }
	
	/// Get xy
	const Vec2<T> xy() const { return (*this); }
		
	/// Get yx
	const Vec2<T> yx() const { return Vec2<T>(y(), x()); }
	
	//@}

	T area() const
	{
		return x() * y();
	}
};


/// Simple vector class specialized for 3D. 
/** Adds common Constructors and swizzle operators
 *  @see tum3D for further functions.
 */
template<class T>
class Vec3 : public Vec<3,T>
{
public:

	/// @name Constructors
	//@{
	/// Default constructor without initialization
	Vec3<T>()
	{}

	/// Copy constructor
	Vec3<T>(const Vec<3,T> &v)
	{
		(*this)[0] = v[0]; 
		(*this)[1] = v[1];
		(*this)[2] = v[2];
	}
	
	/// Cast constructor
	template<int M, class S>
	explicit Vec3<T>(const Vec<M,S> &v)
	{
		for(int k=0; k<std::min(M,3); ++k) (*this)[k] = T(v[k]);
		for(int k=std::min(M,3); k<3; ++k) (*this)[k] = T(0);
	}

	/// Copy constructor with a given array (must have a length >= N)
	Vec3<T>(const T *v)
	{
		(*this)[0] = v[0]; 
		(*this)[1] = v[1];
		(*this)[2] = v[2];
	}

	/// Constructor with an initial value for all elements
	explicit Vec3<T>(const T &w)
	{
		(*this)[0] = w; 
		(*this)[1] = w;
		(*this)[2] = w;
	}
		
	/// Constructor with an initial value for each component
	Vec3<T>(const T &x, const T &y, const T &z)
	{
		(*this)[0] = x; 
		(*this)[1] = y;
		(*this)[2] = z;
	}

	/// Constructor from smaller Vec + scalar
	Vec3<T>(const Vec2<T> &v, const T &z)
	{
		(*this)[0] = v[0]; 
		(*this)[1] = v[1];
		(*this)[2] = z;
	}

	/// Constructor from scalar + smaller Vec
	Vec3<T>(const T &x, const Vec2<T> &v)
	{
		(*this)[0] = x; 
		(*this)[1] = v[0];
		(*this)[2] = v[1];
	}

	//@}

	void set(const T &x, const T &y, const T &z)
	{
		(*this)[0] = x; 
		(*this)[1] = y;
		(*this)[2] = z;
	}
	
	/// @name Swizzle operators
	//@{
	/// Get x
	T &x() { return (*this)[0]; }
	const T &x() const { return (*this)[0]; }
	
	/// Get y
	T &y() { return (*this)[1]; }
	const T &y() const { return (*this)[1]; }
	
	/// Get z
	T &z() { return (*this)[2]; }
	const T &z() const { return (*this)[2]; }
	
	
	/// Get xy
	const Vec2<T> xy() const { return Vec2<T>(x(), y()); }
	
	/// Get xz
	const Vec2<T> xz() const { return Vec2<T>(x(), z()); }
	
	/// Get yx
	const Vec2<T> yx() const { return Vec2<T>(y(), x()); }
	
	/// Get yz
	const Vec2<T> yz() const { return Vec2<T>(y(), z()); }
	
	/// Get zx
	const Vec2<T> zx() const { return Vec2<T>(z(), x()); }
	
	/// Get zy
	const Vec2<T> zy() const { return Vec2<T>(z(), y()); }
	
	
	/// Get xyz
	const Vec3<T> xyz() const { return (*this); }
	
	/// Get xzy
	const Vec3<T> xzy() const { return Vec3<T>(x(), z(), y()); }
	
	/// Get yxz
	const Vec3<T> yxz() const { return Vec3<T>(y(), x(), z()); }
	
	/// Get yzx
	const Vec3<T> yzx() const { return Vec3<T>(y(), z(), x()); }
	
	/// Get zxy
	const Vec3<T> zxy() const { return Vec3<T>(z(), x(), y()); }
	
	/// Get zyx
	const Vec3<T> zyx() const { return Vec3<T>(z(), y(), x()); }
	
	//@}

	T volume() const
	{
		return x() * y() * z();
	}
};


/// Simple vector class specialized for 4D. 
/** Adds common Constructors and swizzle operators
 *  @see tum3D for further functions.
 */
template<class T>
class Vec4 : public Vec<4,T>
{
public:

	/// @name Constructors
	//@{
	/// Default constructor without initialization
	Vec4<T>()
	{}

	/// Copy constructor
	Vec4<T>(const Vec<4,T> &v)
	{
		(*this)[0] = v[0]; 
		(*this)[1] = v[1];
		(*this)[2] = v[2];
		(*this)[3] = v[3];
	}
	
	/// Cast constructor
	template<int M, class S>
	explicit Vec4<T>(const Vec<M,S> &v)
	{
		for(int k=0; k<std::min(M,4); ++k) (*this)[k] = T(v[k]);
		for(int k=std::min(M,4); k<4; ++k) (*this)[k] = T(0);
	}

	/// Copy constructor with a given array (must have a length >= N)
	Vec4<T>(const T *v)
	{
		(*this)[0] = v[0]; 
		(*this)[1] = v[1];
		(*this)[2] = v[2];
		(*this)[3] = v[3];
	}

	/// Constructor with an initial value for all elements
	explicit Vec4<T>(const T &w)
	{
		(*this)[0] = w; 
		(*this)[1] = w;
		(*this)[2] = w;
		(*this)[3] = w;
	}
		
	/// Constructor with an initial value for each component
	Vec4<T>(const T &x, const T &y, const T &z, const T &w)
	{
		(*this)[0] = x; 
		(*this)[1] = y;
		(*this)[2] = z;
		(*this)[3] = w;
	}

	Vec4<T>(const Vec3<T> &v, const T &w)
	{
		(*this)[0] = v[0]; 
		(*this)[1] = v[1];
		(*this)[2] = v[2];
		(*this)[3] = w;
	}

	Vec4<T>(const T &x, const Vec3<T> &v)
	{
		(*this)[0] = x; 
		(*this)[1] = v[0];
		(*this)[2] = v[1];
		(*this)[3] = v[2];
	}

	//@}

	void set(const T &x, const T &y, const T &z, const T &w)
	{
		(*this)[0] = x; 
		(*this)[1] = y;
		(*this)[2] = z;
		(*this)[3] = w;
	}
	
	/// @name Swizzle operators
	//@{
	/// Get x
	T &x(){ return (*this)[0]; }
	const T &x() const { return (*this)[0]; }
	
	/// Get y
	T &y(){ return (*this)[1]; }
	const T &y() const { return (*this)[1]; }
	
	/// Get z
	T &z(){ return (*this)[2]; }
	const T &z() const { return (*this)[2]; }

	/// Get w
	T &w(){ return (*this)[3]; }
	const T &w() const { return (*this)[3]; }
	
	
	/// Get xy
	const Vec2<T> xy() const { return Vec2<T>(x(), y()); }
	
	/// Get xz
	const Vec2<T> xz() const { return Vec2<T>(x(), z()); }
	
	/// Get xw
	const Vec2<T> xw() const { return Vec2<T>(x(), w()); }
	
	/// Get yx
	const Vec2<T> yx() const { return Vec2<T>(y(), x()); }
	
	/// Get yz
	const Vec2<T> yz() const { return Vec2<T>(y(), z()); }
	
	/// Get yw
	const Vec2<T> yw() const { return Vec2<T>(y(), w()); }
	
	/// Get zx
	const Vec2<T> zx() const { return Vec2<T>(z(), x()); }
	
	/// Get zy
	const Vec2<T> zy() const { return Vec2<T>(z(), y()); }
	
	/// Get zw
	const Vec2<T> zw() const { return Vec2<T>(z(), w()); }
	
	/// Get wx
	const Vec2<T> wx() const { return Vec2<T>(w(), x()); }
	
	/// Get wy
	const Vec2<T> wy() const { return Vec2<T>(w(), y()); }
	
	/// Get wz
	const Vec2<T> wz() const { return Vec2<T>(w(), z()); }
	
	
	/// Get xyz
	const Vec3<T> xyz() const { return Vec3<T>(x(), y(), z()); }
	
	/// Get xzy
	const Vec3<T> xzy() const { return Vec3<T>(x(), z(), y()); }
	
	/// Get yxz
	const Vec3<T> yxz() const { return Vec3<T>(y(), x(), z()); }
	
	/// Get yzx
	const Vec3<T> yzx() const { return Vec3<T>(y(), z(), x()); }
	
	/// Get zxy
	const Vec3<T> zxy() const { return Vec3<T>(z(), x(), y()); }
	
	/// Get zyx
	const Vec3<T> zyx() const { return Vec3<T>(z(), y(), x()); }
	
	
	/// Get xyzw
	const Vec4<T> xyzw() const { return (*this); }
	
	/// Get xzyw
	const Vec4<T> xzyw() const { return Vec4<T>(x(), z(), y(), w()); }
	
	/// Get yxzw
	const Vec4<T> yxzw() const { return Vec4<T>(y(), x(), z(), w()); }
	
	/// Get yzxw
	const Vec4<T> yzxw() const { return Vec4<T>(y(), z(), x(), w()); }
	
	/// Get zxyw
	const Vec4<T> zxyw() const { return Vec4<T>(z(), x(), y(), w()); }
	
	/// Get zyxw
	const Vec4<T> zyxw() const { return Vec4<T>(z(), y(), x(), w()); }
	//@}
};





/// Simple matrix class including common operators and functions for MxN matrices.
/** If TUM3D_MAT_STRICT is defined, the arithmetic operators do only accept matrices
 *  and vectors of the same type as operands. Otherwise arrays of type T are valid as well.
 *  @see tum3D for further functions.
 */
template<int M, int N, class T>
class Mat
{
	// row-major
	T val[M*N];

public:

	/// @name Constructors
	// @{
	/// Default constructor
	Mat<M,N,T>()
	{}

	/// Copy constructor
	Mat<M,N,T>(const Mat<M,N,T> &m)
	{
		for(int k=0; k<M*N; ++k) val[k] = m[k]; 
	}

	/// Cast constructor
	template<int K, int L, class S>
	explicit Mat<M,N,T>(const Mat<K,L,S> &m)
	{
		for(int k=0; k<std::min(K,M); ++k)
		{
			for(int l=0; l<std::min(L,N); ++l)
				get(k,l) = T(m.get(k,l));
			for(int l=std::min(L,N); l<N; ++l)
				get(k,l) = T(0);
		}
		for(int k=std::min(K,M); k<M; ++k)
			for(int l=0; l<N; ++l)
				get(k,l) = T(0);
	}

	/// Copy constructor with a given array (must have a length >= M*N)
	Mat<M,N,T>(const T *m)
	{ 
		for(int k=0; k<M*N; ++k) val[k] = m[k]; 
	}

	/// Constructor with an initial value for all elements
	explicit Mat<M,N,T>(const T &w)
	{ 
		for(int k=0; k<M*N; ++k) val[k] = w; 
	}
	// @}

	/// @name Array subscripting, matrix access and dereferencing operators
	// @{
	/// Array subscripting operator
	T &operator[](int k)
	{
		assert( (k>=0) && (k<M*N) );
		return val[k]; 
	}

	/// Constant array subscripting operator.
	const T &operator[](int k) const
	{
		assert( (k>=0) && (k<M*N) );
		return val[k]; 
	}

	/// Matrix access
	T &get(int k, int l)
	{
		assert( (k>=0) && (l>=0) && (k<M) && (l<N) );
		return val[k*N+l]; 
	}
	
	/// Matrix access
	const T &get(int k, int l) const
	{
		assert( (k>=0) && (l>=0) && (k<M) && (l<N) );
		return val[k*N+l]; 
	}
	
	/// Matrix access to column vectors
	Vec<M,T> getCol(int l) const
	{
		assert( (l>=0) && (l<N) );
		Vec<M,T> v;
		for(int k=0; k<M; ++k)
			v[k] = val[k*N+l];
		return v;
	}
		
	/// Matrix access to row vectors
	Vec<N,T> getRow(int k) const
	{
		assert( (k>=0) && (k<M) );
		return Vec<N,T>(&val[k*N]); 
	}
	

	/// Dereferencing operator
	operator T *()
	{
		return val;
	}

	/// Constant dereferencing operator
	operator const T *() const
	{
		return val;
	}

	T* data()
	{
		return &val[0];
	}

	const T* data() const
	{
		return &val[0];
	}
	// @}

	/// @name Assignment operator and arithmetic assignment operators
	// @{
	/// Assignmet operator
#ifdef TUM3D_MAT_STRICT
	Mat<M,N,T> &operator=(const Mat<M,N,T> &m) 
#else
	Mat<M,N,T> &operator=(const T *m) 
#endif
	{
		for(int k=0; k<M*N; ++k) val[k] = m[k]; 
		return (*this);		
	}

	/// Add and assign
#ifdef TUM3D_MAT_STRICT
	Mat<M,N,T> &operator+=(const Mat<M,N,T> &m)
#else
	Mat<M,N,T> &operator+=(const T *m)
#endif
	{
		for(int k=0; k<M*N; ++k) val[k] += m[k]; 
		return (*this);	
	}

	/// Subtract and assign
#ifdef TUM3D_MAT_STRICT
	Mat<M,N,T> &operator-=(const Mat<M,N,T> &m)
#else
	Mat<M,N,T> &operator-=(const T *m)
#endif
	{
		for(int k=0; k<M*N; ++k) val[k] -= m[k]; 
		return (*this);	
	}

	/// Multiply a scalar and assign
	Mat<M,N,T> &operator*=(const T &w)
	{
		for(int k=0; k<M*N; ++k) val[k] *= w; 
		return (*this);	
	}

	/// Multiply a matrix and assign
	template<int L>
#ifdef TUM3D_MAT_STRICT
	Mat<M,L,T> &operator*=(const Mat<N,L,T> &m)
#else
	Mat<M,L,T> &operator*=(const T *m)
#endif
	{
		Mat<M,L,T> erg;
		for(int i=0; i<M; ++i)
			for(int j=0; j<L; ++j){
				T sum(0);
				for(int k=0; k<N; k++)
					sum += val[i*N + k] * m[k*L + j];
				erg[i*L + j] = sum;
				}
		*this = erg;
		return *this;
	}

	/// Divide by a scalar and assign
	Mat<M,N,T> &operator/=(const T &w)
	{
		for(int k=0; k<M*N; ++k) val[k] /= w; 
		return (*this);	
	}

	/// Modulo by a scalar and assign
	Mat<M,N,T> &operator%=(const T &w)
	{
		for(int k=0; k<M*N; ++k) val[k] %= w; 
		return (*this);	
	}

	/// Sum of two matrices
#ifdef TUM3D_MAT_STRICT
	Mat<M,N,T> operator+(const Mat<M,N,T> &m) const
#else
	Mat<M,N,T> operator+(const T *m) const
#endif
	{
		Mat<M,N,T> res;
		for(int k=0; k<M*N; ++k) res[k] = val[k] + m[k];
		return res;
	}

	/// Difference of two matrices
#ifdef TUM3D_MAT_STRICT
	Mat<M,N,T> operator-(const Mat<M,N,T> &m) const
#else
	Mat<M,N,T> operator-(const T *m) const
#endif
	{
		Mat<M,N,T> res;
		for(int k=0; k<M*N; ++k) res[k] = val[k] - m[k];
		return res;
	}

	/// Multiply matrix by scalar
	Mat<M,N,T> operator*(const T &w) const
	{
		Mat<M,N,T> res;
		for(int k=0; k<M*N; ++k) res[k] = val[k] * w;
		return res;
	}

	friend Mat<M,N,T> operator*(const T &left, const Mat<M,N,T> &right)
	{
		return right * left;
	}

	/// Product of matrix and vector
#ifdef TUM3D_MAT_STRICT	
	const Vec<M,T> operator*(Vec<N,T> v) const
#else
	const Vec<M,T> operator*(Vec<N,T> v) const
#endif
	{
		Vec<M,T> res;
		for(int j=0; j<M; ++j){
			T sum = T(0);
			for(int k=0; k<N; ++k)
				sum += val[j*N + k] * v[k];
			res[j] = sum;
			}
		return res;
	}

	/// Product of two matrices
	template<int L>
#ifdef TUM3D_MAT_STRICT
	Mat<M,L,T> operator*(const Mat<N,L,T> &m) const
#else
	Mat<M,L,T> operator*(const T *m) const
#endif
	{
		Mat<M,L,T> res;
		for(int i=0; i<M; ++i)
			for(int j=0; j<L; ++j){
				T sum(0);
				for(int k=0; k<N; k++)
					sum += val[i*N + k] * m[k*L + j];
				res[i*L + j] = sum;
				}
		return res;
	}

	/// Divide matrix by scalar
	Mat<M,N,T> operator/(const T &w) const
	{
		Mat<M,N,T> res;
		for(int k=0; k<M*N; ++k) res[k] = val[k] / w;
		return res;
	}

	/// Modulo matrix by scalar
	Mat<M,N,T> operator%(const T &w) const
	{
		Mat<M,N,T> res;
		for(int k=0; k<M*N; ++k) res[k] = val[k] % w;
		return res;
	}

	/// Unary -
	Mat<M,N,T> operator-() const
	{
		Mat<M,N,T> res;
		for(int k=0; k<M*N; ++k) res[k] = -val[k];
		return res;
	}

	/// Comparison
#ifdef TUM3D_MAT_STRICT
	bool operator==(const Mat<M,N,T> &m) const
#else
	bool operator==(const T *m) const
#endif
	{
		bool res = true;
		for(int k=0; k<M*N; ++k) res = res && (val[k] == m[k]);	
		return res;
	}

#ifdef TUM3D_MAT_STRICT
	bool operator!=(const Mat<M,N,T> &m) const
#else
	bool operator!=(const T *m) const
#endif
	{
		return !(*this == m);
	}

	// @}

	/// @name Matrix functions
	// @{
	
	/// Clear the matrix to zero
	void clear()
	{
		for(int k=0; k<M*N; ++k) 
			val[k] = T(0);
	}
	
	/// Multiply with matrix B and store result in erg
	template<int L>
#ifdef TUM3D_MAT_STRICT
	Mat<M,L,T> &multMat(const Mat<N,L,T> &B, Mat<M,L,T> &erg) const
#else
	T *multMat(const T *B, T *erg) const
#endif
	{
		for(int i=0; i<M; ++i)
			for(int j=0; j<L; ++j){
				T sum(0);
				for(int k=0; k<N; k++)
					sum += val[i*N + k] * B[k*L + j];
				erg[i*L + j] = sum;
				}
		return erg;
	}
	
	
	/// Multiply with matrix B and accumulate result to matrix erg
	template<int L>
#ifdef TUM3D_MAT_STRICT
	Mat<M,L,T> &multMatAdd(const Mat<N,L,T> &B, Mat<M,L,T> &erg) const
#else
	T *multMatAdd(const T *B, T *erg) const
#endif
	{
		for(int i=0; i<M; ++i)
			for(int j=0; j<L; ++j){
				T sum(0);
				for(int k=0; k<N; k++)
					sum += val[i*N + k] * B[k*L + j];
				erg[i*L + j] += sum;
				}
		return erg;
	}

	
	/// Multiply with matrix B^T and store result in erg
	template<int L>
#ifdef TUM3D_MAT_STRICT
	Mat<M,L,T> &multMatT(const Mat<L,N,T> &B, Mat<M,L,T> &erg) const
#else
	T *multMatT(const T *B, T *erg) const
#endif
	{
		for(int i=0; i<M; ++i)
			for(int j=0; j<L; ++j){
				T sum(0);
				for(int k=0; k<N; k++)
					sum += val[i*N + k] * B[j*N + k];
				erg[i*L + j] = sum;
				}
		return erg;
	}

	/// Multiply with matrix B^T and accumulate result to matrix erg
	template<int L>
#ifdef TUM3D_MAT_STRICT
	Mat<M,L,T> &multMatTAdd(const Mat<L,N,T> &B, Mat<M,L,T> &erg) const
#else
	T *multMatTAdd(const T *B, T *erg) const
#endif
	{
		for(int i=0; i<M; ++i)
			for(int j=0; j<L; ++j){
				T sum(0);
				for(int k=0; k<N; k++)
					sum += val[i*N + k] * B[j*N + k];
				erg[i*L + j] += sum;
				}
		return erg;
	}
	
	
	/// Product of matrix and vector b. Result is written to erg and its reference is returned.
#ifdef TUM3D_MAT_STRICT
	Vec<M,T> &multVec(const Vec<N,T> &b, Vec<M,T> &erg) const
#else
	T *multVec(const T *b, T *erg) const
#endif
	{
		for(int j=0; j<M; ++j){
			T sum(0);
			for(int k=0; k<N; ++k)
				sum += val[j*N + k] * b[k];
			erg[j] = sum;
			}
		return erg;
	}
	
	
	/// Product of matrix and vector b. Result is added to vector erg and its reference is returned.
#ifdef TUM3D_MAT_STRICT
	Vec<M,T> &multVecAdd(const Vec<N,T> &b, Vec<M,T> &erg) const
#else
	T *multVecAdd(const T *b, T *erg) const
#endif
	{
		for(int j=0; j<M; ++j){
			T sum(0);
			for(int k=0; k<N; ++k)
				sum += val[j*N + k] * b[k];
			erg[j] += sum;
			}
		return erg;
	}
	
	
	/// Product of transposed matrix and vector b. Result is written to erg and its reference is returned.
#ifdef TUM3D_MAT_STRICT
	Vec<N,T> &multTVec(const Vec<M,T> &b, Vec<N,T> &erg) const
#else
	T *multTVec(const T *b, T *erg) const
#endif
	{
		for(int j=0; j<N; ++j){
			T sum = 0;
			for(int k=0; k<M; ++k)
				sum += val[k*N + j] * b[k];
			erg[j] = sum;
			}
		return erg;
	}
	
	
	/// Product of transposed matrix and vector b. Result is added to vector erg and its reference is returned.
#ifdef TUM3D_MAT_STRICT
	Vec<N,T> &multTVecAdd(const Vec<M,T> &b, Vec<N,T> &erg) const
#else
	T *multTVecAdd(const T *b, T *erg) const
#endif
	{
		for(int j=0; j<N; ++j){
			T sum = 0;
			for(int k=0; k<M; ++k)
				sum += val[k*N + j] * b[k];
			erg[j] += sum;
			}
		return erg;
	}
	
	/// Transpose matrix. Result is written to erg and its reference is returned.
#ifdef TUM3D_MAT_STRICT
	Mat<N,M,T> &transpose(Mat<N,M,T> &erg) const
#else
	T *transpose(T *erg) const
#endif
	{
		for(int i=0; i<M; i++)
			for(int j=0; j<N; j++)
				erg[M*j+i] = val[N*i+j];
		return erg;
	}
	
	/// Gauss Elimination: Perform Gaussian elimination on matrix and given vector b. Solution is written to x. 
	/** WARNING: this matrix and vector b are destroyed
	*/
#ifdef TUM3D_MAT_STRICT
	void gaussElim(Vec<M,T> &b, Vec<N,T> &x)
#else
	void gaussElim(T *b, T *x)
#endif
	{
		for(int k=0; k<M; ++k){
			T *row = &val[k*N];
			T fac = (row[k] != T(0)) ? T(1)/row[k] : T(1);
			for(int l=k+1; l<M; ++l){
				T *actRow = &val[l*N];
				T actFac = fac * actRow[k];
				for(int ri=k+1; ri<N; ++ri)
					actRow[ri] -= actFac * row[ri];
				b[l] -= actFac * b[k];
				}
			}
			
		// Back substitution
		for(int k=M-1; k>=0; --k){
			x[k] = b[k];
			for(int l=k+1; l<N; ++l)
				x[k] -= x[l] * val[k*N+l];
			x[k] = (val[k*N+k] != T(0)) ? x[k] / val[k*N+k] : T(0) ;
			}
	}
	
	
	/// Gauss Elimination: Perform Gaussian elimination on matrix and given vector b using row pivoting. Solution is written to x. 
	/** WARNING: this matrix and vector b are destroyed
	*/
#ifdef TUM3D_MAT_STRICT
	void gaussElimRowPivot(Vec<M,T> &b, Vec<N,T> &x)
#else
	void gaussElimRowPivot(T *b, T *x)
#endif
	{
		Mat<M,N,T> A(*this);
		Vec<M,T> ba(b);
		
		Vec<M,int> perm; // Store row permutation
		for(int i=0; i<M; ++i) 
			perm[i] = i;
		for(int k=0; k<M; ++k){
			
			// Find pivot
			T mx(0);
			int pi(k); // Default initilization: use current row
			for(int li=k; li<M; ++li){
				T m = fabs(val[perm[li]*N+k]);
				if (m > mx){ mx = m; pi = li;}
				}
			
			int old = perm[k];	
			perm[k] = perm[pi]; // Update permutation
			perm[pi] = old;

			
			pi = perm[k]; // pi is row index of current pivot row
			T *row = &val[pi*N];
			T fac = (row[k] != T(0)) ? T(1)/row[k] : T(1);
			for(int l=k+1; l<M; ++l){
				T *actRow = &val[perm[l]*N];
				T actFac = fac * actRow[k];
				for(int ri=k+1; ri<N; ++ri)
					actRow[ri] -= actFac * row[ri];
				b[perm[l]] -= actFac * b[pi];
				}
				
			
			}

			
		// Back substitution
		for(int k=M-1; k>=0; --k){
			x[k] = b[perm[k]];
			for(int l=k+1; l<N; ++l)
				x[k] -= x[l] * val[perm[k]*N+l];
			x[k] = (val[perm[k]*N+k] != T(0)) ? x[k] / val[perm[k]*N+k] : T(1) ;
			}		
	}
	
	
	/// Compute largest eigenvalue (in lambda) and eigenvector (in x). Perform power method. Returns convergence value, that should be close to 1 (positive eigenvalue) or -1 (negative eigenvalue). Optional arguments are the maximum number of iteration steps, and a flag indicates whether x is used as start vector, or a start vector is computed from the matrix. 
	T largestEigenvec(T &lambda, Vec<N,T> &x, int maxSteps=25, bool computeStartVec=true)
	{
		T err(0);
		Vec<M,T> xold;
		if (computeStartVec)
		{
			T mx(0);
			int iMax=0;
			for(int i=0; i<M; ++i)
			{
				lambda = getRow(i).norm();
				if (lambda > mx){ mx = lambda; iMax = i; }
			}
			x = getRow(iMax);
		}
		// Normalize x
		lambda = x.norm();
		x *= (lambda != T(0.0)) ? T(1.0)/lambda : T(1.0);
		int ic(0);
		do{
			xold = x;
			multVec(xold, x);
			
			lambda = x.norm();
			x *= (lambda != T(0.0)) ? T(1.0)/lambda : T(1.0);
		
			err = x.dot(xold);
			
			++ic;
		}
		while( (err*err < T(0.98)) && (ic <= maxSteps) );
		
		if (err < T(0)) lambda = -lambda;
		return err;
	
	}	
	
	
	T largestEigenvec2(T &lambda, Vec<N,T> &x, int maxSteps=25, bool computeStartVec=true)
	{
		T err(0);
		T norm;
		Vec<M,T> xold;
		if (computeStartVec)
		{
			T mx(0);
			int iMax=0;
			for(int i=0; i<M; ++i)
			{
				lambda = getRow(i).norm();
				if (lambda > mx){ mx = lambda; iMax = i; }
			}
			x = getRow(iMax);
		}
		
		norm = x.norm();
		x *= (norm != T(0.0)) ? T(1.0)/norm : T(1.0);
		
		int ic(0);
		do{
			multVec(x, xold);
			lambda =  x.dot(xold) / x.normSqr();
			norm = xold.norm();
			xold *= (norm != T(0.0)) ? T(1.0)/norm : T(1.0);
			err = x.dot(xold);
			if (err*err > T(0.98)) break;
			
			Mat<N,N,T> tmp(*this);
			for(int i=0; i<N; i++)
				tmp.get(i,i) -= lambda;
			
			tmp.gaussElim(x, xold);
			x = xold;
			norm = x.norm();
			x *= (norm != T(0.0)) ? T(1.0)/norm : T(1.0);
			
			++ic;
		}
		while( (err*err < T(0.98)) && (ic <= maxSteps) );
		
		if (err < T(0)) lambda = -lambda;
		return err;
	
	}	
	
	
	// @}
};




/// Some useful methods are domiciled in namespace tum3D
namespace tum3D
{
	
	/// Cross product of two 3D-vectors, a x b. Result is stored in erg and its reference is returned.
	template<class T>
#ifdef TUM3D_VEC_STRICT
	Vec<3,T> &crossProd(const Vec<3,T> &a, const Vec<3,T> &b, Vec<3,T> &erg)
#else
	T *crossProd(const T *a, const T *b, T *erg)
#endif
	{
		erg[0] = a[1]*b[2] - a[2]*b[1];
		erg[1] = a[2]*b[0] - a[0]*b[2];
		erg[2] = a[0]*b[1] - a[1]*b[0];
		return erg;
	}


	/// Dot product of two vectors, a ° b. 
	template<int N, class T>
#ifdef TUM3D_VEC_STRICT
	T dotProd(const Vec<N,T> &a, const Vec<N,T> &b)
#else
	T dotProd(const T *a, const T *b)
#endif
	{
		T sum(0);
		for(int k=0; k<N; ++k) sum += a[k]*b[k];	
		return sum;
	}

	/// Tensor product of two vectors a and b. Result is stored in erg and its reference is returned.
	template<int M, int N, class T>
#ifdef TUM3D_VEC_STRICT
	Mat<M,N,T> &tensorProd(const Vec<M,T> &a, const Vec<N,T> &b, Mat<M,N,T> &erg)
#else
	T *tensorProd(const T *a, const T *b, T *erg)
#endif
	{
		for(int k=0; k<M; ++k)
			for(int l=0; l<N; ++l)
				erg[k*N + l] = a[k] * b[l];
		return erg;
	}

	/// Euclidean distance between the points v0 and v1
	template<int N, class T>
#ifdef TUM3D_VEC_STRICT
	T distance(const Vec<N,T> &v0, const Vec<N,T> &v1)
#else
	T distance(const T *v0, const T *v1)
#endif
	{
		T sum = T(0.0);
		for(int k=0; k<N; ++k) sum += (v0[k] - v1[k]) * (v0[k] - v1[k]);
		return sqrt(sum);
	}

	/// Normalize vector
	template<int N, class T>
#ifdef TUM3D_VEC_STRICT
	Vec<N,T> &normalize(Vec<N,T> &v)
#else
	T *normalize(T *v)
#endif
	{
		T sum = T(0.0);
		for(int k=0; k<N; ++k) sum += v[k] * v[k];
		sum = (sum != T(0.0)) ? T(1.0)/sqrt(sum) : T(1.0);
		for(int k=0; k<N; ++k) v[k] *= sum;
		return v;
	}


	/// Dehomogenize vector
	template<class T>
	const Vec3<T> dehomogenize(const Vec<4,T> &v)
	{
		Vec3<T> res;
		res[0] = v[0] / v[3];
		res[1] = v[1] / v[3];
		res[2] = v[2] / v[3];
		return res;
	}

	/// Cast vector elements of type S to type T
	template<int N, class S, class T>
#ifdef TUM3D_VEC_STRICT
	Vec<N,T> &castVec(const Vec<N,S> &v, Vec<N,T> &res)
#else
	T *castVec(const S *v, T *res)
#endif
	{
		for(int k=0; k<N; ++k) 
			res[k] = T(v[k]);
		return res;
	}

	/// Element-wise minimum
	template<int N, class T>
#ifdef TUM3D_VEC_STRICT
	Vec<N,T> &minimum(Vec<N,T> &res, const Vec<N,T> &v)
#else
	T *minimum(T *res, const T *v)
#endif
	{
		for(int k=0; k<N; ++k)
			if (v[k]<res[k]) res[k] = v[k];
		return res;
	}

	/// Element-wise maximum
	template<int N, class T>
#ifdef TUM3D_VEC_STRICT
	Vec<N,T> &maximum(Vec<N,T> &res, const Vec<N,T> &v)
#else
	T *maximum(T *res, const T *v)
#endif
	{
		for(int k=0; k<N; ++k)
			if (v[k]>res[k]) res[k] = v[k];
		return res;
	}

	/// Creates a 2x2-matrix based on the elements given in a row-wise order. Result is stored in m and its reference is returned.
	template<class T>
	Mat<2,2,T> &mat2x2(const T &v00, const T &v01, const T &v10, const T &v11, Mat<2,2,T> &m)
	{
		m.get(0,0) = v00;
		m.get(0,1) = v01;
		m.get(1,0) = v10;
		m.get(1,1) = v11;
		return m;
	}

	/// Creates a 2x2-matrix based on the two column vectors. Result is stored in m and its reference is returned.
	template<class T>
	Mat<2,2,T> &mat2x2(const Vec<2,T> &v0, const Vec<2,T> &v1, Mat<2,2,T> &m)
	{
		for(int k=0; k<2; ++k){
			m.get(k,0) = v0[k];
			m.get(k,1) = v1[k];
			}
		return m;
	}

	/// Creates a 3x3-matrix based on the elements given in a row-wise order. Result is stored in m and its reference is returned.
	template<class T>
	Mat<3,3,T> &mat3x3(const T &v00, const T &v01, const T &v02, const T &v10, const T &v11, const T &v12, const T &v20, const T &v21, const T &v22, Mat<3,3,T> &m)
	{
		m.get(0,0) = v00;
		m.get(0,1) = v01;
		m.get(0,2) = v02;
		m.get(1,0) = v10;
		m.get(1,1) = v11;
		m.get(1,2) = v12;
		m.get(2,0) = v20;
		m.get(2,1) = v21;
		m.get(2,2) = v22;
		return m;
	}

	/// Creates a 3x3-matrix based on the three column vectors. Result is stored in m and its reference is returned.
	template<class T>
	Mat<3,3,T> &mat3x3(const Vec<3,T> &v0, const Vec<3,T> &v1, const Vec<3,T> &v2, Mat<3,3,T> &m)
	{
		for(int k=0; k<3; ++k){
			m.get(k,0) = v0[k];
			m.get(k,1) = v1[k];
			m.get(k,2) = v2[k];
			}
		return m;
	}

	/// Creates a 3x3-matrix based on the elements given in a row-wise order. Result is stored in m and its reference is returned.
	template<class T>
	Mat<4,4,T> &mat4x4(
			const T &v00, const T &v01, const T &v02, const T &v03,
			const T &v10, const T &v11, const T &v12, const T &v13,
			const T &v20, const T &v21, const T &v22, const T &v23,
			const T &v30, const T &v31, const T &v32, const T &v33,
			Mat<4,4,T> &m
		)
	{
		m.get(0,0) = v00;
		m.get(0,1) = v01;
		m.get(0,2) = v02;
		m.get(0,3) = v03;

		m.get(1,0) = v10;
		m.get(1,1) = v11;
		m.get(1,2) = v12;
		m.get(1,3) = v13;

		m.get(2,0) = v20;
		m.get(2,1) = v21;
		m.get(2,2) = v22;
		m.get(2,3) = v23;

		m.get(3,0) = v30;
		m.get(3,1) = v31;
		m.get(3,2) = v32;
		m.get(3,3) = v33;

		return m;
	}


	/// Creates a 4x4-matrix based on the four column vectors. Result is stored in m and its reference is returned.
	template<class T>
	Mat<4,4,T> &mat4x4(const Vec<4,T> &v0, const Vec<4,T> &v1, const Vec<4,T> &v2, const Vec<4,T> &v3, Mat<4,4,T> &m)
	{
		for(int k=0; k<4; ++k){
			m.get(k,0) = v0[k];
			m.get(k,1) = v1[k];
			m.get(k,2) = v2[k];
			m.get(k,3) = v3[k];
			}
		return m;
	}
	
	
	/// Creates a 4x4-matrix based on a 3x3 matrix. Result is stored in m and its reference is returned.
	template<class T>
	Mat<4,4,T> &homoMat4x4(const Mat<3,3,T> &mIn, Mat<4,4,T> &m)
	{
		for(int k=0; k<3; ++k){
			for(int l=0; l<3; ++l){
				m.get(k,l) = mIn.get(k,l);
				}
			m.get(k,3) = T(0);
			m.get(3,k) = T(0);
			}
		m.get(3,3) = T(1);
		return m;
	}
	
	
	/// Creates a NxN diagonal matrix based on the vector. Result is stored in m and its reference is returned.
	template<int N, class T>
	Mat<N,N,T> &diagMat(const Vec<N,T> &v, Mat<N,N,T> &m)
	{
		m.clear();
		for(int k=0; k<N; ++k)
			m.get(k,k) = v[k];
		return m;
	}

	/// Returns a vector containing the diagonal entries of a NxN matrix. The result is stored in v, and a reference to v is returned.
	template<int N,class T>
	Vec<N,T>& diagonal(const Mat<N,N,T>& A,Vec<N,T>& v) {
		for (int i=0; i<N; i++) v[i]=A.get(i,i);
		return v;
	}

	template <int N, class T>
	Mat<N,N,T> &identityMat(Mat<N,N,T> &m)
	{
		m.clear();
		for(int k=0; k<N; ++k)
			m.get(k,k) = T(1);
		return m;
	}


	/// Cast matrix elements of type S to type T
	template<int M, int N, class S, class T>
#ifdef TUM3D_VEC_STRICT
	Mat<M,N,T> &castMat(const Mat<M,N,S> &m, Mat<M,N,T> &res)
#else
	T *castMat(const S *m, T *res)
#endif
	{
		for(int k=0; k<M*N; ++k) 
			res[k] = T(m[k]);
		return res;
	}


	/// Invert a given 2x2 matrix. Result is stored in Im.
	template<class T>
#ifdef TUM3D_MAT_STRICT
	void invert2x2(const Mat<2,2,T> &A, Mat<2,2,T> &Im)
	{
		T *I = Im;
#else
	void invert2x2(const T *A, T *I)
	{
#endif
		T det = T(1.0)/(A[0]*A[3] - A[1]*A[2]);
		*I++ = det*A[3];
		*I++ = -det*A[1];
		*I++ = -det*A[2];
		*I   = det*A[0];
		I -= 3;
	}
	
	/// Invert a given 3x3 matrix. Result is stored in Im.
	template<class T>
#ifdef TUM3D_MAT_STRICT
	void invert3x3(const Mat<3,3,T> &A, Mat<3,3,T> &Im)
	{
		T *I = Im;
#else
	void invert3x3(const T *A, T *I)
	{
#endif

		// Calculate Inverse
		T det(0);
		det += A[0] * (*I++ =  A[4]*A[8] - A[5]*A[7]);
		det += A[3] * (*I++ = -A[1]*A[8] + A[2]*A[7]);
		det += A[6] * (*I++ =  A[1]*A[5] - A[2]*A[4]);
		det = 1/det;
		*I++ = (-A[3]*A[8] + A[5]*A[6]) * det;
		*I++ = ( A[0]*A[8] - A[2]*A[6]) * det;
		*I++ = (-A[0]*A[5] + A[2]*A[3]) * det;
		*I++ = ( A[3]*A[7] - A[4]*A[6]) * det;
		*I++ = (-A[0]*A[7] + A[1]*A[6]) * det;
		*I   = ( A[0]*A[4] - A[1]*A[3]) * det;
		I-=6;
		*I-- *= det;
		*I-- *= det;
		*I   *= det;
	}
	
	/// Invert a given 4x4 matrix. Result is stored in Im.
	template<class T>
#ifdef TUM3D_MAT_STRICT
	void invert4x4(const Mat<4,4,T> &A, Mat<4,4,T> &Im)
	{
		T *I = Im;
#else
	void invert4x4(const T *A, T *I)
	{
#endif
		T det(0);
		T tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;
		
		// Precalc 2x2 dets
		tmp0 = A[2] * A[7]  - A[3] * A[6];
		tmp1 = A[2] * A[11] - A[3] * A[10];
		tmp2 = A[2] * A[15] - A[3] * A[14];
		tmp3 = A[6] * A[11] - A[7] * A[10];
		tmp4 = A[6] * A[15] - A[7] * A[14];
		tmp5 = A[10]* A[15] - A[11]* A[14];	
		
		// First Row
		T d0 = A[5] * tmp5 - A[9] * tmp4 + A[13] * tmp3;
		T d1 = A[1] * tmp5 - A[9] * tmp2 + A[13] * tmp1;
		T d2 = A[1] * tmp4 - A[5] * tmp2 + A[13] * tmp0;
		T d3 = A[1] * tmp3 - A[5] * tmp1 + A[9]  * tmp0;
		det = T(1) / (A[0] * d0  -  A[4] * d1  +  A[8] * d2  -  A[12] * d3);	
		*I++ =  d0 * det;
		*I++ = -d1 * det;
		*I++ =  d2 * det;
		*I++ = -d3 * det;
		
		// Second Row
		*I++ = -det * (A[4] * tmp5 - A[8] * tmp4 + A[12] * tmp3);
		*I++ =  det * (A[0] * tmp5 - A[8] * tmp2 + A[12] * tmp1);
		*I++ = -det * (A[0] * tmp4 - A[4] * tmp2 + A[12] * tmp0);
		*I++ =  det * (A[0] * tmp3 - A[4] * tmp1 + A[8]  * tmp0);
		
		// Precalc 2x2 dets
		tmp0 = A[0] * A[5]  - A[1] * A[4];
		tmp1 = A[0] * A[9]  - A[1] * A[8];
		tmp2 = A[0] * A[13] - A[1] * A[12];
		tmp3 = A[4] * A[9]  - A[5] * A[8];
		tmp4 = A[4] * A[13] - A[5] * A[12];
		tmp5 = A[8] * A[13] - A[9] * A[12];	
		
		// Third Row
		*I++ =  det * (A[7] * tmp5 - A[11] * tmp4 + A[15] * tmp3);
		*I++ = -det * (A[3] * tmp5 - A[11] * tmp2 + A[15] * tmp1);
		*I++ =  det * (A[3] * tmp4 - A[7]  * tmp2 + A[15] * tmp0);
		*I++ = -det * (A[3] * tmp3 - A[7]  * tmp1 + A[11] * tmp0);
		
		// Fourth Row
		*I++ = -det * (A[6] * tmp5 - A[10] * tmp4 + A[14] * tmp3);
		*I++ =  det * (A[2] * tmp5 - A[10] * tmp2 + A[14] * tmp1);
		*I++ = -det * (A[2] * tmp4 - A[6]  * tmp2 + A[14] * tmp0);
		*I++ =  det * (A[2] * tmp3 - A[6]  * tmp1 + A[10] * tmp0);
		I -= 16;
	}

	/// Compute the determinant of the given 2x2 matrix
	template<class T>
#ifdef TUM3D_MAT_STRICT
	T determinant2x2(const Mat<2,2,T> &A)
#else
	T determinant2x2(const T *A)
#endif
	{
		return (A[0]*A[3] - A[1]*A[2]);
	}


	/// Compute the determinant of the given 3x3 matrix
	template<class T>
#ifdef TUM3D_MAT_STRICT
	T determinant3x3(const Mat<3,3,T> &A)
#else	
	T determinant3x3(const T *A)
#endif
	{
		return (A[0]*A[4]*A[8] + A[3]*A[7]*A[2] + A[6]*A[1]*A[5] - A[0]*A[7]*A[5] - A[3] *A[1] *A[8] - A[6]*A[4]*A[2]);
	}
	

	/// BEGIN DETERMINANT:  in case of questions mailto:jens.schneider@in.tum.de
	/// Compute the determinant of a given square matrix
	template<int N,class T>
	T determinant(const Mat<N,N,T>& A) {
		// Development wrt. last row
		T cof = ((N%2)==0 ? T(-1.0) : T(1.0));
		T det = static_cast<T>(0);
		Mat<N-1,N-1,T> B;
		for (int col=0; col<N; col++) {				
			for (int c=0; c<col; c++) {
				for (int r=0; r<N-1; r++) {
					B.get(c,r) = A.get(c,r);
				}
			}
			for (int c=col+1; c<N; c++) {
				for (int r=0; r<N-1; r++) {
					B.get(c-1,r) = A.get(c,r);
				}
			}
			det += cof*determinant(B);
			cof *= T(-1.0);		
		}
		return det;
	}

	template<class T>
	T determinant(const Mat<1,1,T>& A) {
		return A.get(0,0);
	}

	template<class T>
	T determinant(const Mat<2,2,T>& A) {
		return (A[0]*A[3] - A[1]*A[2]);
	}

	template<class T>
	T determinant(const Mat<3,3,T>& A)	{
		return (A[0]*A[4]*A[8] + A[3]*A[7]*A[2] + A[6]*A[1]*A[5] - A[0]*A[7]*A[5] - A[3] *A[1] *A[8] - A[6]*A[4]*A[2]);
	}
	/// END DETERMINANT: in case of questions mailto:jens.schneider@in.tum.de


	/// Convert a 3x3 rotation matrix to a quaternion w=quat[0], (x,y,z) = (quat[1], quat[2], quat[3])
	template<class T>
#ifdef TUM3D_MAT_STRICT
	void convertRotMatToQuaternion(const Mat<3,3,T> &rot, Vec<4,T> &quat)
#else
	void convertRotMatToQuaternion(const T *rot, T *quat)
#endif
	{
		quat[0] = T(0.5) * sqrt(rot[0] + rot[4] + rot[8] + T(1.0));
		if (quat[0] > T(0.5)){
			quat[1] = (rot[7] - rot[5]) / (T(4.0) * quat[0]);
			quat[2] = (rot[2] - rot[6]) / (T(4.0) * quat[0]);
			quat[3] = (rot[3] - rot[1]) / (T(4.0) * quat[0]);
			}
		else{
			if ( rot[0] >= rot[4] && rot[0] >= rot[8]){
				quat[1] = T(0.5) * sqrt(rot[0] - rot[4] - rot[8] + T(1.0));
				quat[0] = (rot[7] - rot[5]) / (T(4.0) * quat[1]);
				quat[2] = (rot[1] + rot[3]) / (T(4.0) * quat[1]);
				quat[3] = (rot[6] + rot[2]) / (T(4.0) * quat[1]);
				}
			else if ( rot[4] >= rot[0] && rot[4] >= rot[8]){
				quat[2] = T(0.5) * sqrt(rot[4] - rot[0] - rot[8] + T(1.0));
				quat[0] = (rot[2] - rot[6]) / (T(4.0) * quat[2]);
				quat[1] = (rot[1] + rot[3]) / (T(4.0) * quat[2]);
				quat[3] = (rot[5] + rot[7]) / (T(4.0) * quat[2]);
				}
			else if ( rot[8] >= rot[0] && rot[8] >= rot[4]){
				quat[3] = T(0.5) * sqrt(rot[8] - rot[0] - rot[4] + T(1.0));
				quat[0] = (rot[3] - rot[1]) / (T(4.0) * quat[3]);
				quat[1] = (rot[2] + rot[6]) / (T(4.0) * quat[3]);
				quat[2] = (rot[5] + rot[7]) / (T(4.0) * quat[3]);
				}
			}
	}
	
	/// Convert a 2x2 rotation matrix to a quaternion w=quat[0], (x,y,z) = (quat[1], quat[2], quat[3])
	template<class T>
#ifdef TUM3D_MAT_STRICT
	void convert2DRotMatToQuaternion(const Mat<2,2,T> &rot, Vec<4,T> &quat)
#else
	void convert2DRotMatToQuaternion(const T *rot, T *quat)
#endif
	{
		quat[0] = T(0.5) * sqrt(rot[0] + rot[3] + T(1.0) + T(1.0));
		if (quat[0] > T(0.5)){
			quat[1] = T(0.0);
			quat[2] = T(0.0);
			quat[3] = (rot[2] - rot[1]) / (4.0 * quat[0]);
			}
		else{
			quat[3] = T(0.5) * sqrt(T(1.0) - rot[0] - rot[3] + T(1.0));
			quat[0] = (rot[2] - rot[1]) / (T(4.0) * quat[3]);
			quat[1] = T(0.0);
			quat[2] = T(0.0);
			}
	}
	
	
	/// Convert unit quaternion (w,x,y,z) to 3x3 rotation matrix
	template<class T>
#ifdef TUM3D_MAT_STRICT
	void convertQuaternionToRotMat(const Vec<4,T> &quat, Mat<3,3,T> &rot)
#else
	void convertQuaternionToRotMat(const T *quat, T *rot)
#endif	
	{
		rot[0] = T(1.0) - T(2.0)*(quat[2]*quat[2] + quat[3]*quat[3]);
		rot[1] = T(-2.0)*quat[0]*quat[3] + T(2.0)*quat[1]*quat[2];
		rot[2] = T(2.0)*quat[0]*quat[2] + T(2.0)*quat[1]*quat[3];
		rot[3] = T(2.0)*quat[0]*quat[3] + T(2.0)*quat[1]*quat[2];
		rot[4] = T(1.0) - T(2.0)*(quat[1]*quat[1] + quat[3]*quat[3]);
		rot[5] = T(-2.0)*quat[0]*quat[1] + T(2.0)*quat[2]*quat[3];
		rot[6] = T(-2.0)*quat[0]*quat[2] + T(2.0)*quat[1]*quat[3];
		rot[7] = T(2.0)*quat[0]*quat[1] + T(2.0)*quat[2]*quat[3];
		rot[8] = T(1.0) - T(2.0)*(quat[1]*quat[1] + quat[2]*quat[2]);
	}

	/// Convert unit quaternion (w,x,y,z) to 4x4 rotation matrix
	template<class T>
	void convertQuaternionToRotMat(const Vec<4,T> &quat, Mat<4,4,T> &rot)
	{
		rot[0] = T(1.0) - T(2.0)*(quat[2]*quat[2] + quat[3]*quat[3]);
		rot[1] = T(-2.0)*quat[0]*quat[3] + T(2.0)*quat[1]*quat[2];
		rot[2] = T(2.0)*quat[0]*quat[2] + T(2.0)*quat[1]*quat[3];
		rot[3] = T(0.0);
		rot[4] = T(2.0)*quat[0]*quat[3] + T(2.0)*quat[1]*quat[2];
		rot[5] = T(1.0) - T(2.0)*(quat[1]*quat[1] + quat[3]*quat[3]);
		rot[6] = T(-2.0)*quat[0]*quat[1] + T(2.0)*quat[2]*quat[3];
		rot[7] = T(0.0);
		rot[8] = T(-2.0)*quat[0]*quat[2] + T(2.0)*quat[1]*quat[3];
		rot[9] = T(2.0)*quat[0]*quat[1] + T(2.0)*quat[2]*quat[3];
		rot[10] = T(1.0) - T(2.0)*(quat[1]*quat[1] + quat[2]*quat[2]);
		rot[11] = T(0.0);
		rot[12] = T(0.0);
		rot[13] = T(0.0);
		rot[14] = T(0.0);
		rot[15] = T(1.0);
	}
	
	/// Convert unit quaternion (w,x,y,z) to 2x2 rotation matrix
	template<class T>
#ifdef TUM3D_MAT_STRICT
	void convertQuaternionTo2DRotMat(const Vec<4,T> &quat, Mat<2,2,T> &rot)
#else
	void convertQuaternionTo2DRotMat(const T *quat, T *rot)
#endif	
	{
		rot[0] = T(1.0) - T(2.0)*(quat[2]*quat[2] + quat[3]*quat[3]);
		rot[1] = T(-2.0)*quat[0]*quat[3] + T(2.0)*quat[1]*quat[2];
		rot[2] = T(2.0)*quat[0]*quat[3] + T(2.0)*quat[1]*quat[2];
		rot[3] = T(1.0) - T(2.0)*(quat[1]*quat[1] + quat[3]*quat[3]);
	}
	

	/// Multiply two quaternions
	template<class T>
#ifdef TUM3D_MAT_STRICT
	void multQuaternion(const Vec<4,T> &x, const Vec<4,T> &y, Vec<4,T> &res)
#else
	void multQuaternion(const T *x, const T *y, T *res)
#endif
	{
		res[0] = x[0]*y[0] - x[1]*y[1] - x[2]*y[2] - x[3]*y[3];
		res[1] = x[0]*y[1] + x[1]*y[0] + x[2]*y[3] - x[3]*y[2];
		res[2] = x[0]*y[2] - x[1]*y[3] + x[2]*y[0] + x[3]*y[1];
		res[3] = x[0]*y[3] + x[1]*y[2] - x[2]*y[1] + x[3]*y[0];
	}

	
	/// Rotate vector x by unit quaternion q, store rotated x in res
	template<class T>
#ifdef TUM3D_MAT_STRICT
	void rotateVecByQuaternion(const Vec<3,T> &x, const Vec<4,T> &q, Vec<3,T> &res)
#else
	void rotateVecByQuaternion(const T *x, const T *q, T *res)
#endif
	{
		T q0s = q[0]*q[0];
		T q1s = q[1]*q[1];
		T q2s = q[2]*q[2];
		T q3s = q[3]*q[3];
	
		res[0] = (q0s + q1s - q2s - q3s) * x[0] + 
					T(2.0) * (q[1]*q[2] - q[0]*q[3]) * x[1] + 
					T(2.0) * (q[0]*q[2] + q[1]*q[3]) * x[2] ;
				
		res[1] = T(2.0) * (q[1]*q[2] + q[0]*q[3]) * x[0] +
					(q0s - q1s + q2s - q3s) * x[1]  + 
					T(2.0) * (-q[0]*q[1] + q[2]*q[3]) * x[2];
				
		res[2] = T(2.0) * (-q[0]*q[2] + q[1]* q[3]) * x[0] + 
					T(2.0) * ( q[0]*q[1] + q[2]* q[3]) * x[1] + 
					(q0s - q1s -q2s +q3s) * x[2];
	}

	/// Slerp: spherical linear interpolation of two unit quaternions
	template<int N, class T>
#ifdef TUM3D_VEC_STRICT
	Vec<N,T> &slerpQuaternion(T weight, const Vec<N,T> &v0, const Vec<N,T> &v1, Vec<N,T> &res)
#else
	T *slerpQuaternion(T weight, const T *v0, const T *v1, T *res)
#endif
	{
		T angle = dotProd<N,T>(v0, v1);
		T flipSign = T(1.0);
		if (angle < T(0.0)) flipSign = T(-1.0);
		angle = acos(flipSign*angle);
		
		T s = (angle<T(0.01)) ? T(1.0) : sin(angle);
		T s1 = (angle<T(0.01)) ? (T(1.0) - weight) : sin((T(1.0) - weight)*angle);
		T s2 = ((angle<T(0.01)) ? weight : sin(weight*angle)) * flipSign;
		for(int k=0; k<N; ++k) 
			res[k] = (s1*v0[k] + s2*v1[k]) / s;
		return res;
	}

	// Returns a quaternion corresponding to a rotation about the specified axis (which is expected to be a unit vector)
	template<class T>
#ifdef TUM3D_VEC_STRICT
	Vec<4,T>& rotationQuaternion(T angle, const Vec<3,T>& axis, Vec<4,T>& res)
#else
	Vec<4,T>& rotationQuaternion(T angle, const T *axis, Vec<4,T>& res)
#endif
	{
		res = Vec4<T>(cos(angle/T(2.0)), axis * sin(angle/T(2.0)));
		return res;
	}

	// Transformations

	// The following projection matrices map the depth range [-zNear,-zFar] to [0,1]

	template <class T>
	Mat<4,4,T> &perspectiveProjMatD3D(T width, T height, T zNear, T zFar, Mat<4,4,T> &m)
	{
		m.clear();
		m.get(0,0) = T(2.0) * zNear / width;
		m.get(1,1) = T(2.0) * zNear / height;
		m.get(2,2) = zFar / (zNear - zFar);
		m.get(2,3) = zNear * zFar / (zNear - zFar);
		m.get(3,2) = T(-1.0);

		return m;
	}


	// from RTR
	template <class T>
	Mat<4,4,T> &perspectiveOffCenterProjMatGL(T left, T right, T bottom, T top, T zNear, T zFar, Mat<4,4,T> &m)
	{
		m.clear();
		m.get(0,0) = T(2.0) * zNear / (right - left);
		m.get(2,0) = (left + right) / (right - left);
		m.get(1,1) = T(2.0) * zNear / (top - bottom);
		m.get(2,1) = (bottom + top) / (top - bottom);
		m.get(2,2) = (zFar + zNear) / (zNear - zFar);
		m.get(2,3) = T(2.0) * zNear * zFar / (zNear - zFar);
		m.get(3,2) = T(-1.0);

		return m;
	}

	template <class T>
	Mat<4,4,T> &perspectiveOffCenterProjMatD3D(T left, T right, T bottom, T top, T zNear, T zFar, Mat<4,4,T> &m)
	{
		m.clear();
		m.get(0,0) = T(2.0) * zNear / (right - left);
		m.get(0,2) = (left + right) / (right - left);
		m.get(1,1) = T(2.0) * zNear / (top - bottom);
		m.get(1,2) = (bottom + top) / (top - bottom);
		m.get(2,2) = zFar / (zNear - zFar);
		m.get(2,3) = zNear * zFar / (zNear - zFar);
		m.get(3,2) = T(-1.0);

		return m;
	}

	template <class T>
	Mat<4,4,T> &perspectiveFovxProjMatD3D(T fovx, T aspect, T zNear, T zFar, Mat<4,4,T> &m)
	{
		T xScale = T(1.0) / tan(fovx * T(0.5));
		T yScale = xScale * aspect;
		m.clear();
		m.get(0,0) = xScale;
		m.get(1,1) = yScale;
		m.get(2,2) = zFar / (zNear - zFar);
		m.get(2,3) = zNear * zFar / (zNear - zFar);
		m.get(3,2) = T(-1.0);

		return m;
	}


	template <class T>
	Mat<4,4,T> &perspectiveFovyProjMatD3D(T fovy, T aspect, T zNear, T zFar, Mat<4,4,T> &m)
	{
		T yScale = T(1.0) / tan(fovy * T(0.5));
		T xScale = yScale / aspect;
		m.clear();
		m.get(0,0) = xScale;
		m.get(1,1) = yScale;
		m.get(2,2) = zFar / (zNear - zFar);
		m.get(2,3) = zNear * zFar / (zNear - zFar);
		m.get(3,2) = T(-1.0);

		return m;
	}


	template <class T>
	Mat<4,4,T> &orthoProjMatD3D(T width, T height, T zNear, T zFar, Mat<4,4,T> &m)
	{
		m.clear();
		m.get(0,0) = T(2.0) / width;
		m.get(1,1) = T(2.0) / height;
		m.get(2,2) = T(1.0) / (zNear - zFar);
		m.get(2,3) = zNear / (zNear - zFar);
		m.get(3,3) = T(1.0);

		return m;
	}


	template <class T>
	Mat<4,4,T> &orthoOffCenterProjMatD3D(T left, T right, T bottom, T top, T zNear, T zFar, Mat<4,4,T> &m)
	{
		m.clear();
		m.get(0,0) = T(2.0) / (right - left);
		m.get(0,3) = (left + right) / (left - right);
		m.get(1,1) = T(2.0) / (top - bottom);
		m.get(1,3) = (bottom + top) / (bottom - top);
		m.get(2,2) = T(1.0) / (zNear - zFar);
		m.get(2,3) = zNear / (zNear - zFar);
		m.get(3,3) = T(1.0);

		return m;
	}

	template <class T>
	Mat<4,4,T> &translationMat(T x, T y, T z, Mat<4,4,T> &m)
	{
		m.clear();
		m.get(0,0) = T(1.0);
		m.get(1,1) = T(1.0);
		m.get(2,2) = T(1.0);
		m.get(3,3) = T(1.0);
		m.get(0,3) = x;
		m.get(1,3) = y;
		m.get(2,3) = z;
		return m;
	}

	template <class T>
	Mat<4,4,T> &translationMat(const Vec<3,T> &v, Mat<4,4,T> &m)
	{
		m.clear();
		m.get(0,0) = T(1.0);
		m.get(1,1) = T(1.0);
		m.get(2,2) = T(1.0);
		m.get(3,3) = T(1.0);
		m.get(0,3) = v[0];
		m.get(1,3) = v[1];
		m.get(2,3) = v[2];
		return m;
	}

	template <class T>
	Mat<4,4,T> &scalingMat(T xScale, T yScale, T zScale, Mat<4,4,T> &m)
	{
		m.clear();
		m.get(0,0) = xScale;
		m.get(1,1) = yScale;
		m.get(2,2) = zScale;
		m.get(3,3) = T(1.0);

		return m;
	}

	template <class T>
	Mat<4,4,T> &scalingMat(const Vec<3,T> &v, Mat<4,4,T> &m)
	{
		m.clear();
		m.get(0,0) = v[0];
		m.get(1,1) = v[1];
		m.get(2,2) = v[2];
		m.get(3,3) = T(1.0);

		return m;
	}

	/// Creates a 3x3 rotation matrix based on the angle with respect to x-axis. Result is stored in m and its reference is returned.
	template<class T>
	Mat<3,3,T> &rotationXMat(T angle, Mat<3,3,T> &m) 
	{
		m.clear();
		T dCosAngle = cos(angle);
		T dSinAngle = sin(angle);

		m.get(0,0) = T(1.0);
		m.get(1,1) = dCosAngle;	 m.get(1,2) = -dSinAngle;
		m.get(2,1) = dSinAngle;	 m.get(2,2) = dCosAngle;
		return m;
	}


	/// Creates a 3x3 rotation matrix based on the angle with respect to y-axis. Result is stored in m and its reference is returned.
	template<class T>
	Mat<3,3,T> &rotationYMat(T angle, Mat<3,3,T> &m) 
	{
		m.clear();
		T dCosAngle = cos(angle);
		T dSinAngle = sin(angle);

		m.get(1,1) = T(1.0);
		m.get(0,0) = dCosAngle;	 m.get(0,2) = dSinAngle;
		m.get(2,0) =-dSinAngle;	 m.get(2,2) = dCosAngle;
		return m;
	}


	/// Creates a 3x3 rotation matrix based on the angle with respect to z-axis. Result is stored in m and its reference is returned.
	template<class T>
	Mat<3,3,T> &rotationZMat(T angle, Mat<3,3,T> &m) 
	{
		m.clear();
		T dCosAngle = cos(angle);
		T dSinAngle = sin(angle);

		m.get(2,2) = T(1.0);
		m.get(0,0) = dCosAngle;	 m.get(0,1) = -dSinAngle;
		m.get(1,0) = dSinAngle;	 m.get(1,1) = dCosAngle;
		return m;
	}


	/// Creates a 3x3 rotation matrix based on the angle with respect to the given axis. Result is stored in m and its reference is returned.
	template<class T>
	Mat<3,3,T> &rotationMat(const Vec<3,T> &axis, T angle, Mat<3,3,T> &m) 
	{
		T dCosAngle = cos(angle);
		T dSinAngle = sin(angle);
		T dOneMinusCosAngle = T(1)-dCosAngle;

		m.get(0,0) = dCosAngle+dOneMinusCosAngle*axis[0]*axis[0];				
		m.get(0,1) = dOneMinusCosAngle*axis[0]*axis[1]-dSinAngle*axis[2];	
		m.get(0,2) = dOneMinusCosAngle*axis[0]*axis[2]+dSinAngle*axis[1];
		m.get(1,0) = dOneMinusCosAngle*axis[0]*axis[1]+dSinAngle*axis[2];		
		m.get(1,1) = dCosAngle+dOneMinusCosAngle*axis[1]*axis[1];			
		m.get(1,2) = dOneMinusCosAngle*axis[1]*axis[2]-dSinAngle*axis[0];
		m.get(2,0) = dOneMinusCosAngle*axis[0]*axis[2]-dSinAngle*axis[1];		
		m.get(2,1) = dOneMinusCosAngle*axis[1]*axis[2]+dSinAngle*axis[0];	
		m.get(2,2) = dCosAngle+dOneMinusCosAngle*axis[2]*axis[2];
		
		return m;
	}

	template <class T>
	Mat<4,4,T> &rotationXMat(T angle, Mat<4,4,T> &m)
	{
		T cosa = cos(angle);
		T sina = sin(angle);
		m.clear();
		m.get(0,0) = T(1.0);
		m.get(1,1) = cosa;
		m.get(1,2) = -sina;
		m.get(2,1) = sina;
		m.get(2,2) = cosa;
		m.get(3,3) = T(1.0);

		return m;
	}


	template <class T>
	Mat<4,4,T> &rotationYMat(T angle, Mat<4,4,T> &m)
	{
		T cosa = cos(angle);
		T sina = sin(angle);
		m.clear();
		m.get(0,0) = cosa;
		m.get(0,2) = sina;
		m.get(1,1) = T(1.0);
		m.get(2,0) = -sina;
		m.get(2,2) = cosa;
		m.get(3,3) = T(1.0);

		return m;
	}


	template <class T>
	Mat<4,4,T> &rotationZMat(T angle, Mat<4,4,T> &m)
	{
		T cosa = cos(angle);
		T sina = sin(angle);
		m.clear();
		m.get(0,0) = cosa;
		m.get(0,1) = -sina;
		m.get(1,0) = sina;
		m.get(1,1) = cosa;
		m.get(2,2) = T(1.0);
		m.get(3,3) = T(1.0);

		return m;
	}

	template<class T>
	Mat<4,4,T> &rotationMat(const Vec<3,T> &axis, T angle, Mat<4,4,T> &m) 
	{
		T dCosAngle = cos(angle);
		T dSinAngle = sin(angle);
		T dOneMinusCosAngle = T(1)-dCosAngle;

		m.get(0,0) = dCosAngle+dOneMinusCosAngle*axis[0]*axis[0];				
		m.get(0,1) = dOneMinusCosAngle*axis[0]*axis[1]-dSinAngle*axis[2];	
		m.get(0,2) = dOneMinusCosAngle*axis[0]*axis[2]+dSinAngle*axis[1];
		m.get(0,3) = T(0.0);
		m.get(1,0) = dOneMinusCosAngle*axis[0]*axis[1]+dSinAngle*axis[2];		
		m.get(1,1) = dCosAngle+dOneMinusCosAngle*axis[1]*axis[1];			
		m.get(1,2) = dOneMinusCosAngle*axis[1]*axis[2]-dSinAngle*axis[0];
		m.get(1,3) = T(0.0);
		m.get(2,0) = dOneMinusCosAngle*axis[0]*axis[2]-dSinAngle*axis[1];		
		m.get(2,1) = dOneMinusCosAngle*axis[1]*axis[2]+dSinAngle*axis[0];	
		m.get(2,2) = dCosAngle+dOneMinusCosAngle*axis[2]*axis[2];
		m.get(2,3) = T(0.0);
		m.get(3,0) = T(0.0);
		m.get(3,1) = T(0.0);
		m.get(3,2) = T(0.0);
		m.get(3,3) = T(1.0);
		
		return m;
	}


	//////////////////// ANALYSIS FOR SQUARE MATRICES ////////////////////
	/// in case of questions mailto:jens.schneider@in.tum.de
	/// Jan.09 Jens Schneider

	/// MATRIX PROPERTIES
	#define MAT_UNDETERMINED					0xFFFFFFFF
	#define MAT_NONE							0x00000000
	#define MAT_ROW_DIAGONAL_DOMINANT			0x00000001
	#define MAT_ROW_STRICTLY_DIAGONAL_DOMINANT	0x00000003	// implies dominance
	#define MAT_COL_DIAGONAL_DOMINANT			0x00000004
	#define MAT_COL_STRICTLY_DIAGONAL_DOMINANT	0x0000000C	// implies dominance
	#define MAT_DIAGONAL_DOMINANT				0x00000005	// implies both col and row dominance
	#define MAT_STRICTLY_DIAGONAL_DOMINANT		0x0000000F	// implies both col and row strict dominance
	#define MAT_SYMMETRIC						0x00000010
	#define MAT_ANTISYMMETRIC					0x00000020
	#define MAT_POSITIVE_SEMIDEFINITE			0x00000100
	#define MAT_POSITIVE_DEFINITE				0x00008300	// implies semi-definiteness, non-singularity
	#define MAT_NEGATIVE_SEMIDEFINITE			0x00000400
	#define MAT_NEGATIVE_DEFINITE				0x00008C00	// implies semi-definiteness, non-singularity
	#define MAT_ORTHOGONAL						0x00001000
	#define MAT_ORTHONORMAL						0x00003000	// imples orthogonality
	#define MAT_ROTATORY						0x00007000	// implies orthonormality
	#define MAT_NONSINGULAR						0x00008000
	#define MAT_SPD								0x00008310	// implies symmetry, pos.def., and non-singularity

	/// MATRIX PROPERTY CLASS (mainly for writing to ostreams)
	class MatrixProperty {
		public:
			MatrixProperty(int i=MAT_NONE) : m_i(i) {}
			~MatrixProperty(void) {}
			operator int&(void)		 { return m_i; }
			operator int(void) const { return m_i; }
			bool is(int other) const { //< checks if other is implied by this property
				if (other==MAT_NONE) return m_i==other;
				else return (m_i&other)==other;
			}
			MatrixProperty operator| (int other) const {
				return MatrixProperty(m_i|other);
			}
			MatrixProperty operator& (int other) const {
				return MatrixProperty(m_i&other);
			}
			MatrixProperty& operator|=(int other) {
				m_i|=other;
				return *this;
			}
			MatrixProperty& operator &=(int other) {
				m_i&=other;
				return *this;
			}
		private:
			int m_i;
	};
/*
	/// OUTPUT A MATRIX PROPERTY
	std::ostream &operator<<(std::ostream &ost,const MatrixProperty& s) {
		bool bFirst=true;
		if (s.is(MAT_UNDETERMINED)) {
			ost << "undetermined";
			return ost;
		}
		if (s.is(MAT_NONE)) {
			ost << "general";
			return ost;
		}
		// DOMINANCE
		if (s.is(MAT_STRICTLY_DIAGONAL_DOMINANT)) {
			ost << (bFirst ? "" : ", ") << "strictly diagonal dominant";
			bFirst=false;
		}
		else {
			if (s.is(MAT_DIAGONAL_DOMINANT)) {
				ost << (bFirst ? "" : ", ") << "diagonal dominant";
				bFirst=false;
			}
			else {
				if (s.is(MAT_ROW_STRICTLY_DIAGONAL_DOMINANT)) {
					ost << (bFirst ? "" : ", ") << "stricly row diagonal dominant";
					bFirst=false;
				}
				else if (s.is(MAT_ROW_DIAGONAL_DOMINANT)) {
					ost << (bFirst ? "" : ", ") << "row diagonal dominant";
					bFirst=false;
				}
				if (s.is(MAT_COL_STRICTLY_DIAGONAL_DOMINANT)) {
					ost << (bFirst ? "" : ", ") << "stricly column diagonal dominant";
					bFirst=false;
				}
				else if (s.is(MAT_COL_DIAGONAL_DOMINANT)) {
					ost << (bFirst ? "" : ", ") << "column diagonal dominant";
					bFirst=false;
				}
			}
		}
		// SYMMETRY	
		if (s.is(MAT_SYMMETRIC)) {
			ost << (bFirst ? "" : ", ") << "symmetric";
			bFirst=false;
		}
		else if (s.is(MAT_ANTISYMMETRIC)) {
			ost << (bFirst ? "" : ", ") << "anti-symmetric";
			bFirst=false;
		}
		else {
			ost << (bFirst ? "" : ", ") << "non-symmetric";
			bFirst=false;
		}
		// DEFINITENESS
		if (s.is(MAT_POSITIVE_DEFINITE)) {
			ost << (bFirst ? "" : ", ") << "positive definite";
			bFirst=false;
		}
		else if (s.is(MAT_POSITIVE_SEMIDEFINITE)) {
			ost << (bFirst ? "" : ", ") << "positive semi-definite";
			bFirst=false;
		}
		else if (s.is(MAT_NEGATIVE_DEFINITE)) {
			ost << (bFirst ? "" : ", ") << "negative definite";
			bFirst=false;
		}
		else if (s.is(MAT_NEGATIVE_SEMIDEFINITE)) {
			ost << (bFirst ? "" : ", ") << "negative semi-definite";
			bFirst=false;
		}
		else {
			ost <<  (bFirst ? "" : ", ") << "indefinite";
			bFirst=false;
		}
		if (!s.is(MAT_POSITIVE_DEFINITE) && !s.is(MAT_NEGATIVE_DEFINITE) && s.is(MAT_NONSINGULAR)) {
			ost << (bFirst ? "" : ", ") << "non-singular";
			bFirst=false;
		}
		// OTHOGONALITY AND SUCH
		if (s.is(MAT_ROTATORY)) {
			ost << (bFirst ? "" : ", ") << "rotatory";
			bFirst=false;
		}
		else if (s.is(MAT_ORTHONORMAL)) {
			ost << (bFirst ? "" : ", ") << "orthonormal";
			bFirst=false;
		}
		else if (s.is(MAT_ORTHOGONAL))  {
			ost << (bFirst ? "" : ", ") << "orthogonal";
			bFirst=false;
		}
		return ost;
	}
*/
	/// SYMMETRY
	/// TIME:    O(N^2)
	/// SPACE:   O(N^2)
	/// OPTIMAL: Yes.
	/// STABLE:  Mind the epsilon!
	template<int M,int N,class T>
	MatrixProperty symmetry(const Mat<M,N,T>& A,T epsilon=static_cast<T>(1e-8)) {
		if (M!=N) return MatrixProperty(MAT_NONE);
		for (int i=0; i<N; i++) {
			for (int j=i+1; j<N; j++) {
				if (abs(A.get(i,j)-A.get(j,i))>epsilon) goto Test_AntiSym;
			}
		}
		return MatrixProperty(MAT_SYMMETRIC);
	Test_AntiSym:
		for (int i=0; i<N; i++) {
			for (int j=i+1; j<N; j++) {
				if (abs(A.get(i,j)+A.get(j,i))>epsilon) return MatrixProperty(MAT_NONE);
			}
		}
		return MatrixProperty(MAT_ANTISYMMETRIC);
	}

	/// ORTHOGONALITY
	/// TIME:    O(N!)	(due to the determinant)
	/// SPACE:   O(N^2)
	/// OPTIMAL: Probably not.
	/// STABLE:  Mind the epsilon!
	template<int N,class T>
	MatrixProperty orthogonality(const Mat<N,N,T>& A,T epsilon=static_cast<T>(1e-8), bool bHasDet=false, T det=static_cast<T>(0)) {
		MatrixProperty p;
		// Check row-vector lengths
		for (int r=0; r<N; r++) {
			if (abs(A.getRow(r).normSqr()-static_cast<T>(1))>epsilon) goto checkOrthogonality;
		}
		p |= MAT_ORTHONORMAL;
	checkOrthogonality:
		// Check row-vector orthogonality
		//bool bOrthogonal = true;
		for (int ra=0; ra<N; ra++) {
			for (int rb=ra+1; rb<N; rb++) {
				if (dotProd<N,T>(A.getRow(ra),A.getRow(rb))>epsilon) return MatrixProperty(MAT_NONE);
			}
		}
		p |= MAT_ORTHOGONAL;
		if (p.is(MAT_ORTHONORMAL)) {
			T DET = (bHasDet ? det : determinant(A));
			if (DET-static_cast<T>(1)<epsilon) return MatrixProperty(MAT_ROTATORY);
		}
		return p;
}


	/// DEFINITENESS OF A SQUARE MATRIX (Sylvester's criterion)
	
	/// computes the leading minors of a square matrix
	/// they are implicitly sorted with descending dimensionality
	template<int N,class T>
	Vec<N,T> leadingMinors(const Mat<N,N,T>& A) {
		Vec<N,T> result;
		DontCallThisYourself_LeadingMinorsAux(A,static_cast<T*>(result));
		return result;
	}
	
	/// Trust me, you don't want to call this yourself.
	template<int N,class T>
	void DontCallThisYourself_LeadingMinorsAux(const Mat<N,N,T>& A, T* dets) {
		*dets++=tum3D::determinant(A);
		Mat<N-1,N-1,T> B;
		for (int i=0; i<N-1; i++) {
			for (int j=0; j<N-1; j++) {
				B.get(i,j)=A.get(i,j);
			}
		}
		DontCallThisYourself_LeadingMinorsAux(B,dets);
	}
	/// Trust me, you don't want to call this yourself.
	template<class T>
	void DontCallThisYourself_LeadingMinorsAux(const Mat<1,1,T>& A, T* dets) {
		*dets=A.get(0,0);
	}
	
	/// Computes the definiteness of a square matrix and checks for singularity
	/// TIME:    O(N!)
	/// SPACE:   O(N^2)
	/// OPTIMAL: Are you kidding me?
	/// STABLE:  Not as such, no. No epsilon-test for semi-definiteness!
	/// TODO: SYLVESTRE'S CRITERION DOES NOT COVER SEMI-DEFINITENESS!
	/// To test for semi-sefiniteness, all principal minors have to be checked!
	/// TODO: Early out during leading principal minor computation ?
	template<int N,class T>
	MatrixProperty definiteness(const Mat<N,N,T>& A, T& Determinant) {
		// compute leading minors and sort -- Sylvester's criterion
		Vec<N,T> dets = leadingMinors(A);
		Determinant = dets[0];
		T cof=static_cast<T>((N%2)==0 ? 1 : -1);
		T L =dets[0];
		T Ln=cof*dets[0];
		for (int i=1; i<N; i++) {
			cof*=static_cast<T>(-1);
			L = std::min(L,dets[i]);
			Ln= std::min(Ln,cof*dets[i]);
		}
		const T zero=static_cast<T>(0);
		// check for positive definiteness and semi-definiteness
		if (L>zero)	 return MatrixProperty(MAT_POSITIVE_DEFINITE);
		if (Ln>zero) return MatrixProperty(MAT_NEGATIVE_DEFINITE);
		MatrixProperty p(dets[0]!=T(0.0) ? MAT_NONSINGULAR : MAT_NONE);
		//if (L==0)	 return (p|MAT_POSITIVE_SEMIDEFINITE);
		//if (Ln==0)	 return (p|MAT_NEGATIVE_SEMIDEFINITE);
		return p;
	}
	template<int N,class T>
	MatrixProperty definiteness(const Mat<N,N,T>& A) {
		T dummy;
		return definiteness(A,dummy);
	}
	
	/// DIAGONAL DOMINANCE
	/// TIME:    O(N^2)
	/// SPACE:   O(N^2)
	/// OPTIMAL: Yes.
	/// STABLE:  Hopefully yes.
	template<int N,class T>
	MatrixProperty diagonalDominance(const Mat<N,N,T>& A) {
		bool bStrict=true;
		MatrixProperty p;
		// check row dominance
		for (int col=0; col<N; col++) {
			T sum=static_cast<T>(0);
			for (int row=0; row<N; row++) sum+=abs(A.get(col,row));
			if (2*abs(A.get(col,col))<sum) goto colDominance;
			if (2*abs(A.get(col,col))==sum) bStrict=false;
		}
		p = (bStrict ? MAT_ROW_STRICTLY_DIAGONAL_DOMINANT : MAT_ROW_DIAGONAL_DOMINANT);
	colDominance:
		bStrict=false;
		for (int row=0; row<N; row++) {
			T sum=static_cast<T>(0);
			for (int col=0; col<N; col++) sum+=abs(A.get(col,row));
			if (2*abs(A.get(row,row))<sum) return p;
			if (2*abs(A.get(row,row))==sum) bStrict=false;
		}
		return p|(bStrict ? MAT_COL_STRICTLY_DIAGONAL_DOMINANT : MAT_COL_DIAGONAL_DOMINANT);
	}
	
	/// FULL ANALYSIS
	/// STILL IN PROGRESS!
	/// GOOD: Returns all properties.
	/// Mind the epsilon!
	template<int N,class T>
	MatrixProperty analysis(const Mat<N,N,T>& A,T epsilon=static_cast<T>(1e-8)) {
		T det;
		MatrixProperty p = diagonalDominance(A) | symmetry(A,epsilon) | definiteness(A,det);
		return p|orthogonality<N,T>(A,epsilon,true,det);
	}

	/// Jan.09 Jens Schneider
	/// in case of questions mailto:jens.schneider@in.tum.de
	//////////////////// ANALYSIS FOR SQUARE MATRICES ////////////////////

	/// output Vector to a stream
	template<int N, class T>
	std::ostream &operator<<(std::ostream &ost, const Vec<N,T> &v)
	{
	  for (int i = 0; i < N; i++)
	  {
		if (i)
		  ost << ' ';
		ost << v[i];
	  }
	  return ost;
	}

	/// input Vector from a stream
	template<int N, class T>
	std::istream &operator>>(std::istream &ist, Vec<N,T> &v)
	{
	  for (int i = 0; i < N; i++)
		ist >> v[i];
	  return ist;
	}

	/// output Matrix to a stream
	template<int M, int N, class T>
	std::ostream &operator<<(std::ostream &ost, const Mat<M,N,T> &m)
	{
	  for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
		{
			if (j)
			ost << ' ';
			ost << m[i*N+j];
		}
			ost << std::endl;
		}
	  return ost;
	}

	/// input Matrix from a stream
	template<int M, int N, class T>
	std::istream &operator>>(std::istream &ist, Mat<M,N,T> &m)
	{
	  for (int i = 0; i < M*N; i++)
		ist >> m[i];
	  return ist;
	}


	/** @defgroup vecMatTypedefs Vector and matrix typedefs
	 *  Some typedefs for the mostly used instances
	 *  @{
	 */
	typedef Vec2<unsigned char> Vec2uc; ///< two-dimensional unsigned char vector
	typedef Vec3<unsigned char> Vec3uc; ///< three-dimensional unsigned char vector
	typedef Vec4<unsigned char> Vec4uc; ///< four-dimensional unsigned char vector

	typedef Vec2<char> Vec2c; ///< two-dimensional char vector
	typedef Vec3<char> Vec3c; ///< three-dimensional char vector
	typedef Vec4<char> Vec4c; ///< four-dimensional char vector

	typedef Vec2<short int> Vec2s; ///< two-dimensional short int vector
	typedef Vec3<short int> Vec3s; ///< three-dimensional short int vector
	typedef Vec4<short int> Vec4s; ///< four-dimensional short int vector

	typedef Vec2<unsigned short int> Vec2us; ///< two-dimensional unsigned short int vector
	typedef Vec3<unsigned short int> Vec3us; ///< three-dimensional unsigned short int vector
	typedef Vec4<unsigned short int> Vec4us; ///< two-dimensional unsigned short int vector

	typedef Vec2<int> Vec2i; ///< two-dimensional in vector
	typedef Vec3<int> Vec3i; ///< three-dimensional in vector
	typedef Vec4<int> Vec4i; ///< four-dimensional in vector

	typedef Vec2<unsigned int> Vec2ui; ///< two-dimensional unsigned int vector
	typedef Vec3<unsigned int> Vec3ui; ///< three-dimensional unsigned int vector
	typedef Vec4<unsigned int> Vec4ui; ///< four-dimensional unsigned int vector

	typedef Vec2<float> Vec2f; ///< two-dimensional float vector
	typedef Vec3<float> Vec3f; ///< three-dimensional float vector
	typedef Vec4<float> Vec4f; ///< four-dimensional float vector

	typedef Vec2<double> Vec2d; ///< two-dimensional double vector
	typedef Vec3<double> Vec3d; ///< three-dimensional double vector
	typedef Vec4<double> Vec4d; ///< four-dimensional double vector


	typedef Mat<2,2,int> Mat2i; ///< 2x2 int matrix
	typedef Mat<3,3,int> Mat3i; ///< 3x3 int matrix
	typedef Mat<4,4,int> Mat4i; ///< 4x4 int matrix

	typedef Mat<2,2,float> Mat2f; ///< 2x2 float matrix
	typedef Mat<3,3,float> Mat3f; ///< 3x3 float matrix
	typedef Mat<4,4,float> Mat4f; ///< 4x4 float matrix

	typedef Mat<2,2,double> Mat2d; ///< 2x2 double matrix
	typedef Mat<3,3,double> Mat3d; ///< 3x3 double matrix
	typedef Mat<4,4,double> Mat4d; ///< 4x4 double matrix
	// @}

}
#endif
