#pragma once
#ifndef VECTORTEMPLATE_H
#define VECTORTEMPLATE_H

#include <initializer_list>
#include <assert.h>
#include <sstream>
#include <cmath>
#include <array>

#include "../Base/templateFor.h"

MATH_NAMESPACE_BEG

   /**
	* Template based mathematical vector implementation
	*/
	template<typename T, size_t N> struct vector_c : std::array<T, N>
	{
		using type = vector_c<T, N>;
		static const size_t value = N;

		vector_c() {}

		explicit vector_c(const T& val)
		{
			base::For<0, N>::Do([&](size_t idx) { (*this)[idx] = val; });
		}

		vector_c(const vector_c& v)
		{
			base::For<0, N>::Do([&](size_t idx) { (*this)[idx] = v[idx]; });
		}

		vector_c& operator=(const vector_c& v)
		{
			base::For<0, N>::Do([&](size_t idx) { (*this)[idx] = v[idx]; });
			return *this;
		}

		vector_c(std::initializer_list<T> list)
		{
			assert(list.size() == N);
			typename std::initializer_list<T>::const_iterator it = list.begin();
			base::For<0, N>::Do([&](size_t idx) { (*this)[idx] = *(it++); });
		}
	};

#define DEF_VECTOR_INLINE_TEMPLATE template<typename T, size_t N> inline

	/**
	* Vector and number addition
	*/
	DEF_VECTOR_INLINE_TEMPLATE vector_c<T, N>& operator += (vector_c<T, N>& v, const T& h)
	{
		base::For<0, N>::Do([&v, h](size_t idx) { v[idx] += h; });
		return v;
	}

	/**
	* Vector vector inplace addition
	*/
	DEF_VECTOR_INLINE_TEMPLATE vector_c<T, N>& operator += (vector_c<T, N>& vl, const vector_c<T, N>& vr)
	{
		base::For<0, N>::Do([&vl, vr](size_t idx) { vl[idx] += vr[idx]; });
		return vl;
	}

	/**
	* Vector vector addition
	*/
	DEF_VECTOR_INLINE_TEMPLATE vector_c<T, N> operator  + (const vector_c<T, N>& vl, const vector_c<T, N>& vr)
	{
		vector_c<T, N> res(vl); 
		return res += vr;
	}

	/**
	* Vector negation
	*/
	DEF_VECTOR_INLINE_TEMPLATE vector_c<T, N> operator - (const vector_c<T, N>& v)
	{
		vector_c<T, N> res;
		base::For<0, N>::Do([&res, v](size_t idx) { res[idx] = -v[idx]; });
		return res;
	}

	/**
	* Vector-number in place subtraction
	*/
	DEF_VECTOR_INLINE_TEMPLATE vector_c<T, N>& operator -= (vector_c<T, N>& v, const T& h)
	{
		base::For<0, N>::Do([&v, h](size_t idx) { v[idx] -= h; });
		return v;
	}

	/**
	* vector in place subtraction
	*/
	DEF_VECTOR_INLINE_TEMPLATE vector_c<T, N>& operator -= (vector_c<T, N>& vl, const vector_c<T, N>& vr)
	{
		base::For<0, N>::Do([&vl, vr](size_t idx) { vl[idx] -= vr[idx]; });
		return vl;
	}

	/**
	* vector subtraction
	*/
	DEF_VECTOR_INLINE_TEMPLATE vector_c<T, N> operator - (const vector_c<T, N>& vl, const vector_c<T, N>& vr)
	{
		vector_c<T, N> result(vl); 
		return result -= vr;
	}

	/**
	* Vector inplace multiplication by a number
	*/
	DEF_VECTOR_INLINE_TEMPLATE vector_c<T, N>& operator *= (vector_c<T, N>& v, const T& h)
	{
		base::For<0, N>::Do([&v, h](size_t idx) { v[idx] *= h; });;
		return v;
	}

	/**
	* Vector multiplication by a number left and right forms
	*/
	DEF_VECTOR_INLINE_TEMPLATE vector_c<T, N> operator * (const vector_c<T, N>& vl, const T& h)
	{
		vector_c<T, N> result(vl); 
		return result *= h;
	}
	DEF_VECTOR_INLINE_TEMPLATE vector_c<T, N> operator * (const T& h, const vector_c<T, N>& vl)
	{
		vector_c<T, N> result(vl); 
		return result *= h;
	}

	/**
	* Vector inplace division by a number
	*/
	DEF_VECTOR_INLINE_TEMPLATE vector_c<T, N>& operator /= (vector_c<T, N>& v, const T& h)
	{
		base::For<0, N>::Do([&v, h](size_t idx) { v[idx] /= h; });
		return v;
	}

	/**
	* Vector division by a number
	*/
	DEF_VECTOR_INLINE_TEMPLATE vector_c<T, N> operator / (const vector_c<T, N>& v, const T& h)
	{
		vector_c<T, N> result(v); 
		return result /= h;
	}

	/**
	* Vector dot multiplication
	*/
	DEF_VECTOR_INLINE_TEMPLATE T operator * (const vector_c<T, N>& vl, const vector_c<T, N>& vr)
	{
		T res = 0.0;
		base::For<0, N>::Do([&res, vl, vr](size_t idx) { res += (vl[idx] * vr[idx]); });
		return res;
	}

	/**
	* Vector printing
	*/
	DEF_VECTOR_INLINE_TEMPLATE std::ostream& operator << (std::ostream& out, const vector_c<T, N>& v)
	{
		out << "( ";
		base::For<0, N>::Do([&out, v](size_t idx) { out << v[idx] << " "; });
		out << ")";
		return out;
	}

	/**
	 * Input vector from an std::istringstream
	 */
	DEF_VECTOR_INLINE_TEMPLATE std::istream& operator >> (std::istream& in, vector_c<T, N>& v)
	{
		base::For<0, N>::Do([&](size_t idx) { in >> v[idx]; });
		return in;
	}

	/**
	 * Square length of a vector
	 */
	DEF_VECTOR_INLINE_TEMPLATE T sqr(const vector_c<T, N>& v) { return v*v; }
	DEF_VECTOR_INLINE_TEMPLATE T sqr(vector_c<T, N>&& v) { return v*v; }

	/**
	* Vector's euclidian length
	*/
	DEF_VECTOR_INLINE_TEMPLATE T abs(const vector_c<T, N>& v) { return ::sqrt(sqr(v)); }

	/**
	* Sum of all vector elements
	*/
	DEF_VECTOR_INLINE_TEMPLATE T sum(const vector_c<T, N>& v)
	{
		T res = 0.0;
		base::For<0, N>::Do([&res, v](size_t idx) { res += v[idx]; });
		return res;
	}

	/**
	* Product of all vector elements
	*/
	DEF_VECTOR_INLINE_TEMPLATE T prod(const vector_c<T, N>& v)
	{
		T res = 1.0;
		base::For<0, N>::Do([&res, v](size_t idx) { res *= v[idx]; });
		return res;
	}

	/**
	 * Cross product of 3D vectors
	 */
	template<typename T> vector_c<T, 3> crossProduct(const vector_c<T, 3>& v1, const vector_c<T, 3>& v2)
	{
		vector_c<T, 3> res;
		res[0] = v1[1] * v2[2] - v1[2] * v2[1];
		res[1] = v1[2] * v2[0] - v1[0] * v2[2];
		res[2] = v1[0] * v2[1] - v1[1] * v2[0];
		return res;
	}

#undef DEF_VECTOR_INLINE_TEMPLATE

MATH_NAMESPACE_END

#endif // VECTORTEMPLATE_H

