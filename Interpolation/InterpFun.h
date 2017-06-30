/**
 * @file InterpFun.h
 */
#pragma once
#ifndef _INTERPFUN_
#define _INTERPFUN_

#include <cmath>
#include <array>
#include <vector>
#include <memory>
#include <utility>
#include <stdexcept>

#include "../Base/solvers.h"

template<typename Float> class InterpFun;
template<typename Float> class ECubicSpline;

/**
 * @class InterpFun interface to interpolating function
 */
template<typename Float>
class InterpFun
{
public:
	using Pair = std::pair<Float, Float>;
	using XYDataVector = std::vector<Pair>;
	using PInterpFun = std::unique_ptr<InterpFun>;

	/**
	 * Interpolating function types
	 */
	enum InterpFunType {
		ECubic ///< Cubic spline on equally spaced intervals
	};

	/**
	 * Interpolating function params
	 */
	class InterpFunParams {
	public:
		/**
		 * @brief Checks whether the params is empty
		 * @return Returns true if the params is empty
		 */
		virtual bool valid() const { return false; }
	};

	/**
	 * @brief Creates interpolating function of a certain type
	 * @param type Type of an interpolation
	 * @param xyVals (xi, yi) pairs
	 * @param pars Additional parameters
	 * @return Pointer to interpolating function interface
	 */
	static PInterpFun create(InterpFunType type, const XYDataVector& xyVals,
		const InterpFunParams& pars = InterpFunParams()) 
	{
		switch (type)
		{
		case ECubic: return PInterpFun(new ECubicSpline<Float>(xyVals, pars));
		default: throw std::runtime_error("InterpFun::create: Unknown interpolation type.");
		}
	}

	/**
	 * @brief interpolate
	 * @param x value at whitch interpolation should be calculated
	 * @returns interpolated y value
	 */
	virtual Float interpolate(Float x) = 0;

	/**
	 * @brief differentiation of an interpolating function
	 */
	virtual void diff() = 0;
};

/**
 * @class ECubicSpline implements cubic spline with equally spaced x intervals
 * Note, it is mach faster than a usial spline because it can find an x interval for an O(1)
 */
template<typename Float>
class ECubicSpline : public InterpFun<Float>
{
public:
	using Base = InterpFun<Float>;
	using DataVector = std::vector<Float>;
	//S(x) = SplineCoef[0] + SplineCoef[1]*(x-x0) + SplineCoef[2]*(x-x0)^2 + SplineCoef[3]*(x-x0)^3
 	using SplineCoef = std::array<Float, 4>;
	using SplineCoefs = std::vector<SplineCoef>;

	/**
	 * @class ECubicSplineParams
	 * @brief Keeps weights in every point
	 */
	class ECubicSplineParams : public Base::InterpFunParams
	{
		const DataVector& m_vW;
	public:
		virtual bool valid() { return true; }
		ECubicSplineParams(const DataVector& vW) : m_vW(vW) {}

		const DataVector& weights() const { return m_vW; }
	};

	/**
	 * @param xyVals (xi, yi) pairs. It peaks only first two x values to calculate delta
	 * @param params Additional params
	 */
	ECubicSpline(const typename Base::XYDataVector& xyVals,
		const typename Base::InterpFunParams& params)
	{
		DataVector w;
		if (params.valid()) 
		{ //Calculates smoothed cubic spline
			w = static_cast<const ECubicSplineParams&>(params).weights();
		}
		else
		{ //Calculates regular cubic spline
			w.assign(xyVals.size(), 1.0);
		}

		DataVector a(w.size()), b(w.size()), c(w.size()), d(w.size());
		DataVector x(w.size()), y(w.size());
		this->m_fX0 = xyVals[0].first;
		this->m_fH = xyVals[1].first - this->m_fX0;

		x[0] = this->m_fX0; y[0] = xyVals[0].second;
		for (size_t i = 1; i < w.size(); ++i)
		{
			x[i] = x[i-1] + this->m_fH;
			y[i] = xyVals[i].second;
		}

		math::cubic_spline_coefficients<Float>(
			w.size(), a.data(), b.data(), c.data(), d.data(),
			x.data(), y.data(), w.data());

		this->m_vCoefs.resize(w.size());
		for (size_t i = 0; i < w.size(); ++i)
			this->m_vCoefs[i] = SplineCoef{ a[i], b[i], c[i] / 2., d[i] / 6. };
	}

	/**
	 * @brief Calculates index of coeffcients in array
	 * @param x Current x-value
	 * @return Index of an interval
	 */
	virtual size_t intervalIdx(Float x) const
	{
		int res = static_cast<int>((x - this->m_fX0) / this->m_fH);
		if (res < 0) return 0;
		if (res >= this->m_vCoefs.size()) return this->m_vCoefs.size() - 1;
		return static_cast<size_t>(res);
	}

	virtual Float interpolate(Float x)
	{
		size_t idx = this->intervalIdx(x);
		double dx = x - idx*this->m_fH;
		return m_vCoefs[idx][0] + dx*(m_vCoefs[idx][1] + dx*(m_vCoefs[idx][2] + dx*m_vCoefs[idx][3]));
	}

	virtual void diff()
	{
		size_t N = this->m_vCoefs.size();
		for (size_t i = 0; i < N; ++i) {
			Float 
				a = this->m_vCoefs[i][1],
				b = 2. * this->m_vCoefs[i][2],
				c = 3. * this->m_vCoefs[i][3];
			this->m_vCoefs[i] = SplineCoef{ a, b, c, 0.0 };
		}
	}

private:
	Float m_fX0, m_fH;
	SplineCoefs m_vCoefs;
};

#endif