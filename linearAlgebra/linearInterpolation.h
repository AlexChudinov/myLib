#pragma once
#ifndef _LINEAR_INTERPOLATION
#define _LINEAR_INTERPOLATION

#include "matrixTemplate.h"
/**
 * Linear interpolation procedures
 */
MATH_NAMESPACE_BEG

/**
 * Mono-dimensional interpolation
 * Returns an interpolation coefs for an interpolation on a line
 */
template<typename Float>
std::tuple<Float, Float> lineInterpolation(
	const vector_c<Float, 3>& pos, 
	const vector_c<Float, 3>& x0,
	const vector_c<Float, 3>& x1
)
{
	std::tuple<Float, Float> res;
	vector_c<Float, 3>
		line = x1 - x0,
		pos0 = pos - x0;
	Float t = pos0*line / sqr(line);
	std::get<0>(res) = (1. - t);
	std::get<1>(res) = t;
	return res;
}

/**
 * Two-dimensional interpolation
 * Returns an interpolation coefs for an interpolation inside a triangle {x0 x1 x2}
 */
template<typename Float>
std::tuple<Float, Float, Float> triInterpolation(
	const vector_c<Float, 3>& pos,
	const vector_c<Float, 3>& x0,
	const vector_c<Float, 3>& x1,
	const vector_c<Float, 3>& x2
)
{
	std::tuple<Float, Float, Float> res;
	vector_c<Float, 3>
		pos0 = pos - x0,
		xx1 = x1 - x0,
		xx2 = x2 - x0;

	//Create plane basis
	vector_c<Float, 3>
		e1 = xx1 / abs(xx1),
		e2 = xx2 - (xx2*e1)*e1; e2 /= abs(e2);

	//Shift to a plane basis
	vector_c<Float, 2>
		pos00{ e1*pos0, e2*pos0 },
		xxx1{ e1*xx1, e2*xx1 },
		xxx2{ e1*xx2, e2*xx2 };

	//Solve linear equations
	matrix_c<Float, 2, 2> eqns, mt1, mt2;
	eqns.column(0) = xxx1 - xxx2; eqns.column(1) = pos00;
	mt1.column(0) = xxx1; mt1.column(1) = eqns.column(1);
	mt2.column(0) = eqns.column(0); mt2.column(1) = xxx1;

	double fDet = det(eqns), t1 = det(mt1) / fDet, t2 = det(mt2) / fDet;

	std::get<0>(res) = (1. - 1. / t2);
	std::get<1>(res) = (1. - t1) / t2;
	std::get<2>(res) = t1 / t2;

	return res;
}

/**
 * 3D - interpolation into a tetrahedral cell
 */
template<typename Float>
std::tuple<Float, Float, Float, Float> tetInterpolation(
	const vector_c<Float, 3>& pos,
	const vector_c<Float, 3>& x0,
	const vector_c<Float, 3>& x1,
	const vector_c<Float, 3>& x2,
	const vector_c<Float, 3>& x3
)
{
	vector_c<Float, 3>
		pos0 = pos - x0,
		xx1 = x1 - x0,
		xx2 = x2 - x0,
		xx3 = x3 - x0;

	//Solve linear equations system
	matrix_c<Float, 3, 3> eqns, tm1;
	eqns.column(0) = pos0; eqns.column(1) = xx1 - xx2; eqns.column(2) = xx1 - xx3;
	tm1.column(0) = xx1;  tm1.column(1) = eqns.column(1); tm1.column(2) = eqns.column(2);

	Float t1 = det(tm1) / det(eqns);

	std::tuple<Float, Float, Float> t1t2t3 = triInterpolation(pos0*t1, xx1, xx2, xx3);

	std::get<0>(t1t2t3) = std::get<0>(t1t2t3) / t1;
	std::get<1>(t1t2t3) = std::get<1>(t1t2t3) / t1;
	std::get<2>(t1t2t3) = std::get<2>(t1t2t3) / t1;
	t1 = 1. - 1. / t1;

	return std::tuple_cat(std::tie(t1), t1t2t3);
}

MATH_NAMESPACE_END

#endif // !_LINEAR_INTERPOLATION

