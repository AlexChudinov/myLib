#pragma once
#ifndef MATRIXTEMPLATE_H
#define MATRIXTEMPLATE_H

#include <array>
#include <vector>
#include <numeric>

#include "../Base/templateFor.h"
#include "vectorTemplate.h"

/**
* Template defines simple math matrix operations
*/

MATH_NAMESPACE_BEG

#define DEF_MATRIX_TEMPLATE_PARAMS T, M, N
#define DEF_SQUARE_MATRIX_TEMPLATE_PARAMS T, M, M
#define DEF_MATRIX_TEMPLATE template<typename T, size_t M, size_t N>
#define DEF_SQUARE_MATRIX_TEMPLATE template<typename T, size_t M>
#define DEF_MATRIX_TEMPLATE_INLINE DEF_MATRIX_TEMPLATE inline
#define DEF_SQUARE_MATRIX_TEMPLATE_INLINE DEF_SQUARE_MATRIX_TEMPLATE inline

	DEF_MATRIX_TEMPLATE	class matrix_c : public std::array<T, M * N>
	{
	public:
		using column_array = std::array<T, M>;
		using row_array = std::array<T, N>;
		using base_type = std::array<T, M * N>;
		using type = matrix_c;

		/**
		* Returns the number of a matrix rows
		*/
		static const size_t nrows = M;

		/**
		* Returns the number of a matrix columns
		*/
		static const size_t ncols = N;

		/**
		* Returns matrix element
		*/
		inline const T& operator()(size_t j, size_t i) const { return (*this)[i*nrows + j]; }
		inline T& operator()(size_t j, size_t i) { return (*this)[i*nrows + j]; }

		/**
		* Constant matrix's column
		*/
		class const_matrix_column
		{
			const matrix_c& m_;
			const size_t col_idx_;

		public:
			inline const_matrix_column(const matrix_c& m, size_t col_idx) : m_(m), col_idx_(col_idx) {}
			inline const T& operator[](size_t idx) const { return m_(idx, col_idx_); }

		};

		/**
		* Matrix's column
		*/
		class matrix_column
		{
			matrix_c& m_;
			const size_t col_idx_;

		public:
			inline matrix_column(matrix_c& m, size_t col_idx) : m_(m), col_idx_(col_idx) {}
			inline T& operator[](size_t idx) { return m_(idx, col_idx_); }

			inline matrix_column& operator=(const matrix_column & c)
			{
				base::For<0, M>::Do([&](size_t idx) { (*this)[idx] = c[idx]; });
				return *this;
			}

			inline matrix_column& operator=(const const_matrix_column & c)
			{
				base::For<0, M>::Do([&](size_t idx) { (*this)[idx] = c[idx]; });
				return *this;
			}

			inline matrix_column& operator=(const column_array & c)
			{
				base::For<0, M>::Do([&](size_t idx) { (*this)[idx] = c[idx]; });
				return *this;
			}
		};

		/**
		 * Constant matrix's row
		 */
		class const_matrix_row
		{
			const matrix_c& m_;
			const size_t row_idx_;

		public:
			inline const_matrix_row(const matrix_c& m, size_t row_idx) : m_(m), row_idx_(row_idx){}
			inline const T& operator[](size_t idx) const { return m_(row_idx_, idx); }
		};

		/**
	     * Matrix's row
		 */
		class matrix_row
		{
			matrix_c& m_;
			const size_t row_idx_;
			
		public:
			inline matrix_row(matrix_c& m, size_t row_idx) : m_(m), row_idx_(row_idx){}
			inline T& operator[](size_t idx) { return m_(row_idx_, idx); }

			inline matrix_row& operator=(const matrix_row & r)
			{
				base::For<0, N>::Do([&](size_t idx) { (*this)[idx] = r[idx]; });
				return *this;
			}

			inline matrix_row& operator=(const const_matrix_row & r)
			{
				base::For<0, N>::Do([&](size_t idx) { (*this)[idx] = r[idx]; });
				return *this;
			}

			inline matrix_row& operator=(const row_array & c)
			{
				base::For<0, N>::Do([&](size_t idx) { (*this)[idx] = c[idx]; });
				return *this;
			}
		};

		/**
		* Returns proxy to a matrix column
		*/
		inline matrix_column column(size_t idx) { return matrix_column(*this, idx); }
		inline const_matrix_column column(size_t idx) const { return const_matrix_column(*this, idx); }

		/**
		* Returns proxy to a matrix row
		*/
		inline matrix_row row(size_t idx) { return matrix_row(*this, idx); }
		inline const_matrix_row row(size_t idx) const { return const_matrix_row(*this, idx); }

		matrix_c() {}

		/**
		 * Initialise the matrix with a list of column vectors
		 */
		matrix_c(std::initializer_list<row_array> list)
		{
			std::initializer_list<row_array>::iterator cur = list.begin();
			base::For<0, M>::Do([&](size_t idx) 
			{
				matrix_row _row(*this, idx);
				_row = *(cur++);
			});
		}	
	};

	/**
	 * Identity matrix
	 */
	DEF_SQUARE_MATRIX_TEMPLATE_INLINE matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS> eye()
	{
		matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS> I;
		I.fill(0.0);
		base::For<0, M>::Do([&](size_t i) { I(i, i) = 1.0; });
		return I;
	}

	/**
	 * Matrix printing
	 */
	DEF_MATRIX_TEMPLATE_INLINE std::ostream& operator<<(std::ostream& out, const matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& m)
	{
		base::For<0, M>::Do([&](size_t i)
		{
			out << i << "| ";
			base::For<0, N>::Do([&](size_t j) { out << m(i, j) << "\t"; });
			out << "\n";
		});
		return out;
	}

	/**
	 * Matrix addition
	 */
	DEF_MATRIX_TEMPLATE_INLINE matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& operator += (
		matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& m1,
		const matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& m2
		)
	{
		base::For<0, M * N>::Do([&](size_t i) { m1[i] += m2[i]; });
		return m1;
	}
	DEF_MATRIX_TEMPLATE_INLINE matrix_c<DEF_MATRIX_TEMPLATE_PARAMS> operator + (
		const matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& m1,
		const matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& m2
		)
	{
		matrix_c<DEF_MATRIX_TEMPLATE_PARAMS> res = m1;
		return res += m2;
	}

	/**
	 * Matrix negation
	 */
	DEF_MATRIX_TEMPLATE_INLINE matrix_c<DEF_MATRIX_TEMPLATE_PARAMS> operator-(const matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& m)
	{
		matrix_c<DEF_MATRIX_TEMPLATE_PARAMS> res;
		base::For<0, M * N>::Do([&](size_t i) { res[i] = -m[i]; });
		return res;
	}

	/**
	 * Matrix subtraction
	 */
	DEF_MATRIX_TEMPLATE_INLINE matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& operator-=(
		matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& m1,
		const matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& m2)
	{
		base::For<0, M * N>::Do([&](size_t i) { m1[i] -= m2[i]; });
		return m1;
	}
	DEF_MATRIX_TEMPLATE_INLINE matrix_c<DEF_MATRIX_TEMPLATE_PARAMS> operator - (
		const matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& m1,
		const matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& m2)
	{
		matrix_c<DEF_MATRIX_TEMPLATE_PARAMS> res = m1;
		return res -= m2;
	}

	/**
	* Matrix multiplication by a right vector
	*/
	DEF_MATRIX_TEMPLATE_INLINE vector_c<T, M> operator * (const matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& A, const vector_c<T, N>& x)
	{
		vector_c<T, M> res(0.0);		
		base::For<0, M>::Do([&](size_t i)
		{
			matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>::const_matrix_row _row = A.row(i);
			base::For<0, N>::Do([&](size_t j) { res[i] += _row[j] * x[j]; });
		});
		return res;
	}

	/**
	 * Matrix multiplication by a left vector
	 */
	DEF_MATRIX_TEMPLATE_INLINE vector_c<T, N> operator * (const vector_c<T, M>& x, const matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& A)
	{
		vector_c<T, N> res(0.0);
		base::For<0, N>::Do([&](size_t i)
		{
			matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>::const_matrix_column _col = A.column(i);
			base::For<0, M>::Do([&](size_t j) { res[i] += _col[j] * x[j]; });
		});
		return res;
	}

	/**
	* Matrix multiplication
	*/
	template<typename T, size_t M, size_t N, size_t K> matrix_c<T, M, K> operator * (
			const matrix_c<T, M, N>& m1,
			const matrix_c<T, N, K>& m2
		)
	{
		matrix_c<T, M, K> res;
		res.fill(0.0);
		base::For<0, M>::Do([&](size_t i)
		{
			matrix_c<T, M, N>::const_matrix_row _row = m1.row(i);
			base::For<0, K>::Do([&](size_t j)
			{
				matrix_c<T, N, K>::const_matrix_column _col = m2.column(j);
				base::For<0, N>::Do([&](size_t k) { res(i, j) += _row[k] * _col[k]; });
			});
		});
		return res;
	}

	/**
	* Matrix division by a number
	*/
	DEF_MATRIX_TEMPLATE_INLINE matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& operator /= (matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& m, const T& h)
	{
		base::For<0, M * N>::Do([&](size_t idx) { m[idx] /= h; });
		return m;
	}
	DEF_MATRIX_TEMPLATE_INLINE matrix_c<DEF_MATRIX_TEMPLATE_PARAMS> operator / (const matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& m, const T& h)
	{
		matrix_c<DEF_MATRIX_TEMPLATE_PARAMS> res(m);
		return res /= h;
	}

	/**
	* Matrix multiplication by a number
	*/
	DEF_MATRIX_TEMPLATE_INLINE matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& operator *= (matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& m, const T& h)
	{
		base::For<0, M * N>::Do([&](size_t idx) { m[idx] *= h;  });
		return m;
	}
	DEF_MATRIX_TEMPLATE_INLINE matrix_c<DEF_MATRIX_TEMPLATE_PARAMS> operator * (const matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& m, const T& h)
	{
		matrix_c<DEF_MATRIX_TEMPLATE_PARAMS> res(m);
		return res *= h;
	}
	DEF_MATRIX_TEMPLATE_INLINE matrix_c<DEF_MATRIX_TEMPLATE_PARAMS> operator * (const T& h, const matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& m)
	{
		matrix_c<DEF_MATRIX_TEMPLATE_PARAMS> res(m);
		return res *= h;
	}

	/**
	* Matrix transposition
	*/
	DEF_MATRIX_TEMPLATE_INLINE matrix_c<DEF_MATRIX_TEMPLATE_PARAMS> transpose(const matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>& m)
	{
		matrix_c<T, N, M> res;
		base::For<0, M>::Do([&](size_t i)
		{
			base::For<0, N>::Do([&](size_t j) { res(j, i) = m(i, j); });
		});
		return res;
	}

	/**
	* Calculates covariance matrix
	*/
	DEF_SQUARE_MATRIX_TEMPLATE_INLINE matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS> cov
	(
		const std::vector<vector_c<T, M>>& vectors, //Vectors in a N-hyperspace
		const std::vector<T>& w = std::vector<T>()  //Weightings
	)
	{
		using matrix = matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS>;
		using vector_c = vector_c<T, M>;

		T norm_w;
		vector_c mean_v;
		matrix acc; acc.fill(0.0);

		if (w.empty() || vectors.size() != w.size()) //Do not use weightings
		{
			norm_w = static_cast<T>(vectors.size());
			mean_v = std::accumulate(vectors.begin(), vectors.end(), vector_c(0.0)) / norm_w;

			return std::accumulate(vectors.begin(), vectors.end(),
				acc, [&](matrix& m, const vector_c& v)->matrix&
			{
				vector_c vv = v - mean_v;
				base::For<0, M>::Do([&](size_t j)
				{
					m(j, j) += vv[j] * vv[j];
					for (size_t k = 0; k < j; ++k) m(k, j) = m(j, k) += vv[j] * vv[k];
				});
				return m;
			}) / (norm_w - 1.0);
		}
		else
		{
			norm_w = std::accumulate(w.begin(), w.end(), 0.0);
			mean_v = std::inner_product(w.begin(), w.end(), vectors.begin(), vector_c(0.0)) / norm_w;

			size_t i = 0;
			return std::accumulate(vectors.begin(), vectors.end(),
				acc, [&](matrix& m, const vector_c& v)->matrix&
			{
				vector_c vv = v - mean_v;
				base::For<0, M>::Do([&](size_t j)
				{
					m(j, j) += w[i] * vv[j] * vv[j];
					for (size_t k = 0; k < j; ++k) m(k, j) = m(j, k) += w[i] * vv[j] * vv[k];
				});
				return m;
			}) / norm_w * static_cast<T>(vectors.size() / (vectors.size() - 1));
		}
	}

	/**
	* Calculates first principal component
	*/
/*	template<class T, size_t N>
	vector_c<T, N> pc1
	(
		const vector<vector_c<T, N>>& vectors,
		const vector<T>& w,
		T relTol = 1.0e-10,
		size_t maxItter = 1000
	)
	{
		using vector_c = vector_c<T, N>;
		using matrix = matrix_c<T, N, N>;

		//Calculate cloud center
		vector_c v0(0.0);
		math::mean<vector_c, double, vector>(vectors, w, v0);

		//Make first approximation
		vector_c eigen_vector(0.0);
		typename vector<T>::const_iterator it = w.begin();
		for (const vector_c& v : vectors)
		{
			vector_c vv = v - v0;
			if (eigen_vector*vv < 0.0)
				eigen_vector -= (vv*vv)*vv * *(it++);
			else
				eigen_vector += (vv*vv)*vv * *(it++);
		}

		if (abs(eigen_vector) == 0) return eigen_vector;
		eigen_vector /= abs(eigen_vector);

		matrix covMatrix = cov(vectors, w, v0);
		eigen_vector = covMatrix * eigen_vector;
		T disp0, disp1 = abs(eigen_vector);
		size_t iter = 0;
		do
		{
			disp0 = disp1;
			eigen_vector /= disp1;
			eigen_vector = covMatrix * eigen_vector;
			disp1 = abs(eigen_vector);
		} while (std::fabs(disp1 - disp0) / disp0 > relTol
			&& (++iter) != maxItter);

		return eigen_vector;
	}

	/**
	* Matrix determinant
	*/
/*	template<class T, size_t N>
	inline T  det(const matrix_c<T, N, N>& m)
	{
		using matrix = matrix_c<T, N, N>;
		using trace = const_proxy_matrix_diag<T, N, N>;

		matrix tri(m);
		int det_factor = 1;

		auto matrix_triangulation = [&tri, &det_factor](size_t row)->void
		{
			size_t next_row = row + 1;
			while (tri[row][row] == 0.0 && next_row < N)
			{
				std::swap(tri[row], tri[next_row++]);
				det_factor = -det_factor;
			}
			for (; next_row < N; ++next_row)
			{
				T coef = tri[next_row][row] / tri[row][row];
				tri[next_row][row] = 0.0;
				for (size_t col = row + 1; col < N; ++col)
					tri[next_row][col] -= coef*tri[row][col];
			}
		};

		For<0, N - 1, true>().Do(matrix_triangulation);
		double res = static_cast<double>(det_factor);
		math::array_operations<trace, trace, N - 1> op;
		op.ufold(math::in_place_mul<T>(), res, tri.diag());

		return res;
	}

	/**
	* Solves a linear equation system Ax=b,
	* VarNum is an index in the x array
	* determinant should be previously estimated
	*/
/*	template<class T, size_t N, size_t VarNum>
	inline T solve
	(
		const matrix_c<T, N, N>& A,
		const vector_c<T, N>& b,
		const T& D)
	{
		static_assert(N > VarNum, "Index of a variable is too big");

		using matrix = matrix_c<T, N, N>;

		matrix A_b;

		for (size_t i = 0; i < VarNum; ++i) A_b.column(i) = A.column(i);
		for (size_t i = VarNum + 1; i < N; ++i) A_b.column(i) = A.column(i);

		A_b.column(VarNum) = b;

		return math::det(A_b) / D;
	}*/

#undef DEF_SQUARE_MATRIX_TEMPLATE_INLINE
#undef DEF_MATRIX_TEMPLATE_INLINE
#undef DEF_SQUARE_MATRIX_TEMPLATE
#undef DEF_MATRIX_TEMPLATE
#undef DEF_SQUARE_MATRIX_TEMPLATE_PARAMS
#undef DEF_MATRIX_TEMPLATE_PARAMS

MATH_NAMESPACE_END

#endif // MATRIXTEMPLATE_H

