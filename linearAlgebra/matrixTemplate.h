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

#define DEF_EPS_VAL(eps) constexpr T _Eps = eps

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
		using column_vector = vector_c<T, M>;
		using row_vector = vector_c<T, N>;
		using base_type = std::array<T, M * N>;
		using type = matrix_c;

		/**
		 * Data direction in a vector array
		 */
		enum DATA_DIRECTION
		{
			COLUMN_WISE = 0x00,
			ROW_WISE = 0x01
		};

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
			operator column_vector() const
			{
				column_vector res;
				base::For<0, M>::Do([&](size_t i) { res[i] = (*this)[i]; });
				return res;
			}
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

			inline matrix_column& operator=(matrix_column & c)
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

			operator column_vector()
			{
				column_vector res;
				base::For<0, M>::Do([&](size_t i) { res[i] = (*this)[i]; });
				return res;
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

			operator row_vector() const
			{
				row_vector res;
				base::For<0, N>::Do([&](size_t i) { res[i] = (*this)[i]; });
				return res;
			}
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

			inline matrix_row& operator=(matrix_row & r)
			{
				base::For<0, N>::Do([&](size_t idx) { (*this)[idx] = r[idx]; });
				return *this;
			}

			inline matrix_row& operator=(const const_matrix_row & r)
			{
				base::For<0, N>::Do([&](size_t idx) { (*this)[idx] = r[idx]; });
				return *this;
			}

			inline matrix_row& operator=(const row_array & r)
			{
				base::For<0, N>::Do([&](size_t idx) { (*this)[idx] = r[idx]; });
				return *this;
			}

			operator row_vector()
			{
				row_vector res;
				base::For<0, N>::Do([&](size_t i) { res[i] = (*this)[i]; });
				return res;
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
			base::For<0, M>::Do([&](size_t i) 
			{
				matrix_row _row(*this, i);
				_row = *(cur++);
			});
		}
	};

	/**
	 * Zeros matrix
	 */
	DEF_MATRIX_TEMPLATE_INLINE matrix_c<DEF_MATRIX_TEMPLATE_PARAMS> zeros()
	{
		matrix_c<DEF_MATRIX_TEMPLATE_PARAMS> res; res.fill(0.0);
		return res;
	}

	/**
	 * Matrix of ones
	 */
	DEF_MATRIX_TEMPLATE_INLINE matrix_c<DEF_MATRIX_TEMPLATE_PARAMS> ones()
	{
		matrix_c<DEF_MATRIX_TEMPLATE_PARAMS> res; res.fill(1.0);
		return res;
	}

	/**
	 * Identity matrix
	 */
	DEF_SQUARE_MATRIX_TEMPLATE_INLINE matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS> eye()
	{
		matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS> res = zeros<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS>();
		base::For<0, M>::Do([&](size_t i) { res(i, i) = 1.0; });
		return res;
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
		const std::vector<vector_c<T, M>>& vs,      //Vectors in a N-hyperspace
		const std::vector<T>& ws                     //Weightings
	)
	{
		using matrix = matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS>;
		using vector_c = vector_c<T, M>;

		std::vector<T> wws(ws);
		wws.resize(vs.size(), 1.0);

		T norm_w = std::accumulate(wws.begin(), wws.end(), 0.0);
		vector_c mean_v = std::inner_product(wws.begin(), wws.end(), vs.begin(), vector_c(0.0)) / norm_w;
		matrix acc = zeros<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS>();

		return std::inner_product(vs.begin(), vs.end(), wws.begin(), acc, std::plus<matrix>(),
			[mean_v](const vector_c& v, const T& w)->matrix
		{
			matrix m;
			vector_c vv = v - mean_v;
			base::For<0, M>::Do([&](size_t i)
			{
				m(i, i) = vv[i] * vv[i] * w;
				for (size_t j = 0; j < i; ++j) m(i, j) = m(j, i) = vv[i] * vv[j] * w;
			});
			return m;
		}) / norm_w * static_cast<T>(vs.size()) / static_cast<T>(vs.size() - 1);
	}

	DEF_SQUARE_MATRIX_TEMPLATE_INLINE matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS> cov(const std::vector<vector_c<T, M>> & vs)
	{
		using matrix = matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS>;
		using vector_c = vector_c<T, M>;

		T norm_w = static_cast<T>(vs.size());
		vector_c mean_v = std::accumulate(vs.begin(), vs.end(), vector_c(0.0)) / norm_w;
		matrix acc = zeros<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS>();

		return std::accumulate(vs.begin(), vs.end(),
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
	
	/**
	 * Gram-Schmidt QR-factorization
	 */
	DEF_SQUARE_MATRIX_TEMPLATE using qr_factorization_result 
		= std::pair< matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS>, matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS> >;
	DEF_SQUARE_MATRIX_TEMPLATE_INLINE qr_factorization_result<T, M> qrGramSchmidt(const matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS>& m)
	{
		using vector = vector_c<T, M>;
		qr_factorization_result<T, M> res; res.first.fill(0.0); res.second.fill(0.0);

		res.first.column(0) = vector(m.column(0)) / abs(vector(m.column(0)));
		res.second(0, 0) = vector(m.column(0)) * vector(res.first.column(0));
		
		base::For<1, M>::Do([&](size_t i)
		{
			vector coli = m.column(i);
			for (size_t j = 0; j < i; ++j)
			{ 
				vector colj = res.first.column(j);
				coli -= (vector(m.column(i)) * colj) * colj;
			}

			res.first.column(i) = coli / abs(coli);

			for (size_t j = 0; j <= i; ++j)
			{
				res.second(j, i) = vector(m.column(i)) * vector(res.first.column(j));
			}
		});

		return res;
	}

	/**
	 * Sum of a matrix diagonal elements
	 */
	DEF_SQUARE_MATRIX_TEMPLATE_INLINE T tr(const matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS>& m)
	{
		T res = 0.0;
		base::For<0, M>::Do([&](size_t i) { res += m(i, i); });
		return res;
	}
	
	/**
	 * Get matrix diagonal elements
	 */
	DEF_SQUARE_MATRIX_TEMPLATE_INLINE vector_c<T, M> diag(const matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS>& m)
	{
		vector_c<T, M> res;
		base::For<0, M>::Do([&](size_t i) { res[i] = m(i, i); });
		return res;
	}

	/**
	 * Calculates eigen values using QR factorization with the Gram-Schmidt algorithm
	 */
	DEF_SQUARE_MATRIX_TEMPLATE_INLINE vector_c<T, M> eigenValsQRGramSchmidt(const matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS>& m)
	{
		DEF_EPS_VAL(std::numeric_limits<T>::epsilon() * 10.0);

		using matrix = matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS>;
		using vector = vector_c<T, M>;
 
		T eps = _Eps * *std::max_element(m.begin(), m.end());
		qr_factorization_result<T, M> QR;
		matrix A(m);
		vector d;

		do
		{
			d = diag(A);
			QR = qrGramSchmidt(A);
			A = QR.second * QR.first;
		} while (abs(diag(A) - d) / abs(d) > eps);

		return diag(A);
	}

	/**
	 * Finds eigen vector with the biggest eigen value using simple eigenvalue algorithm
	 */
	DEF_SQUARE_MATRIX_TEMPLATE_INLINE vector_c<T, M> eigenVectorSimple(const matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS>& m)
	{
		DEF_EPS_VAL(std::numeric_limits<T>::epsilon() * 10.0);
		vector_c<T, M> cur, next(1.0);
		int maxIterNum = 1000, iterCount = 0;

		do
		{
			cur = next;
			(next = m*cur) /= abs(next);
		} while (abs(cur - next) > _Eps && iterCount++ < maxIterNum);

		return next /= abs(next);
	}

	/**
	* Finds all eigen vectors using simple eigenvalue algorithm
	*/
	DEF_SQUARE_MATRIX_TEMPLATE_INLINE matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS> 
		eigenVectorsSimple(const matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS>& m)
	{
		using vector = vector_c<T, M>;

		DEF_EPS_VAL(std::numeric_limits<T>::epsilon() * 10.0);
		matrix_c<DEF_SQUARE_MATRIX_TEMPLATE_PARAMS> eigenVectors;
		base::For<0, M>::Do([&](size_t i)
		{
			vector cur, next(1.0);
			int maxIterNum = 1000, iterCount = 0;
			do
			{
				cur = next;
				next = m*cur;
				for (size_t j = 0; j < i; ++j)
					next -= (next*vector(eigenVectors.column(j))) * vector(eigenVectors.column(j));
				next /= abs(next);
			} while (abs(cur - next) > _Eps && iterCount++ < maxIterNum);
			eigenVectors.column(i) = next;
		});
		return eigenVectors;
	}

#undef DEF_SQUARE_MATRIX_TEMPLATE_INLINE
#undef DEF_MATRIX_TEMPLATE_INLINE
#undef DEF_SQUARE_MATRIX_TEMPLATE
#undef DEF_MATRIX_TEMPLATE
#undef DEF_SQUARE_MATRIX_TEMPLATE_PARAMS
#undef DEF_MATRIX_TEMPLATE_PARAMS

MATH_NAMESPACE_END

#endif // MATRIXTEMPLATE_H

