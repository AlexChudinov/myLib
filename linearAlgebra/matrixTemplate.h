#pragma once
#ifndef MATRIXTEMPLATE_H
#define MATRIXTEMPLATE_H

#include <array>

#include "../Base/templateFor.h"

/**
* Template defines simple math matrix operations
*/

MATH_NAMESPACE_BEG

#define DEF_MATRIX_TEMPLATE_PARAMS T, M, N
#define DEF_MATRIX_TEMPLATE template<class T, size_t M, size_t N>
#define DEF_MATRIX_TEMPLATE_INLINE DEF_MATRIX_TEMPLATE inline

	DEF_MATRIX_TEMPLATE	class matrix_c : private std::array<T, M * N>
	{
	public:
		using column_array = std::array<T, M>;
		using row_array = std::array<T, N>;
		using base_type = std::array<T, M * N>;
		using type = matrix_c;

		/**
		* Constant matrix's column
		*/
		class const_matrix_column
		{
			const matrix_c& m_;
			const size_t idx_col_;

		public:
			inline const_matrix_column(const matrix_c& m, size_t idx_col) : m_(m), idx_col_(idx_col) {}
			inline const T& operator[](size_t idx_row) const { return m_[idx_col_ * M + idx_row]; }
		};

		/**
		* Matrix's column
		*/
		class matrix_column
		{
			matrix_c& m_;
			const size_t idx_col_;

		public:
			inline matrix_column(matrix_c& m, size_t idx_col) : m_(m), idx_col_(idx_col) {}

			inline T& operator[](size_t idx_row) { return m_[idx_col_ * M + idx_row]; }

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

		matrix_c() {}

		/**
		 * Initialise the matrix with a list of column vectors
		 */
		matrix_c(std::initializer_list<row_array> list)
		{
			base_type::iterator dest_first = begin();
			std::initializer_list<row_array>::const_iterator src_first = list.begin();
			base::For<0, M>::Do([&](size_t idx) 
			{
				std::copy(src_first->begin(), src_first->end(), dest_first);
				++src_first;
				dest_first += N;
			});
		}

		/**
		 * Returns proxy to a matrix column
		 */
		inline proxy_matrix_col<DEF_MATRIX_TEMPLATE_PARAMS> column(size_t idx_col);

		/**
		 * Returns proxy to a matrix row
		 */
		inline proxy_matrix_row<DEF_MATRIX_TEMPLATE_PARAMS> row(size_t idx_row);
		inline proxy_matrix_row<DEF_MATRIX_TEMPLATE_PARAMS> operator[](size_t idx_row)
		{
			return row(idx_row);
		}

		/**
		 * Returns proxy to a matrix main diagonal = trace
		 */
		inline proxy_matrix_diag<DEF_MATRIX_TEMPLATE_PARAMS> diag();

		/**
		* Returns proxy to a constant matrix column
		*/
		inline const_proxy_matrix_col<DEF_MATRIX_TEMPLATE_PARAMS> column(size_t idx_col) const;

		/**
		* Returns proxy to a constant matrix row
		*/
		inline const_proxy_matrix_row<DEF_MATRIX_TEMPLATE_PARAMS> row(size_t idx_row) const;
		inline const_proxy_matrix_row<DEF_MATRIX_TEMPLATE_PARAMS> operator[](size_t idx_row) const 
		{ 
			return row(idx_row);
		}

		/**
		* Returns proxy to a constant matrix main diagonal
		*/
		inline const_proxy_matrix_diag<DEF_MATRIX_TEMPLATE_PARAMS> diag() const;

		/**
		* Returns the number of a matrix rows
		*/
		static const size_t nrows = M;

		/**
		* Returns the number of a matrix columns
		*/
		static const size_t ncols = N;
	};

	DEF_MATRIX_TEMPLATE class proxy_matrix_col
	{
	public:
		using matrix_type = matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>;

	private:
		matrix_type& A_;
		const size_t idx_col_;

	public:
		using const_column_type = const_proxy_matrix_col<DEF_MATRIX_TEMPLATE_PARAMS>;
		using vector_column_type = vector_c<T, N>;

	};

	DEF_MATRIX_TEMPLATE	struct const_proxy_matrix_col
	{
	public:
		using matrix_type = matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>;

	private:
		const matrix& A_;
		const size_t idx_col_;

	public:
		using column_type = proxy_matrix_col<DEF_MATRIX_TEMPLATE_PARAMS>;

		inline const_proxy_matrix_col(const matrix_type& A, size_t idx_col) : A_(A), idx_col_(idx_col) {}

		inline const_proxy_matrix_col(const proxy_matrix_col<T, m, n>& col) : A_(col.A_), idx_col_(col.idx_col_) {}

		inline const T& operator[](size_t idx_row) const{ return A_[idx_row][idx_col_]; }
	};

	DEF_MATRIX_TEMPLATE_INLINE proxy_matrix_col<DEF_MATRIX_TEMPLATE_PARAMS> matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>::column(size_t idx_col)
	{
		return proxy_matrix_col<DEF_MATRIX_TEMPLATE_PARAMS>(*this, idx_col);
	}

	DEF_MATRIX_TEMPLATE_INLINE proxy_matrix_col<DEF_MATRIX_TEMPLATE_PARAMS> matrix_c<DEF_MATRIX_TEMPLATE_PARAMS>::column(size_t idx_col) const
	{
		return const_proxy_matrix_col<T, m, n>(*this, idx_col);
	}

	template<class T, size_t M, size_t N>
	struct proxy_matrix_row
	{
		using matrix = matrix_c<T, M, N>;
		using vector = vector_c<T, N>;
		using const_row = const_proxy_matrix_row<T, M, N>;

		matrix& A_;
		size_t idx_row_;

		inline proxy_matrix_row(matrix& A, size_t idx_row)
			:
			A_(A), idx_row_(idx_row) {}

		inline T& operator[](size_t idx_col) { return A_[idx_row_][idx_col]; }

		inline proxy_matrix_row& operator=(const proxy_matrix_row& r)
		{
			math::array_operations<proxy_matrix_row, proxy_matrix_row, M - 1> op;
			op.bmap(math::set_val<T>(), *this, r);
			return *this;
		}

		inline proxy_matrix_row& operator=(const const_row& r)
		{
			math::array_operations<proxy_matrix_row, const_row, M - 1> op;
			op.bmap(math::set_val<T>(), *this, r);
			return *this;
		}

		inline proxy_matrix_row& operator=(const vector& v)
		{
			math::array_operations<proxy_matrix_row, vector, M - 1> op;
			op.bmap(math::set_val<T>(), *this, v);
			return *this;
		}
	};

	template<class T, size_t m, size_t n>
	struct const_proxy_matrix_row
	{
		using matrix = matrix_c<T, m, n>;
		const matrix& A_;
		size_t idx_row_;

		inline const_proxy_matrix_row(const matrix& A, size_t idx_row)
			:
			A_(A), idx_row_(idx_row)
		{}

		inline const_proxy_matrix_row(const proxy_matrix_row<T, m, n>& row)
			:
			A_(row.A_), idx_row_(row.idx_row_)
		{}

		inline const T& operator[](size_t idx_col) const
		{
			return A_[idx_row_][idx_col];
		}
	};

	template<class T, size_t m, size_t n> inline
		proxy_matrix_row<T, m, n> matrix_c<T, m, n>::row(size_t idx_row)
	{
		return proxy_matrix_row<T, m, n>(*this, idx_row);
	}

	template<class T, size_t m, size_t n> inline
		const_proxy_matrix_row<T, m, n> matrix_c<T, m, n>::row(size_t idx_row) const
	{
		return const_proxy_matrix_row<T, m, n>(*this, idx_row);
	}

	template<class T, size_t M, size_t N>
	struct proxy_matrix_diag
	{
		using matrix = matrix_c<T, M, N>;
		matrix& A_;
		inline proxy_matrix_diag(matrix& A) : A_(A) {}
		inline T& operator[](size_t diag_idx) { return A_[diag_idx][diag_idx]; }
	};

	template<class T, size_t M, size_t N>
	struct const_proxy_matrix_diag
	{
		using matrix = matrix_c<T, M, N>;
		const matrix& A_;
		inline const_proxy_matrix_diag(const matrix& A) : A_(A) {}
		inline const_proxy_matrix_diag
		(
			const proxy_matrix_diag<T, M, N>& D) : A_(D.A_) {}
		inline const T& operator[](size_t diag_idx) const { return A_[diag_idx][diag_idx]; }
	};

	template<class T, size_t M, size_t N> inline
		proxy_matrix_diag<T, M, N> matrix_c<T, M, N>::diag()
	{
		return proxy_matrix_diag<T, M, N>(*this);
	}

	template<class T, size_t M, size_t N> inline
		const_proxy_matrix_diag<T, M, N> matrix_c<T, M, N>::diag() const
	{
		return const_proxy_matrix_diag<T, M, N>(*this);
	}

	/**
	* Folds a matrix row with a matrix column
	*/
	template<class T, size_t m, size_t k, size_t n>
	T operator *
		(
			const_proxy_matrix_row<T, m, k>& row,
			const_proxy_matrix_col<T, k, n>& col)
	{
		using row_vector = const_proxy_matrix_row<T, m, k>;
		using col_vector = const_proxy_matrix_col<T, k, n>;
		T res(0.0);
		math::array_operations<row_vector, col_vector, k - 1> op;
		op.bfold(math::in_place_plus<T>(), std::multiplies<T>(), res, row, col);
		return res;
	}
	/**
	* Folds a matrix row with a matrix column (rvalue variant)
	*/
	template<class T, size_t m, size_t k, size_t n>
	T operator *
		(
			const_proxy_matrix_row<T, m, k>&& row,
			const_proxy_matrix_col<T, k, n>&& col)
	{
		using row_vector = const_proxy_matrix_row<T, m, k>;
		using col_vector = const_proxy_matrix_col<T, k, n>;
		T res(0.0);
		math::array_operations<row_vector, col_vector, k - 1> op;
		op.bfold(math::in_place_plus<T>(), std::multiplies<T>(), res, row, col);
		return res;
	}

	/**
	* Matrix multiplication by a vector
	*/
	template<class T, size_t m, size_t n>
	vector_c<T, m> operator * (const matrix_c<T, m, n>& A, const vector_c<T, n>& x)
	{
		using matrix = matrix_c<T, m, n>;
		using vector_row = vector_c<T, n>;
		using vector_col = vector_c<T, m>;

		vector_col res(0.0);

		struct fold_matrix_row
		{
			const matrix& A_;
			const vector_row& x_;
			vector_col& y_;

			inline fold_matrix_row
			(
				const matrix& A,
				const vector_row& x,
				vector_col& y)
				:
				A_(A), x_(x), y_(y) {}

			inline void operator()(size_t idx_row)
			{
				y_[idx_row] = A_[idx_row] * x_;
			}
		};

		math::For<0, m, true>().Do(fold_matrix_row(A, x, res));

		return res;
	}

	/**
	* Matrix multiplication
	*/
	template<class T, size_t m, size_t k, size_t n>
	matrix_c<T, m, n> operator *
		(
			const matrix_c<T, m, k>& m1,
			const matrix_c<T, k, n>& m2)
	{
		using lmatrix_type = matrix_c<T, m, k>;
		using rmatrix_type = matrix_c<T, k, n>;
		using result_type = matrix_c<T, m, n>;
		using result_row = proxy_matrix_row<T, m, n>;

		result_type res(0.0);

		struct one_result_row
		{
			const lmatrix_type& m1_;
			const rmatrix_type& m2_;
			result_type& res_;
			inline one_result_row
			(
				const lmatrix_type& m1,
				const rmatrix_type& m2,
				result_type& res) : m1_(m1), m2_(m2), res_(res)
			{}

			inline void operator () (size_t row_idx)
			{
				struct one_result_elem
				{
					const lmatrix_type& m1_;
					const rmatrix_type& m2_;
					result_row res_;

					inline one_result_elem
					(
						const lmatrix_type& m1,
						const rmatrix_type& m2,
						result_row res)
						:m1_(m1), m2_(m2), res_(res)
					{}

					inline void operator()(size_t col_idx)
					{
						res_[col_idx] = m1_.row(res_.idx_row_) * m2_.column(col_idx);
					}
				};

				math::For<0, n, true>()
					.Do(one_result_elem(m1_, m2_, res_.row(row_idx)));
			}
		};

		math::For<0, m, true>().Do(one_result_row(m1, m2, res));

		return res;
	}

	/**
	* Matrix division by a number
	*/
	template<class T, size_t m, size_t n>
	matrix_c<T, m, n>& operator /= (matrix_c<T, m, n>& M, const T& h)
	{
		using row_type = vector_c<T, n>;
		using matrix = matrix_c<T, m, n>;
		DEF_OPERATION_WITH_VAL_1(T, row_type, /=);
		math::array_operations<matrix, matrix, m - 1> op;
		op.umap(operation(h), M);
		return M;
	}
	template<class T, size_t m, size_t n>
	matrix_c<T, m, n> operator / (const matrix_c<T, m, n>& M, const T& h)
	{
		matrix_c<T, m, n> result(M);
		return result /= h;
	}

	/**
	* Matrix multiplication by a number
	*/
	template<class T, size_t m, size_t n>
	matrix_c<T, m, n>& operator *= (matrix_c<T, m, n>& M, const T& h)
	{
		using row_type = vector_c<T, n>;
		using matrix = matrix_c<T, m, n>;
		DEF_OPERATION_WITH_VAL_1(T, row_type, *=);
		math::array_operations<matrix, matrix, m - 1> op;
		op.umap(operation(h), M);
		return M;
	}
	template<class T, size_t m, size_t n>
	matrix_c<T, m, n> operator * (const matrix_c<T, m, n>& M, const T& h)
	{
		matrix_c<T, m, n> result(M);
		return result *= h;
	}
	template<class T, size_t m, size_t n>
	matrix_c<T, m, n> operator * (const T& h, const matrix_c<T, m, n>& M)
	{
		matrix_c<T, m, n> result(M);
		return result *= h;
	}

	/**
	* Matrix transposition
	*/
	template<class T, size_t m, size_t n>
	matrix_c<T, n, m> transpose(const matrix_c<T, m, n>& M)
	{
		using matrix = matrix_c<T, m, n>;
		using matrix_result = matrix_c<T, n, m>;
		using row_vector = const_proxy_matrix_row<T, m, n>;
		using col_vector = proxy_matrix_col<T, n, m>;

		matrix_result result;

		struct set_elements
		{
			matrix_result& res_;
			const matrix& M_;
			inline set_elements(matrix_result& res, const matrix& M)
				: res_(res), M_(M)
			{}
			inline void operator()(size_t res_row_idx)
			{
				math::array_operations<col_vector, row_vector, n - 1> op;
				col_vector col(res_, res_row_idx);
				op.bmap(set_val<T>(), col, M_.row(res_row_idx));
			}
		};

		math::For<0, m, true>().Do(set_elements(result, M));

		return result;
	}

	/**
	* Calculates covariance matrix
	*/
	template<class T, std::size_t N>
	matrix_c<T, N, N> cov
	(
		const vector<vector_c<T, N>>& vectors,
		const vector<T>& w,
		const vector_c<T, N>& v_mean = vector_c<T, N>(0.0)
	)
	{
		using matrix = matrix_c<T, N, N>;
		using vector_c = vector_c<T, N>;

		assert(vectors.size() == w.size());

		matrix covMatrix(0.0);

		T total = 0.0;
		typename vector<T>::const_iterator it = w.begin();
		for (const vector_c& v : vectors)
		{
			total += *it;
			vector_c vv = v - v_mean;

			math::For<0, N, true>().Do([&covMatrix, vv, it](size_t row_idx)->void
			{
				covMatrix[row_idx][row_idx] += *it * vv[row_idx] * vv[row_idx];
				for (size_t col_idx = 0; col_idx < row_idx; ++col_idx)
				{
					covMatrix[row_idx][col_idx] += *it * vv[row_idx] * vv[col_idx];
					covMatrix[col_idx][row_idx] = covMatrix[row_idx][col_idx];
				}
			});
		}

		return (covMatrix / total) *
			static_cast<double>(w.size())
			/ static_cast<double>(w.size() - 1);
	}

	/**
	* Calculates first principal component
	*/
	template<class T, size_t N>
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
	template<class T, size_t N>
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
	template<class T, size_t N, size_t VarNum>
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
	}

#undef DEF_MATRIX_TEMPLATE_INLINE
#undef DEF_MATRIX_TEMPLATE
#undef DEF_MATRIX_TEMPLATE_PARAMS

MATH_NAMESPACE_END

#endif // MATRIXTEMPLATE_H

