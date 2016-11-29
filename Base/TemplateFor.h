#pragma once
#ifndef ARRAY_OPERATIONS_H 
#define ARRAY_OPERATIONS_H

#include <cstddef>
#include "myLibDefs.h"

BASE_NAMESPACE_BEG

/**
 * Templates implementing recursive for-loop
 */
template<size_t begin, size_t end>
struct For
{
	using next = For<begin + 1, end>;
	
#define DEF_CHECK_ENDS \
	static_assert(begin < end, "Start value is bigger than the finish value in a template for loop.");

	template<typename O, typename ... T>
	static inline void Do(O Op, T ... args)
	{
		DEF_CHECK_ENDS
		Op(begin, args...);
		next::Do(Op, args...);
	}
#undef DEF_CHECK_ENDS
};
/**
 * If begin == end then stop the loop
 */
template<size_t stop>
struct For<stop, stop>
{
	template<typename O, typename ... T>
	static inline void Do(O, T ... ){}
};

BASE_NAMESPACE_END

#endif // ARRAY_OPERATIONS_H 
