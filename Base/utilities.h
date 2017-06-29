#pragma once
#ifndef _UTILITIES_
#define _UTILITIES_

#include "myLibDefs.h"
#include <utility>
#include <stdexcept>

UTIL_NAMESPACE_BEG

template<typename T> 
struct MayBe : public std::pair<T, bool>
{
	using baseType = std::pair<T, bool>;

	MayBe() : baseType(T(), false) {}

	MayBe(const T& val, bool f) : baseType(val, f) {}

	operator T&()
	{
		if (baseType::second) return baseType::first;
		throw std::logic_error("class MayBe: Wrapped value is not valid.");
	}

	operator const T&() const
	{
		if (baseType::second) return baseType::first;
		throw std::logic_error("class MayBe: Wrapped value is not valid.");
	}
	
	bool flag() const { return baseType::second; }
	void flag(bool f) { baseType::second = f; }
};

UTIL_NAMESPACE_END

#endif // !_UTILITIES_
