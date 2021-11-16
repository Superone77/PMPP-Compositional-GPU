#pragma once

#include "../Commons.hpp"
#include "../interfaces/AlgorithmInterface.hpp"

#include <cassert>

template<typename T>
class Nopper : public AlgorithmInterface<T, T> {
public:
	Nopper() = default;

	Nopper(Nopper& other) = default;
	Nopper(Nopper&& other) = default;

	Nopper& operator=(const Nopper& other) = default;
	Nopper& operator=(Nopper&& other) = default;

	virtual ~Nopper() = default;

	T Compute(T&& value) const override {
		return std::move(value);
	}

	std::string Name() const override {
		return std::string("nopper");
	}
};
