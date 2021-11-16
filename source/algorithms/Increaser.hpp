#pragma once

#include "../Commons.hpp"
#include "../interfaces/AlgorithmInterface.hpp"

#include <cassert>
#include <vector>
#include <string>

template<typename T>
class Increaser : public AlgorithmInterface<std::vector<T>, std::vector<T>> {
public:
	Increaser() = default;

	Increaser(Increaser& other) = default;
	Increaser(Increaser&& other) = default;

	Increaser& operator=(const Increaser& other) = default;
	Increaser& operator=(Increaser&& other) = default;

	virtual ~Increaser() = default;

	std::vector<T> Compute(std::vector<T>&& vector) const override {
		size_t length = vector.size();
		assert(length > 0);

		for (size_t i = 0; i < length; i++) {
			vector[i] = vector[i] + 1;
		}

		return std::move(vector);
	}

	std::string Name() const override {
		return std::string("increaser");
	}
};
