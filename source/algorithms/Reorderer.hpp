#pragma once

#include "../Commons.hpp"
#include "../interfaces/AlgorithmInterface.hpp"

#include <cassert>
#include <vector>
#include <string>
#include <random>

template<typename T>
class Reorderer : public AlgorithmInterface<std::vector<T>, std::vector<T>> {
public:
	Reorderer() = default;	

	Reorderer(Reorderer& other) = default;
	Reorderer(Reorderer&& other) = default;

	Reorderer& operator=(const Reorderer& other) = default;
	Reorderer& operator=(Reorderer&& other) = default;

	virtual ~Reorderer() = default;

	std::vector<T> Compute(std::vector<T>&& vector) const override {
		size_t length = vector.size();
		assert(length > 0);

		std::mt19937 mt(rand());
		std::uniform_int_distribution<int> uid(0, (int)length - 1);

		for (size_t i = 0; i < length; i++) {
			auto rand1 = uid(mt);
			auto rand2 = uid(mt);

			std::swap(vector[rand1], vector[rand2]);
		}

		return std::move(vector);
	}

	std::string Name() const override {
		return std::string("reorderer");
	}
};
