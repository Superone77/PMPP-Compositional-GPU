#pragma once

#include "../interfaces/AlgorithmInterface.hpp"

#include <tuple>
#include <vector>

template<typename T_input, typename T_output>
class ReduceAdd : public AlgorithmInterface<std::tuple<T_input, T_output>, T_output> {
public:
	ReduceAdd() = default;

	ReduceAdd(ReduceAdd& other) = default;
	ReduceAdd(ReduceAdd&& other) = default;

	ReduceAdd& operator=(const ReduceAdd& other) = default;
	ReduceAdd& operator=(ReduceAdd&& other) = default;

	virtual ~ReduceAdd() = default;

	T_output Compute(std::tuple<T_input, T_output>&& value) const override {
		return std::get<0>(value) + std::get<1>(value);
	}

	std::string Name() const override {
		return std::string("reduce_add");
	}
};

template<typename T_input>
class ReduceAddVector : public AlgorithmInterface<std::vector<T_input>, T_input> {
public:
	ReduceAddVector() = default;

	ReduceAddVector(ReduceAddVector& other) = default;
	ReduceAddVector(ReduceAddVector&& other) = default;

	ReduceAddVector& operator=(const ReduceAddVector& other) = default;
	ReduceAddVector& operator=(ReduceAddVector&& other) = default;

	virtual ~ReduceAddVector() = default;

	T_input Compute(std::vector<T_input>&& value) const override {
		assert(value.size() > 0);

		T_input result = value[0];

		for (auto j = 0; j < 100; j++) {
			for (auto i = 1; i < value.size(); i++) {
				result += value[i];
			}
		}

		return result;
	}

	std::string Name() const override {
		return std::string("reduce_add_vector");
	}
};
