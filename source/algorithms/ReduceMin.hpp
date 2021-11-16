#pragma once

#include "../interfaces/AlgorithmInterface.hpp"

#include <tuple>

template<typename T_input, typename T_output>
class ReduceMin : public AlgorithmInterface<std::tuple<T_input, T_output>, T_output> {
public:
	ReduceMin() = default;

	ReduceMin(ReduceMin& other) = default;
	ReduceMin(ReduceMin&& other) = default;

	ReduceMin& operator=(const ReduceMin& other) = default;
	ReduceMin& operator=(ReduceMin&& other) = default;

	virtual ~ReduceMin() = default;

	T_output Compute(std::tuple<T_input, T_output>&& value) const override {
		return std::min(std::get<0>(value), std::get<1>(value));
	}

	std::string Name() const override {
		return std::string("reduce_min");
	}
};
