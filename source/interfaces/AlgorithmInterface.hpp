#pragma once

#include <string>

template<typename T_input, typename T_output>
class AlgorithmInterface {
public:
	AlgorithmInterface() = default;

	AlgorithmInterface(const AlgorithmInterface& other) = default;
	AlgorithmInterface(AlgorithmInterface&& other) = default;

	AlgorithmInterface& operator=(const AlgorithmInterface& other) = default;
	AlgorithmInterface& operator=(AlgorithmInterface&& other) = default;

	virtual ~AlgorithmInterface() = default;

	virtual T_output Compute(T_input&& input) const = 0;

	virtual std::string Name() const = 0;
};
