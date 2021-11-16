#pragma once

#include "../Commons.hpp"
#include "AlgorithmInterface.hpp"
#include "PatternInterface.hpp"

#include <future>
#include <memory>

template<typename T_input, typename T_output>
class AlgorithmWrapper : public PatternInterface<T_input, T_output> {
	AlgoIntPtr<T_input, T_output> interface;

	void InternallyCompute(std::future<T_input> future, std::promise<T_output> promise) override {
		auto input = future.get();
		auto result = interface->Compute(std::move(input));
		promise.set_value(std::move(result));
	}

	T_output InternallyComputePure(T_input&& input) override {
		return interface->Compute(std::move(input));
	}

	explicit AlgorithmWrapper(AlgoIntPtr<T_input, T_output> i) : interface(i) {}

public:
	static std::shared_ptr<PatternInterface<T_input, T_output>> create(AlgoIntPtr<T_input, T_output> i) {
		auto wrapper = new AlgorithmWrapper<T_input, T_output>(i);
		auto s_ptr = std::shared_ptr<PatternInterface<T_input, T_output>>(wrapper);
		return s_ptr;
	}

	AlgorithmWrapper(const AlgorithmWrapper<T_input, T_output>& other) = delete;
	AlgorithmWrapper(AlgorithmWrapper<T_input, T_output>&& other) = delete;

	AlgorithmWrapper& operator=(const AlgorithmWrapper<T_input, T_output>& other) = delete;
	AlgorithmWrapper& operator=(AlgorithmWrapper<T_input, T_output>&& other) = delete;

	virtual ~AlgorithmWrapper() = default;

	std::string Name() const override {
		return interface->Name();
	}

	size_t ThreadCount() const noexcept override {
		return 0;
	}

	PatIntPtr<T_input, T_output> create_copy() override {
		this->assertNoInit();

		auto copied_version = create(interface);
		return copied_version;
	}

	bool IsBlocking() const noexcept override {
		return true;
	}

	void Init() override {
		((void)(0));
	}

	void Dispose() override {
		((void)(0));
	}
};
