#pragma once

#include "../Commons.hpp"
#include "ThreadSafeQueue.hpp"

#include <cassert>
#include <future>
#include <string>
#include <memory>

#include "../interfaces/LoadBalancer.hpp"

template<typename T_input, typename T_output>
class PatternInterface : public LoadBalancingBase {
	template<typename T1, typename T2>
	friend class Executor;

	virtual void InternallyCompute(std::future<T_input>, std::promise<T_output>) = 0;

	virtual T_output InternallyComputePure(T_input&& input) {
		std::promise<T_input> promise_input;
		auto future_input = promise_input.get_future();
		promise_input.set_value(std::move(input));

		std::promise<T_output> promise_output;
		auto future_output = promise_output.get_future();

		InternallyCompute(std::move(future_input), std::move(promise_output));

		return future_output.get();
	}

protected:
	std::atomic<bool> initialized = ATOMIC_VAR_INIT(false);
	std::atomic<bool> dying = ATOMIC_VAR_INIT(false);

	void assertInit() const {
		if (!initialized) {
			assert(false && "Data structure is not initialized");
			throw std::runtime_error("Data structure is not initialized");
		}
	}

	void assertNoInit() const {
		if (initialized) {
			assert(false && "Data structure is initialized");
			throw std::runtime_error("Data structure is initialized");
		}
	}
public:
	PatternInterface() { }

	PatternInterface(const PatternInterface& other) = delete;
	PatternInterface(PatternInterface&&) = delete;

	PatternInterface& operator=(const PatternInterface& other) = delete;
	PatternInterface& operator=(PatternInterface&&) = delete;

	virtual PatIntPtr<T_input, T_output> create_copy() = 0;

	std::future<T_output> Compute(std::future<T_input> future) {
		std::promise<T_output> promise;
		auto result = promise.get_future();

		InternallyCompute(std::move(future), std::move(promise));

		return result;
	}

	T_output ComputePure(T_input&& input) {
		return InternallyComputePure(std::move(input));
	}

	virtual size_t ThreadCount() const noexcept = 0;

	virtual std::string Name() const = 0;

	virtual bool IsBlocking() const noexcept {
		return false;
	}

	virtual void Init() = 0;
	virtual void Dispose() = 0;

	virtual ~PatternInterface() = default;
};
