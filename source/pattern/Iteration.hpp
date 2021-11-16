#pragma once

#include "../Commons.hpp"
#include "../interfaces/PatternInterface.hpp"
#include "../interfaces/Executor.hpp"

#include <cassert>
#include <future>
#include <vector>

template<typename T_input, typename T_output>
class Iteration : public PatternInterface<FutVec<T_input>, FutVec<T_output>> {
	Executor<T_input, T_output> executor;

	Iteration(PatIntPtr<T_input, T_output>& func) : executor(func) { }

protected:
	void InternallyCompute(std::future<FutVec<T_input>> futValue, std::promise<FutVec<T_output>> prom) override {
		FutVec<T_input> value = futValue.get();

		size_t count = value.size();
		assert(count > 0);

		FutVec<T_output> results(count);

		for (size_t i = 0; i < count; i++) {
			std::promise<T_output> promise;

			std::future<T_output> fut = promise.get_future();
			results[i] = std::move(fut);

			std::future<T_input> single_future = std::move(value[i]);
			executor.Compute(std::move(single_future), std::move(promise));
		}

		prom.set_value(std::move(results));
	}

public:
	static PatIntPtr<FutVec<T_input>, FutVec<T_output>> create(PatIntPtr<T_input, T_output> func) {
		auto it = new Iteration(func);
		auto s_ptr = std::shared_ptr<PatternInterface<FutVec<T_input>, FutVec<T_output>>>(it);
		return s_ptr;
	}

	Iteration(Iteration& other) = delete;
	Iteration(Iteration&& other) = delete;

	Iteration& operator=(const Iteration& other) = delete;
	Iteration& operator=(Iteration&& other) = delete;

	PatIntPtr<FutVec<T_input>, FutVec<T_output>> create_copy() override {
		this->assertNoInit();

		auto copied_version = create(executor.GetTask());
		return copied_version;
	}

	void Init() override {
		if (!this->initialized) {
			this->dying = false;

			executor.Init();

			this->initialized = true;
		}
	}

	void Dispose() override {
		if (this->initialized) {
			this->dying = true;

			executor.Dispose();

			this->initialized = false;
		}
	}

	size_t ThreadCount() const noexcept override {
		return executor.ThreadCount();
	}

	std::string Name() const override {
		return std::string("iteration(") + executor.Name() + std::string(")");
	}

	~Iteration() {
		Iteration<T_input, T_output>::Dispose();
	}
};

