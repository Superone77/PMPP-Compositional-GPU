#pragma once

#include "../Commons.hpp"
#include "../interfaces/PatternInterface.hpp"
#include "../interfaces/Executor.hpp"

template<typename T_input, typename T_intermediate, typename T_output>
class Composition : public PatternInterface<T_input, T_output> {
	Executor<T_input, T_intermediate> executor1;
	Executor<T_intermediate, T_output> executor2;

	Composition(PatIntPtr<T_input, T_intermediate>& interface1, PatIntPtr<T_intermediate, T_output>& interface2)
		: executor1(interface1), executor2(interface2) { }

protected:
	void InternallyCompute(std::future<T_input> future, std::promise<T_output> promise) override {
		std::promise<T_intermediate> intermediate_promise;
		std::future<T_intermediate> intermediate_future = intermediate_promise.get_future();

		executor1.Compute(std::move(future), std::move(intermediate_promise));
		executor2.Compute(std::move(intermediate_future), std::move(promise));
	}

public:
	static PatIntPtr<T_input, T_output> create
	(PatIntPtr<T_input, T_intermediate> interface1, PatIntPtr<T_intermediate, T_output> interface2) {
		auto comp = new Composition(interface1, interface2);
		auto s_ptr = std::shared_ptr<PatternInterface<T_input, T_output>>(comp);
		return s_ptr;
	}

	Composition(const Composition& other) = delete;
	Composition(Composition&& other) = delete;

	Composition& operator=(const Composition& other) = delete;
	Composition& operator=(Composition&& other) = delete;

	PatIntPtr<T_input, T_output> create_copy() override {
		this->assertNoInit();

		auto copied_version = create(executor1.GetTask(), executor2.GetTask());
		return copied_version;
	}

	void Init() override {
		if (!this->initialized) {
			this->dying = false;

			executor1.Init();
			executor2.Init();

			this->initialized = true;
		}
	}

	void Dispose() override {
		if (this->initialized) {
			this->dying = true;

			executor1.Dispose();
			executor2.Dispose();

			this->initialized = false;
		}
	}

	size_t ThreadCount() const noexcept override {
		return executor1.ThreadCount() + executor2.ThreadCount();
	}

	std::string Name() const override {
		return std::string("composition(") + executor1.Name() + std::string(",") + executor2.Name() + std::string(")");
	}

	~Composition() {
		Composition<T_input, T_intermediate, T_output>::Dispose();
	}
};
