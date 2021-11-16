#pragma once

#include "PatternInterface.hpp"

#include <atomic>
#include <vector>

template<typename T_input, typename T_output>
class Executor {
	std::vector<PatIntPtr<T_input, T_output>> patterns;

	std::atomic<unsigned long long int> round_robin_counter;
	bool holds_single;
	size_t size;

public:
	Executor(PatIntPtr<T_input, T_output> pattern, size_t count = 1) : round_robin_counter(0) {
		assert(count > 0 && "Have to repeat the pattern");

		auto pattern_is_blocking = pattern->IsBlocking();
		holds_single = count == 1 || pattern_is_blocking;

		if (holds_single) {
			patterns = { pattern->create_copy() };
			return;
		}

		patterns = std::vector<PatIntPtr<T_input, T_output>>(count);

		for (auto i = 0; i < count; i++) {
			patterns[i] = pattern->create_copy();
		}

		size = count;
	}

	Executor(const Executor&) = delete;
	Executor& operator=(const Executor&) = delete;

	Executor(Executor&&) = default;
	Executor& operator=(Executor&&) = default;

	void Compute(std::future<T_input> future, std::promise<T_output> promise) {
		auto mod_index = 0ull;

		if (!holds_single) {
			auto index = round_robin_counter.fetch_add(1, std::memory_order::memory_order_relaxed);
			mod_index = index % size;
		}

		auto& pattern = patterns[mod_index];
		pattern->InternallyCompute(std::move(future), std::move(promise));
	}

	std::future<T_output> Compute(std::future<T_input> future) {
		auto mod_index = 0ull;

		if (!holds_single) {
			auto index = round_robin_counter.fetch_add(1, std::memory_order::memory_order_relaxed);
			mod_index = index % size;
		}

		auto& pattern = patterns[mod_index];
		return pattern->Compute(std::move(future));
	}

	T_output Compute(T_input&& input) {
		auto mod_index = 0ull;

		if (!holds_single) {
			auto index = round_robin_counter.fetch_add(1);
			mod_index = index % size;
		}

		auto& pattern = patterns[mod_index];
		return pattern->InternallyComputePure(std::move(input));
	}

	PatIntPtr<T_input, T_output> GetTask(size_t index = 0) {
		if (index < patterns.size()) {
			return patterns[index];
		}

		return PatIntPtr<T_input, T_output>();
	}

	void Init() {
		for (auto iterator = patterns.begin(); iterator != patterns.end(); ++iterator) {
			(*iterator)->Init();
		}
	}

	void Dispose() {
		for (auto iterator = patterns.begin(); iterator != patterns.end(); ++iterator) {
			(*iterator)->Dispose();
		}
	}

	std::string Name() const {
		return patterns[0]->Name();
	}

	size_t ThreadCount() const noexcept {
		return patterns[0]->ThreadCount();
	}
};
