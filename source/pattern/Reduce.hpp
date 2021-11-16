#pragma once

#include "../Commons.hpp"
#include "../interfaces/PatternInterface.hpp"
#include "../interfaces/ThreadSafeQueue.hpp"
#include "../interfaces/Executor.hpp"

#include "../interfaces/LoadBalancer.hpp"

#include <cassert>
#include <future>
#include <thread>
#include <tuple>
#include <vector>

template<typename T_input>
class Reduce : public PatternInterface<FutVec<T_input>, T_input> {
	std::vector<std::thread> threads;

	Executor<std::tuple<T_input, T_input>, T_input> executor;

	TSQueue<std::tuple<std::future<T_input>, std::future<T_input>, std::promise<T_input>>> inner_queue;

	friend class LoadBalancingCapabilities<Reduce>;

	void PerformTask()
	{
		while (!this->dying)
		{
			if (!doWork())
			{
				if (this->dying)
				{
					break;
				}

				if (this->load_balancer)
				{
					if (!this->load_balancer->provideWork())
					{
						std::this_thread::yield();
					}
				}
				else
				{
					std::this_thread::yield();
				}

				continue;
			}
		}
	}

	bool doWork()
	{
		std::tuple<std::future<T_input>, std::future<T_input>, std::promise<T_input>> data;
		bool success = this->inner_queue.try_pop(data);
		if (!success)
			return false;

		auto future_1 = std::move(std::get<0>(data));
		auto future_2 = std::move(std::get<1>(data));
		auto promise = std::move(std::get<2>(data));

		if (this->load_balancer)
			if (future_1.wait_for(std::chrono::seconds(0)) != std::future_status::ready || future_2.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
			{
				this->inner_queue.push(std::move(std::tuple<std::future<T_input>, std::future<T_input>, std::promise<T_input>>(std::move(future_1), std::move(future_2), std::move(promise))));
				return false;
			}

		std::promise<std::tuple<T_input, T_input>> wrapper_promise;
		wrapper_promise.set_value(std::make_tuple(future_1.get(), future_2.get()));

		auto wrapper_future = wrapper_promise.get_future();

		executor.Compute(std::move(wrapper_future), std::move(promise));
		return true;
	}

	Reduce(PatIntPtr<std::tuple<T_input, T_input>, T_input>& func, size_t thread_count) : threads(thread_count), executor(func, thread_count) { }

protected:
	void InternallyCompute(std::future<FutVec<T_input>> fut, std::promise<T_input> prom) override {
		auto values = fut.get();

		size_t count = values.size();
		assert(count > 0);

		int current_vec_index = 0;

		for (size_t i = 0; i < count - 1; i++) {
			std::promise<T_input> promise;
			std::future<T_input> future = promise.get_future();

			int first_index = current_vec_index++;
			int second_index = current_vec_index++;

			auto tup = std::make_tuple(std::move(values[first_index]), std::move(values[second_index]), std::move(promise));
			inner_queue.push(std::move(tup));

			values.emplace_back(std::move(future));
		}

		T_input final_value = values[current_vec_index].get();
		prom.set_value(final_value);
	}

public:
	static PatIntPtr<FutVec<T_input>, T_input> create(PatIntPtr<std::tuple<T_input, T_input>, T_input> func, size_t thread_count, std::shared_ptr<LoadBalancer> load_balancer) {
		assert(thread_count > 0);

		auto red = new Reduce(func, thread_count);
		auto s_ptr = PatIntPtr<FutVec<T_input>, T_input>(red);

		s_ptr->load_balancer = load_balancer;
		s_ptr->self = s_ptr;

		return s_ptr;
	}

	Reduce(Reduce& other) = delete;
	Reduce(Reduce&& other) = delete;

	Reduce& operator=(const Reduce& other) = delete;
	Reduce& operator=(Reduce&& other) = delete;

	PatIntPtr<FutVec<T_input>, T_input> create_copy() override {
		this->assertNoInit();

		return create(executor.GetTask(), threads.size(), this->load_balancer);
	}
	
	bool doTask() override {
		return doWork();
	}

	void Init() override {
		if (!this->initialized) {
			this->dying = false;

			if (this->load_balancer)
				LoadBalancer::registerLoadBalancer(this->self, this->load_balancer);

			executor.Init();

			for (size_t i = 0; i < threads.size(); i++) {
				threads[i] = std::thread(&Reduce::PerformTask, this);
			}

			this->initialized = true;
		}
	}

	void Dispose() override {
		if (this->initialized) {
			this->dying = true;
			
			if (this->load_balancer)
				LoadBalancer::deregisterLoadBalancer(this->self, this->load_balancer);

			for (std::thread& thread : threads) {
				if (thread.joinable()) {
					thread.join();
				}
			}

			executor.Dispose();

			this->initialized = false;
		}
	}

	size_t ThreadCount() const noexcept override {
		return threads.size() * executor.ThreadCount();
	}

	std::string Name() const override {
		return std::string("reduce(") + std::to_string(threads.size()) + std::string(",") + (executor.Name()) + std::string(")");
	}

	~Reduce() {
		Reduce<T_input>::Dispose();
	}
};

template<typename T_input, typename T_key>
class ReduceKeyed : public PatternInterface<std::tuple<FutVec<T_input>, T_key>, std::tuple<T_input, T_key>> {
	std::vector<std::thread> threads;

	Executor<std::tuple<T_input, T_input>, T_input> executor;

	TSQueue<std::tuple<std::future<T_input>, std::future<T_input>, std::promise<T_input>>> inner_queue;

	void PerformTask() {
		while (!this->dying) {
			
			if (!doWork()) {
				if (this->dying) {
					break;
				}

				if (this->load_balancer)
				{
					if (!this->load_balancer->provideWork())
					{
						std::this_thread::yield();
					}
				}
				else
				{
					std::this_thread::yield();
				}

				continue;
			}

		}
	}

	bool doWork()
	{
		std::tuple<std::future<T_input>, std::future<T_input>, std::promise<T_input>> data;
		bool success = this->inner_queue.try_pop(data);
		if (!success)
			return false;

		std::future<T_input> future_1 = std::move(std::get<0>(data));
		std::future<T_input> future_2 = std::move(std::get<1>(data));
		std::promise<T_input> promise = std::move(std::get<2>(data));

		if (this->load_balancer)
			if (future_1.wait_for(std::chrono::seconds(0)) != std::future_status::ready || future_2.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
			{
				this->inner_queue.push(std::move(std::tuple<std::future<T_input>, std::future<T_input>, std::promise<T_input>>(std::move(future_1), std::move(future_2), std::move(promise))));
				return false;
			}

		std::promise<std::tuple<T_input, T_input>> wrapper_promise;
		wrapper_promise.set_value(std::make_tuple(future_1.get(), future_2.get()));

		std::future<std::tuple<T_input, T_input>> wrapper_future = wrapper_promise.get_future();

		executor.Compute(std::move(wrapper_future), std::move(promise));
		return true;
	}

	ReduceKeyed(PatIntPtr<std::tuple<T_input, T_input>, T_input> func, size_t thread_count) : threads(thread_count), executor(func, thread_count) { }

protected:
	void InternallyCompute(std::future<std::tuple<FutVec<T_input>, T_key>> fut, std::promise<std::tuple<T_input, T_key>> prom) override {
		std::tuple<FutVec<T_input>, T_key > tuple = fut.get();

		FutVec<T_input> values = std::move(std::get<0>(tuple));
		T_key key = std::move(std::get<1>(tuple));

		size_t count = values.size();
		assert(count > 0);

		int current_vec_index = 0;

		for (size_t i = 0; i < count - 1; i++) {
			std::promise<T_input> promise;

			std::future<T_input> future = promise.get_future();

			int first_index = current_vec_index++;
			int second_index = current_vec_index++;

			inner_queue.push(std::make_tuple(std::move(values[first_index]), std::move(values[second_index]), std::move(promise)));

			values.emplace_back(std::move(future));
		}

		T_input final_value = values[current_vec_index].get();
		std::tuple<T_input, T_key> result_tuple = std::make_tuple(final_value, key);
		prom.set_value(result_tuple);
	}

public:
	static PatIntPtr<std::tuple<FutVec<T_input>, T_key>, std::tuple<T_input, T_key>> create(PatIntPtr<std::tuple<T_input, T_input>, T_input> func, size_t thread_count, std::shared_ptr<LoadBalancer> load_balancer = nullptr) {
		assert(thread_count > 0);

		auto red = new ReduceKeyed(func, thread_count);
		auto s_ptr = PatIntPtr<std::tuple<FutVec<T_input>, T_key>, std::tuple<T_input, T_key>>(red);

		s_ptr->load_balancer = load_balancer
		s_ptr->self = s_ptr;

		return s_ptr;
	}

	ReduceKeyed(ReduceKeyed& other) = delete;
	ReduceKeyed(ReduceKeyed&& other) = delete;

	ReduceKeyed& operator=(const ReduceKeyed& other) = delete;
	ReduceKeyed& operator=(ReduceKeyed&& other) = delete;

	PatIntPtr<std::tuple<FutVec<T_input>, T_key>, std::tuple<T_input, T_key>> create_copy() override {
		this->assertNoInit();

		return create(executor.GetTask(), threads.size(), this->load_balancer);
	}
	
	bool doTask() override {
		return doWork();
	}

	void Init() override {
		if (!this->initialized) {
			this->dying = false;

			if (this->load_balancer)
				LoadBalancer::registerLoadBalancer(this->self, this->load_balancer);

			executor.Init();

			for (size_t i = 0; i < threads.size(); i++) {
				threads[i] = std::thread(&ReduceKeyed::PerformTask, this);
			}

			this->initialized = true;
		}
	}

	void Dispose() override {
		if (this->initialized) {
			this->dying = true;

			if (this->load_balancer)
				LoadBalancer::deregisterLoadBalancer(this->self, this->load_balancer);

			for (std::thread& thread : threads) {
				if (thread.joinable()) {
					thread.join();
				}
			}

			executor.Dispose();

			this->initialized = false;
		}
	}

	size_t ThreadCount() const noexcept override {
		return threads.size() * executor.ThreadCount();
	}

	std::string Name() const override {
		return std::string("reduce_keyed(") + std::to_string(threads.size()) + std::string(",") + executor.Name() + std::string(")");
	}

	~ReduceKeyed() {
		ReduceKeyed<T_input, T_key>::Dispose();
	}
};


template<typename T_input>
class ReducePure : public PatternInterface<std::vector<T_input>, T_input> {
	std::vector<std::thread> threads;

	Executor<std::tuple<T_input, T_input>, T_input> executor;

	TSQueue<std::tuple<T_input, T_input, std::promise<T_input>>> inner_queue;

	void PerformTask() {
		while (!this->dying)
		{
			if (!doWork())
			{
				if (this->dying)
				{
					break;
				}

				if (this->load_balancer)
				{
					if (!this->load_balancer->provideWork())
					{
						std::this_thread::yield();
					}
				}
				else
				{
					std::this_thread::yield();
				}

				continue;
			}
		}
	}

	bool doWork()
	{
		std::tuple<T_input, T_input, std::promise<T_input>> data;
		bool success = this->inner_queue.try_pop(data);

		if (!success)
			return false;

		T_input value_1 = std::move(std::get<0>(data));
		T_input value_2 = std::move(std::get<1>(data));
		std::promise<T_input> promise = std::move(std::get<2>(data));

		std::promise<std::tuple<T_input, T_input>> wrapper_promise;
		wrapper_promise.set_value(std::make_tuple(value_1, value_2));

		std::future<std::tuple<T_input, T_input>> wrapper_future = wrapper_promise.get_future();

		executor.Compute(std::move(wrapper_future), std::move(promise));
		return true;
	}

	ReducePure(PatIntPtr<std::tuple<T_input, T_input>, T_input>& func, size_t thread_count) : threads(thread_count), executor(func, thread_count) {	}

protected:
	void InternallyCompute(std::future<std::vector<T_input>> fut, std::promise<T_input> prom) override {
		std::vector<T_input> values = fut.get();

		size_t count = values.size();
		assert(count > 0);

		int current_vec_index = 0;

		FutVec<T_input> intermediate_vector;
		int intermediate_vector_index = 0;

		for (size_t i = 0; i < count - 1; i++) {
			std::promise<T_input> promise;

			std::future<T_input> future = promise.get_future();

			int first_index = current_vec_index++;
			int second_index = current_vec_index++;

			if (values.size() == first_index) {
				values.emplace_back(intermediate_vector[intermediate_vector_index].get());
				intermediate_vector_index++;
			}

			if (values.size() == second_index) {
				values.emplace_back(intermediate_vector[intermediate_vector_index].get());
				intermediate_vector_index++;
			}

			T_input	first_element = std::move(values[first_index]);
			T_input	second_element = std::move(values[second_index]);

			inner_queue.push(std::make_tuple(std::move(first_element), std::move(second_element), std::move(promise)));

			intermediate_vector.emplace_back(std::move(future));
		}

		if (values.size() == current_vec_index) {
			values.emplace_back(intermediate_vector[intermediate_vector_index].get());
		}

		T_input final_value = values[current_vec_index];
		prom.set_value(final_value);
	}

public:
	static PatIntPtr<std::vector<T_input>, T_input> create(PatIntPtr<std::tuple<T_input, T_input>, T_input> func, size_t thread_count, std::shared_ptr<LoadBalancer> load_balancer = nullptr) {
		assert(thread_count > 0);

		auto red = new ReducePure(func, thread_count);
		auto s_ptr = PatIntPtr<std::vector<T_input>, T_input>(red);

		s_ptr->load_balancer = load_balancer;
		s_ptr->self = s_ptr;

		return s_ptr;
	}

	ReducePure(ReducePure& other) = delete;
	ReducePure(ReducePure&& other) = delete;

	ReducePure& operator=(const ReducePure& other) = delete;
	ReducePure& operator=(ReducePure&& other) = delete;

	PatIntPtr<std::vector<T_input>, T_input> create_copy() override {
		this->assertNoInit();

		return create(executor.GetTask(), threads.size(), this->load_balancer);
	}
	
	bool doTask() override {
		return doWork();
	}

	void Init() override {
		if (!this->initialized) {
			this->dying = false;

			if (this->load_balancer)
				LoadBalancer::registerLoadBalancer(this->self, this->load_balancer);

			executor.Init();

			for (size_t i = 0; i < threads.size(); i++) {
				threads[i] = std::thread(&ReducePure::PerformTask, this);
			}

			this->initialized = true;
		}
	}

	void Dispose() override {
		if (this->initialized) {
			this->dying = true;

			if (this->load_balancer)
				LoadBalancer::deregisterLoadBalancer(this->self, this->load_balancer);

			for (std::thread& thread : threads) {
				if (thread.joinable()) {
					thread.join();
				}
			}

			executor.Dispose();

			this->initialized = false;
		}
	}

	size_t ThreadCount() const noexcept override {
		return threads.size() * executor.ThreadCount();
	}

	std::string Name() const override {
		return std::string("reduce_pure(") + std::to_string(threads.size()) + std::string(",") + executor.Name() + std::string(")");
	}

	~ReducePure() {
		ReducePure<T_input>::Dispose();
	}
};

template<typename T_input, typename T_key>
class ReduceKeyedPure : public PatternInterface<std::tuple<std::vector<T_input>, T_key>, std::tuple<T_input, T_key>> {
	std::vector<std::thread> threads;

	Executor<std::tuple<T_input, T_input>, T_input> executor;

	TSQueue<std::tuple<T_input, T_input, std::promise<T_input>>> inner_queue;

	void PerformTask() {
		while (!this->dying)
		{
			if (!doWork())
			{
				if (this->dying)
				{
					break;
				}

				if (this->load_balancer)
				{
					if (!this->load_balancer->provideWork())
					{
						std::this_thread::yield();
					}
				}
				else
				{
					std::this_thread::yield();
				}

				continue;
			}
		}
	}

	bool doWork()
	{
		std::tuple<T_input, T_input, std::promise<T_input>> data;
		bool success = this->inner_queue.try_pop(data);
		if (!success)
			return false;

		T_input value_1 = std::move(std::get<0>(data));
		T_input value_2 = std::move(std::get<1>(data));
		std::promise<T_input> promise = std::move(std::get<2>(data));

		std::promise<std::tuple<T_input, T_input>> wrapper_promise;
		wrapper_promise.set_value(std::make_tuple(value_1, value_2));

		std::future<std::tuple<T_input, T_input>> wrapper_future = wrapper_promise.get_future();

		executor.Compute(std::move(wrapper_future), std::move(promise));
		return true;
	}

	ReduceKeyedPure(PatIntPtr<std::tuple<T_input, T_input>, T_input>& func, size_t thread_count) : threads(thread_count), executor(func, thread_count) { }

protected:
	void InternallyCompute(std::future<std::tuple<std::vector<T_input>, T_key>> fut, std::promise<std::tuple<T_input, T_key>> prom) override {
		std::tuple<std::vector<T_input>, T_key > tuple = fut.get();

		std::vector<T_input> values = std::move(std::get<0>(tuple));
		T_key key = std::move(std::get<1>(tuple));

		size_t count = values.size();
		assert(count > 0);

		int current_vec_index = 0;

		FutVec<T_input> intermediate_vector;
		int intermediate_vector_index = 0;

		for (size_t i = 0; i < count - 1; i++) {
			std::promise<T_input> promise;

			std::future<T_input> future = promise.get_future();

			int first_index = current_vec_index++;
			int second_index = current_vec_index++;

			if (values.size() == first_index) {
				values.emplace_back(intermediate_vector[intermediate_vector_index].get());
				intermediate_vector_index++;
			}

			if (values.size() == second_index) {
				values.emplace_back(intermediate_vector[intermediate_vector_index].get());
				intermediate_vector_index++;
			}

			T_input	first_element = std::move(values[first_index]);
			T_input	second_element = std::move(values[second_index]);

			inner_queue.push(std::make_tuple(std::move(first_element), std::move(second_element), std::move(promise)));

			intermediate_vector.emplace_back(std::move(future));
		}

		if (values.size() == current_vec_index) {
			values.emplace_back(intermediate_vector[intermediate_vector_index].get());
		}

		T_input final_value = values[current_vec_index];
		std::tuple<T_input, T_key> result_tuple = std::make_tuple(final_value, key);
		prom.set_value(result_tuple);
	}

public:
	static PatIntPtr<std::tuple<std::vector<T_input>, T_key>, std::tuple<T_input, T_key>> create(PatIntPtr<std::tuple<T_input, T_input>, T_input> func, size_t thread_count, std::shared_ptr<LoadBalancer> load_balancer = nullptr) {
		assert(thread_count > 0);

		auto red = new ReduceKeyedPure(func, thread_count);
		auto s_ptr = PatIntPtr<std::tuple<std::vector<T_input>, T_key>, std::tuple<T_input, T_key>>(red);

		s_ptr->load_balancer = load_balancer;
		s_ptr->self = s_ptr;

		return s_ptr;
	}

	ReduceKeyedPure(ReduceKeyedPure& other) = delete;
	ReduceKeyedPure(ReduceKeyedPure&& other) = delete;

	ReduceKeyedPure& operator=(const ReduceKeyedPure& other) = delete;
	ReduceKeyedPure& operator=(ReduceKeyedPure&& other) = delete;

	PatIntPtr<std::tuple<std::vector<T_input>, T_key>, std::tuple<T_input, T_key>> create_copy() override {
		this->assertNoInit();

		return create(executor.GetTask(), threads.size(), this->load_balancer);
	}
	
	bool doTask() override {
		return doWork();
	}

	void Init() override {
		if (!this->initialized) {
			this->dying = false;

			if (this->load_balancer)
				LoadBalancer::registerLoadBalancer(this->self, this->load_balancer);

			executor.Init();

			for (size_t i = 0; i < threads.size(); i++) {
				threads[i] = std::thread(&ReduceKeyedPure::PerformTask, this);
			}

			this->initialized = true;
		}
	}

	void Dispose() override {
		if (this->initialized) {
			this->dying = true;

			if (this->load_balancer)
				LoadBalancer::deregisterLoadBalancer(this->self, this->load_balancer);

			for (std::thread& thread : threads) {
				if (thread.joinable()) {
					thread.join();
				}
			}

			executor.Dispose();

			this->initialized = false;
		}
	}

	size_t ThreadCount() const noexcept override {
		return threads.size() * executor.ThreadCount();
	}

	std::string Name() const override {
		return std::string("reduce_pure_keyed(") + std::to_string(threads.size()) + std::string(",") + executor.Name() + std::string(")");
	}

	~ReduceKeyedPure() {
		ReduceKeyedPure<T_input, T_key>::Dispose();
	}
};
