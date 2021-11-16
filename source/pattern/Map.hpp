#pragma once

#include "../Commons.hpp"
#include "../interfaces/PatternInterface.hpp"
#include "../interfaces/ThreadSafeQueue.hpp"
#include "../interfaces/Executor.hpp"

#include "../interfaces/LoadBalancer.hpp"

#include <cassert>
#include <future>
#include <thread>
#include <vector>

template<typename T_input, typename T_output>
class Map : public PatternInterface<FutVec<T_input>, FutVec<T_output>> {
	std::vector<std::thread> threads;

	Executor<T_input, T_output> executor;

	TSQueue<std::tuple<std::future<T_input>, std::promise<T_output>>> inner_queue;

	friend class LoadBalancingCapabilities<Map>;

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

				if (this->load_balancer) {
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
		std::tuple<std::future<T_input>, std::promise<T_output>> data;
		auto success = this->inner_queue.try_pop(data);
		if (!success)
			return false;
			
		auto future = std::move(std::get<0>(data));
		auto promise = std::move(std::get<1>(data));

		if (this->load_balancer)
			if (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
			{
				this->inner_queue.push(std::move(std::tuple<std::future<T_input>, std::promise<T_output>>(std::move(future), std::move(promise))));
				return false;
			}

		executor.Compute(std::move(future), std::move(promise));
		return true;
	}

	Map(PatIntPtr<T_input, T_output> task, size_t thread_count) : threads(thread_count), executor(task, thread_count) { }

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
			inner_queue.push(std::make_tuple(std::move(single_future), std::move(promise)));
		}

		prom.set_value(std::move(results));
	}

public:
	static PatIntPtr<FutVec<T_input>, FutVec<T_output>> create(PatIntPtr<T_input, T_output> task, size_t thread_count, std::shared_ptr<LoadBalancer> load_balancer = nullptr) {
		assert(thread_count > 0);

		auto map = new Map(task, thread_count);
		auto s_ptr = std::shared_ptr<PatternInterface<FutVec<T_input>, FutVec<T_output>>>(map);

		s_ptr->self = s_ptr;
		s_ptr->load_balancer = load_balancer;
				
		return s_ptr;
	}

	Map(Map& other) = delete;
	Map(Map&& other) = delete;

	Map& operator=(const Map& other) = delete;
	Map& operator=(Map&& other) = delete;

	PatIntPtr<FutVec<T_input>, FutVec<T_output>> create_copy() override {
		this->assertNoInit();

		return create(executor.GetTask(), threads.size(), this->load_balancer);
	}
	
	bool doTask() override {
		return doWork();
	}

	size_t ThreadCount() const noexcept override {
		return threads.size() * executor.ThreadCount();
	}

	std::string Name() const override {
		return std::string("map(") + std::to_string(threads.size()) + std::string(",") + executor.Name() + std::string(")");
	}

	void Init() override {
		if (!this->initialized) {
			this->dying = false;

			if (this->load_balancer)
				LoadBalancer::registerLoadBalancer(this->self, this->load_balancer);

			executor.Init();

			for (size_t i = 0; i < threads.size(); i++) {
				threads[i] = std::thread(&Map::PerformTask, this);
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

	~Map() {
		Map<T_input, T_output>::Dispose();
	}
};
