#pragma once

#include "../Commons.hpp"
#include "../interfaces/PatternInterface.hpp"
#include "../interfaces/ThreadSafeQueue.hpp"
#include "../interfaces/Executor.hpp"

#include "../interfaces/LoadBalancer.hpp"


#include <future>
#include <thread>

template<typename T_input, typename T_intermediate, typename T_output>
class Pipeline : public PatternInterface<T_input, T_output> {
	Executor<T_input, T_intermediate> executor1;
	Executor<T_intermediate, T_output> executor2;

	std::thread stage1_thread;
	std::thread stage2_thread;

	TSQueue<std::tuple<std::future<T_input>, std::promise<T_intermediate>>> first_queue;
	TSQueue<std::tuple<std::future<T_intermediate>, std::promise<T_output>>> second_queue;

	void PerformFirstStage()
	{
		while (!this->dying)
		{
			if (!doWork_FirstStage())
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

	void PerformSecondStage() {
		while (!this->dying) {
			if (!doWork_SecondStage()) {
				if (this->dying) {
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

	bool doWork_FirstStage() {
		std::tuple<std::future<T_input>, std::promise<T_intermediate>> data;
		bool success = this->first_queue.try_pop(data);

		if (!success)
			return false;

		auto future = std::move(std::get<0>(data));
		auto promise = std::move(std::get<1>(data));

		if (this->load_balancer)
			if (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
			{
				this->first_queue.push(std::move(std::tuple<std::future<T_input>, std::promise<T_output>>(std::move(future), std::move(promise))));
				return false;
			}

		executor1.Compute(std::move(future), std::move(promise));

		return true;
	}

	bool doWork_SecondStage() {
		std::tuple<std::future<T_intermediate>, std::promise<T_output>> data;
		bool success = this->second_queue.try_pop(data);

		if (!success)
			return false;

		auto future = std::move(std::get<0>(data));
		auto promise = std::move(std::get<1>(data));

		if (this->load_balancer)
			if (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
			{
				this->second_queue.push(std::move(std::tuple<std::future<T_input>, std::promise<T_output>>(std::move(future), std::move(promise))));
				return false;
			}


		executor2.Compute(std::move(future), std::move(promise));

		return true;
	}

	bool doWork()
	{
		if (doWork_FirstStage())
			return true;

		if (doWork_SecondStage())
			return true;

		return false;
	}

	Pipeline(PatIntPtr<T_input, T_intermediate> interface1, PatIntPtr<T_intermediate, T_output> interface2)
		: executor1(interface1), executor2(interface2) { }

protected:
	void InternallyCompute(std::future<T_input> future, std::promise<T_output> promise) override {
		std::promise<T_intermediate> intermediate_promise;
		std::future<T_intermediate> intermediate_future = intermediate_promise.get_future();

		first_queue.push(std::make_tuple(std::move(future), std::move(intermediate_promise)));
		second_queue.push(std::make_tuple(std::move(intermediate_future), std::move(promise)));
	}

public:
	static PatIntPtr<T_input, T_output> create
		(PatIntPtr<T_input, T_intermediate> interface1, PatIntPtr<T_intermediate, T_output> interface2, std::shared_ptr<LoadBalancer> load_balancer = nullptr) {
		auto pipe = new Pipeline(interface1, interface2);
		auto s_ptr = std::shared_ptr<PatternInterface<T_input, T_output>>(pipe);

		s_ptr->load_balancer = load_balancer;
		s_ptr->self = s_ptr;

		return s_ptr;
	}

	Pipeline(Pipeline& other) = delete;
	Pipeline(Pipeline&& other) = delete;

	Pipeline& operator=(const Pipeline& other) = delete;
	Pipeline& operator=(Pipeline&& other) = delete;

	PatIntPtr<T_input, T_output> create_copy() override {
		this->assertNoInit();

		return create(executor1.GetTask(), executor2.GetTask(), this->load_balancer);
	}
	
	bool doTask() override {
		return doWork();
	}

	size_t ThreadCount() const noexcept override {
		return executor1.ThreadCount() + executor2.ThreadCount() + 2;
	}

	std::string Name() const override {
		return std::string("pipeline(") + executor1.Name() + std::string(",") + executor2.Name() + std::string(")");
	}

	void Init() override {
		if (!this->initialized) {
			this->dying = false;

			if (this->load_balancer)
				LoadBalancer::registerLoadBalancer(this->self, this->load_balancer);

			executor1.Init();
			executor2.Init();

			stage1_thread = std::thread(&Pipeline::PerformFirstStage, this);
			stage2_thread = std::thread(&Pipeline::PerformSecondStage, this);

			this->initialized = true;
		}
	}

	void Dispose() override {
		if (this->initialized) {
			this->dying = true;

			if (this->load_balancer)
				LoadBalancer::deregisterLoadBalancer(this->self, this->load_balancer);
				
			stage1_thread.join();
			stage2_thread.join();

			executor1.Dispose();
			executor2.Dispose();

			this->initialized = false;
		}
	}

	~Pipeline() {
		Pipeline<T_input, T_intermediate, T_output>::Dispose();
	}
};
