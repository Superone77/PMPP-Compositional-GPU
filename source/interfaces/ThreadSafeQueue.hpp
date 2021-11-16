#pragma once


#include "../Commons.hpp"

#include "concurrentqueue-master/concurrentqueue.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <queue>


namespace Fabian_TS {
	template<typename T>
	class QueueInterface {
	public:
		QueueInterface() = default;

		QueueInterface(QueueInterface& other) = delete;
		QueueInterface(QueueInterface&& other) = delete;

		QueueInterface& operator=(const QueueInterface& other) = delete;
		QueueInterface& operator=(QueueInterface&& other) = delete;

		virtual ~QueueInterface() = default;

		virtual void push(T new_val) = 0;

		virtual bool try_pop(T& value) = 0;
	};

	template<typename T>
	class OneMutexStdQueue : public QueueInterface<T> {
	protected:
		std::mutex mut;
		std::queue<T> data;
		std::condition_variable condition;

	public:
		OneMutexStdQueue() = default;

		OneMutexStdQueue(OneMutexStdQueue& other) = delete;
		OneMutexStdQueue(OneMutexStdQueue&& other) = delete;

		OneMutexStdQueue& operator=(const OneMutexStdQueue& other) = delete;
		OneMutexStdQueue& operator=(OneMutexStdQueue&& other) = delete;

		virtual ~OneMutexStdQueue() = default;

		void push(T new_val) override {
			std::lock_guard<std::mutex> lock(mut);
			data.push(new_val);
			condition.notify_one();
		}

		bool try_pop(T& value) override {
			std::lock_guard<std::mutex> lock(mut);
			if (data.empty()) {
				return false;
			}

			value = std::move(data.front());
			data.pop();
			return true;
		}

		std::shared_ptr<T> try_pop() override {
			std::lock_guard<std::mutex> lock(mut);
			if (data.empty()) {
				return std::shared_ptr<T>();
			}

			std::shared_ptr<T> pointer(std::make_shared<T>(data.front()));
			data.pop();
			return pointer;
		}
	};

	template<typename T>
	class FineGrainedLocking : public QueueInterface<T> {
		struct node {
			std::shared_ptr<T> data;
			std::unique_ptr<node> next;
		};

		std::mutex head_mutex;
		std::unique_ptr<node> head;

		std::mutex tail_mutex;
		node* tail;

		std::condition_variable data_cond;

		node* get_tail() {
			std::lock_guard<std::mutex> tail_lock(tail_mutex);
			return tail;
		}

		std::unique_ptr<node> pop_head() {
			std::unique_ptr<node> old_head = std::move(head);
			head = std::move(old_head->next);
			return old_head;
		}

		std::unique_lock<std::mutex> wait_for_data() {
			std::unique_lock<std::mutex> head_lock(head_mutex);
			data_cond.wait(head_lock, [&] {return head.get() != get_tail(); });
			return std::move(head_lock);
		}

		std::unique_ptr<node> wait_pop_head() {
			auto head_lock(wait_for_data());
			return pop_head();
		}

		std::unique_ptr<node> wait_pop_head(T& value) {
			auto head_lock(wait_for_data());
			value = std::move(*head->data);
			return pop_head();
		}

		std::unique_ptr<node> try_pop_head() {
			std::lock_guard<std::mutex> head_lock(head_mutex);
			if (head.get() == get_tail()) {
				return std::unique_ptr<node>();
			}
			return pop_head();
		}

		std::unique_ptr<node> try_pop_head(T& value) {
			std::lock_guard<std::mutex> head_lock(head_mutex);
			if (head.get() == get_tail()) {
				return std::unique_ptr<node>();
			}
			value = std::move(*head->data);
			return pop_head();
		}

	public:
		FineGrainedLocking() : head(new node), tail(head.get()) {}

		FineGrainedLocking(FineGrainedLocking& other) = delete;
		FineGrainedLocking(FineGrainedLocking&& other) = delete;

		FineGrainedLocking& operator=(const FineGrainedLocking& other) = delete;
		FineGrainedLocking& operator=(FineGrainedLocking&& other) = delete;

		virtual ~FineGrainedLocking() = default;

		void push(T new_val) override {
			std::shared_ptr<T> new_data(std::make_shared<T>(std::move(new_val)));
			std::unique_ptr<node> p(new node);

			{
				std::lock_guard<std::mutex> tail_lock(tail_mutex);
				tail->data = new_data;
				node* const new_tail = p.get();
				tail->next = std::move(p);
				tail = new_tail;
			}

			data_cond.notify_one();
		}

		std::shared_ptr<T> try_pop() override {
			std::unique_ptr<node> old_head = try_pop_head();
			return old_head ? old_head->data : std::shared_ptr<T>();
		}

		bool try_pop(T& value) override {
			std::unique_ptr<node> const old_head = try_pop_head(value);
			return static_cast<bool>(old_head);
		}
	};

	template<typename T>
	class ReferenceCountingQueue :public QueueInterface<T> {

		struct node;

		struct counted_node_ptr {
		public:
			int external_count = 2;
			node* ptr = nullptr;
		};

		struct node_counter {
		public:
			unsigned int internal_counter;
			unsigned int external_counter;
		};

		struct node {
			std::atomic<T*> data;
			std::atomic<node_counter> count;

			counted_node_ptr next;

			node() {
				node_counter new_count;
				new_count.internal_counter = 0;
				new_count.external_counter = 2;
				count.store(new_count);

				next.ptr = nullptr;
				next.external_count = 0;

				data.store(nullptr);
			}

			void release_ref()
			{
				node_counter old_counter = count.load(std::memory_order_relaxed);
				node_counter new_counter;

				do {
					new_counter = old_counter;
					--new_counter.internal_counter;
				} while (!count.compare_exchange_strong(old_counter,
					new_counter, std::memory_order_acquire, std::memory_order_relaxed));

				if (!new_counter.internal_counter && !new_counter.external_counter) {
					delete this;
				}
			}
		};

		std::atomic<counted_node_ptr> head;
		std::atomic<counted_node_ptr> tail;

		static void increase_external_count(
			std::atomic<counted_node_ptr>& counter,
			counted_node_ptr& old_counter)
		{
			counted_node_ptr new_counter;
			do
			{
				new_counter = old_counter;
				++new_counter.external_count;
			} while (!counter.compare_exchange_strong(
				old_counter, new_counter,
				std::memory_order_acquire, std::memory_order_relaxed));
			old_counter.external_count = new_counter.external_count;
		}

		static void free_external_counter(counted_node_ptr& old_node_ptr)
		{
			node* const ptr = old_node_ptr.ptr;
			int const count_increase = old_node_ptr.external_count - 2;
			node_counter old_counter = ptr->count.load(std::memory_order_relaxed);

			node_counter new_counter;
			do {
				new_counter = old_counter;
				--new_counter.external_counter;
				new_counter.internal_counter += count_increase;

			} while (!ptr->count.compare_exchange_strong(old_counter,
				new_counter, std::memory_order_acquire, std::memory_order_relaxed));

			if (!new_counter.internal_counter && !new_counter.external_counter)
			{
				delete ptr;
			}
		}

	public:

		ReferenceCountingQueue() {
			counted_node_ptr cnp;
			cnp.external_count = 1;
			cnp.ptr = new node;
			cnp.ptr->data.store(nullptr);

			head.store(cnp);
			tail.store(head.load());
		}

		ReferenceCountingQueue(const ReferenceCountingQueue& other) = delete;
		ReferenceCountingQueue(ReferenceCountingQueue&& other) = delete;

		ReferenceCountingQueue& operator=(const ReferenceCountingQueue& other) = delete;
		ReferenceCountingQueue& operator=(ReferenceCountingQueue&& other) = delete;

		virtual ~ReferenceCountingQueue() {
			while (true)
			{
				counted_node_ptr old_head = head.load();
				node* old_ptr = old_head.ptr;

				if (old_ptr == nullptr) {
					break;
				}

				T* old_data = old_ptr->data;
				counted_node_ptr new_head = old_ptr->next;

				head.store(new_head);
				delete old_ptr;

				if (old_data != nullptr) {
					delete old_data;
				}
			}
		}

		void push(T new_val) override {
			std::unique_ptr<T> new_data(new T(std::move(new_val)));
			counted_node_ptr new_next;

			new_next.ptr = new node;
			new_next.external_count = 1;
			counted_node_ptr old_tail = tail.load();

			T* old_data = nullptr;
			while (true) {
				increase_external_count(tail, old_tail);

				if ((old_tail.ptr->data).compare_exchange_strong(old_data, new_data.get())) {
					old_tail.ptr->next = new_next;
					old_tail = tail.exchange(new_next);

					free_external_counter(old_tail);
					new_data.release();
					break;
				}

				old_tail.ptr->release_ref();
			}
		}

		bool try_pop(T& value) override {
			std::unique_ptr<T> u_ptr = pop();

			bool holds_value = static_cast<bool>(u_ptr);

			if (holds_value) {
				value = std::move(*(u_ptr.get()));
			}

			return holds_value;
		}

		std::shared_ptr<T> try_pop() override {
			std::unique_ptr<T> u_ptr = pop();

			bool holds_value = static_cast<bool>(u_ptr);

			if (holds_value) {
				return std::make_shared<T>(std::move(*(u_ptr.get())));
			}

			return std::shared_ptr<T>();
		}

		std::unique_ptr<T> pop()
		{
			counted_node_ptr old_head = head.load(std::memory_order_relaxed);
			while (true)
			{
				increase_external_count(head, old_head);
				node* const ptr = old_head.ptr;
				if (ptr == tail.load().ptr)
				{
					ptr->release_ref();
					return std::unique_ptr<T>();
				}
				if (head.compare_exchange_strong(old_head, ptr->next))
				{
					T* const res = ptr->data.exchange(nullptr);
					free_external_counter(old_head);
					return std::unique_ptr<T>(res);
				}
				ptr->release_ref();
			}
		}

	};

	template<typename T>
	class RefCountingCameronQueue :public QueueInterface<T> {
		moodycamel::ConcurrentQueue<T> queue;

	public:
		void push(T new_val) override {
			queue.enqueue(std::move(new_val));
		}

		bool try_pop(T& value) override { 
			return queue.try_dequeue(value);
		}
	};
}

template<typename T>
using TSQueue = Fabian_TS::RefCountingCameronQueue<T>;

