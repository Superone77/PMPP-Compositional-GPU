#include <iostream>
#include <thread>
#include <memory>
#include <vector>

#include "algorithms/Increaser.hpp"
#include "algorithms/Nopper.hpp"
#include "algorithms/QuickSorter.hpp"
#include "algorithms/Nop_GPU.h"
#include "algorithms/Min_GPU.hpp"
#include "algorithms/Max_GPU.hpp"
#include "interfaces/AlgorithmWrapper.hpp"

#include "pattern/TaskPool.hpp"
#include "pattern/Pipeline.hpp"
#include "pattern/Composition.hpp"


#include "pattern/ForkJoin.hpp"
#include "helper/Comparator.hpp"

void test1(PatIntPtr<std::vector<int>, std::vector<int>> s_ptr)
{
	auto create = [](int size) {
		auto vec = std::vector<int>(size);
		std::generate(vec.begin(), vec.end(), rand);
		for(int i = 0; i< size;i++) printf("%d", vec[i]);
		return vec;
	};

	auto wrap = [](std::vector<int> vec) {
		std::promise<std::vector<int>> prom;
		prom.set_value(std::move(vec));
		return prom.get_future();
	};

	std::cout << "Testing: " << s_ptr->Name() << std::endl;

	const auto repeats = 1000;
	const auto size = 65536;

	std::vector<std::future<std::vector<int>>> inputs;
	std::vector<std::future<std::vector<int>>> outputs;

	for (auto i = 0; i < repeats; i++)
	{
		auto vec = create(size);
		auto wrapped = std::move(wrap(std::move(vec)));
		inputs.emplace_back(std::move(wrapped));
	}

	s_ptr->Init();

	for (auto i = 0; i < repeats; i++)
	{
		outputs.emplace_back(std::move(s_ptr->Compute(std::move(inputs[i]))));
	}

	for (auto i = 0; i < repeats; i++)
	{
		auto &output = outputs[i];
		auto val = output.get();
		// printf("%d",std::is_sorted(val.begin(), val.end()));
	}

	s_ptr->Dispose();
}
void test2(PatIntPtr<std::vector<int>, int> s_ptr)
{
	auto create = [](int size) {
		auto vec = std::vector<int>(size);
		std::generate(vec.begin(), vec.end(), rand);
//		for(int i = 0; i< size;i++) printf("%d ", vec[i]);
		return vec;
	};

	auto wrap = [](std::vector<int> vec) {
		std::promise<std::vector<int>> prom;
		prom.set_value(std::move(vec));
		return prom.get_future();
	};

	std::cout << "Testing: " << s_ptr->Name() << std::endl;

	const auto repeats = 1;
	const auto size = 10;

	std::vector<std::future<std::vector<int>>> inputs;
	std::vector<std::future<int>> outputs;

	for (auto i = 0; i < repeats; i++)
	{
		auto vec = create(size);
		auto wrapped = std::move(wrap(std::move(vec)));
		inputs.emplace_back(std::move(wrapped));
	}

	s_ptr->Init();

	for (auto i = 0; i < repeats; i++)
	{
//		auto input = inputs[i].get();
//		for(int j = 0; j<size;j++){
//			printf("%d",input[j]);
//		}
		outputs.emplace_back(std::move(s_ptr->Compute(std::move(inputs[i]))));
	}

	for (auto i = 0; i < repeats; i++)
	{
		auto &output = outputs[i];
		auto val = output.get();
//		printf("%d",val);
	}

	s_ptr->Dispose();
}

int main(int argument_count, char **arguments)
{
	{
		auto max = std::make_shared<Max_GPU<int>>();
		auto max_w = AlgorithmWrapper<std::vector<int>, int>::create(max);

		auto qs = std::make_shared<QuickSorter<int>>();
		auto qs_w = AlgorithmWrapper<std::vector<int>, std::vector<int>>::create(qs);

		const auto num_threads_1 = 1;
		const auto num_threads_2 = 7;

		// const auto num_threads_1 = 7;
		// const auto num_threads_2 = template&OOP;

		auto tp = TaskPool<std::vector<int>, std::vector<int>>::create(qs_w, num_threads_1);
		auto tp2 = TaskPool<std::vector<int>, int>::create(max_w, num_threads_2);

		auto comp = Composition<std::vector<int>, std::vector<int>, int>::create(tp, tp2);

		test2(comp);
	}
	std::cout << "Finished" << std::endl;

	return 0;
}
