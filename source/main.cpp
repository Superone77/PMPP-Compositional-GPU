

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
#include "algorithms/DotPro_GPU.hpp"
#include "interfaces/AlgorithmWrapper.hpp"

#include "pattern/TaskPool.hpp"
#include "pattern/Pipeline.hpp"
#include "pattern/Composition.hpp"


#include "pattern/ForkJoin.hpp"
#include "helper/Comparator.hpp"

void testMax(PatIntPtr<std::vector<int>, int> s_ptr)
{
	auto create = [](int size) {
		auto vec = std::vector<int>(size);
		std::generate(vec.begin(), vec.end(), rand);
		return vec;
	};

	auto wrap = [](std::vector<int> vec) {
		std::promise<std::vector<int>> prom;
		prom.set_value(std::move(vec));
		return prom.get_future();
	};

	std::cout << "Testing: " << s_ptr->Name() << std::endl;

	const auto repeats = 100;
	const auto size = 10;

	std::vector<std::future<std::vector<int>>> inputs;
	std::vector<std::future<int>> outputs;
	std::vector<int> validation;

	for (auto i = 0; i < repeats; i++)
	{
		auto vec = create(size);
		auto maxValue = *max_element(vec.begin(),vec.end());
		validation.emplace_back(maxValue);
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
		if(val != validation[i]){
			std::cout<<"Find Max element false"<<std::endl;
			return;
		}
	}

	s_ptr->Dispose();
	std::cout<<"Test finding Max successfully"<<std::endl;
}

void testMin(PatIntPtr<std::vector<int>, int> s_ptr)
{
	auto create = [](int size) {
		auto vec = std::vector<int>(size);
		std::generate(vec.begin(), vec.end(), rand);
		return vec;
	};

	auto wrap = [](std::vector<int> vec) {
		std::promise<std::vector<int>> prom;
		prom.set_value(std::move(vec));
		return prom.get_future();
	};

	std::cout << "Testing: " << s_ptr->Name() << std::endl;

	const auto repeats = 100;
	const auto size = 10;

	std::vector<std::future<std::vector<int>>> inputs;
	std::vector<std::future<int>> outputs;
	std::vector<int> validation;

	for (auto i = 0; i < repeats; i++)
	{
		auto vec = create(size);
		auto maxValue = *min_element(vec.begin(),vec.end());
		validation.emplace_back(maxValue);
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
		if(val != validation[i]){
			std::cout<<"Find Min element false"<<std::endl;
			return;
		}
	}

	s_ptr->Dispose();
	std::cout<<"Test finding Min successfully"<<std::endl;

}

void testNop(PatIntPtr<std::vector<int>, std::vector<int>> s_ptr)
{
	auto create = [](int size) {
		auto vec = std::vector<int>(size);
		std::generate(vec.begin(), vec.end(), rand);
		return vec;
	};

	auto wrap = [](std::vector<int> vec) {
		std::promise<std::vector<int>> prom;
		prom.set_value(std::move(vec));
		return prom.get_future();
	};

	std::cout << "Testing: " << s_ptr->Name() << std::endl;

	const auto repeats = 100;
	const auto size = 10;

	std::vector<std::future<std::vector<int>>> inputs;
	std::vector<std::future<std::vector<int>>> outputs;
	std::vector<int> validation;

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
	s_ptr->Dispose();
	std::cout<<"Test finding Nop successfully"<<std::endl;
}



int main(int argument_count, char **arguments)
{
	{
//		auto max = std::make_shared<DotPro_GPU<int>>();
//		auto max_w = AlgorithmWrapper<<std::vector<int>,std::vector<int>>, int>::create(max);
//
//		auto qs = std::make_shared<QuickSorter<int>>();
//		auto qs_w = AlgorithmWrapper<std::vector<int>, std::vector<int>>::create(qs);
//
//		const auto num_threads_1 = 1;
//		const auto num_threads_2 = 7;
//
//		// const auto num_threads_1 = 7;
//		// const auto num_threads_2 = template&OOP;
//
//		auto tp = TaskPool<std::vector<int>, std::vector<int>>::create(qs_w, num_threads_1);
//		auto tp2 = TaskPool<std::vector<int>, int>::create(max_w, num_threads_2);
//
//		auto comp = Composition<std::vector<int>, std::vector<int>, int>::create(tp, tp2);
//
////		test2(comp);
		auto max = std::make_shared<Max_GPU<int>>();
		auto max_w = AlgorithmWrapper<std::vector<int>, int>::create(max);
		auto min = std::make_shared<Min_GPU<int>>();
		auto min_w = AlgorithmWrapper<std::vector<int>, int>::create(min);
		const auto num_threads_1 = 2;
		auto tp_max = TaskPool<std::vector<int>, int>::create(max_w, num_threads_1);
		auto tp_min = TaskPool<std::vector<int>, int>::create(min_w, num_threads_1);
		testMax(tp_max);
		testMin(tp_min);
		auto nop = std::make_shared<Nop_GPU<int>>();
		auto nop_w = AlgorithmWrapper<std::vector<int>, std::vector<int>>::create(nop);
		auto tp_nop = TaskPool<std::vector<int>, std::vector<int>>::create(nop_w, num_threads_1);
		testNop(tp_nop);
//		auto dot = std::make_shared<DotPro_GPU<int>>();
//		auto dot_w = AlgorithmWrapper<std::vector<int>, int>::create(dot);
//		auto tp_dot = TaskPool<std::vector<int>, int>::create(dot_w, num_threads_1);
//		testDot(tp_dot);

	}
	std::cout << "Finished" << std::endl;

	return 0;
}
