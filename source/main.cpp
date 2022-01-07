

#include <iostream>
#include <thread>
#include <memory>
#include <vector>
#include <numeric>
#include <chrono>

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

using namespace std::chrono;
const int test_repeats = 100;
const int repeats = 10000;
const int size = 100;//size must equal to N in gpuCommon.cu

int randNum() {
    return 0 + rand() % 100;
}

void testMax(PatIntPtr<std::vector<int>, int> s_ptr) {
    auto create = [](int size) {
        auto vec = std::vector<int>(size);
        std::generate(vec.begin(), vec.end(), rand);
        return vec;
    };

    auto wrap = [](std::vector<int> vec) {
        std::promise <std::vector<int>> prom;
        prom.set_value(std::move(vec));
        return prom.get_future();
    };

    std::cout << "Testing: " << s_ptr->Name() << std::endl;

    std::vector < std::future < std::vector < int>>> inputs;
    std::vector <std::future<int>> outputs;
    std::vector<int> validation;

    for (auto i = 0; i < test_repeats; i++) {
        auto vec = create(size);
        auto maxValue = *max_element(vec.begin(), vec.end());
        validation.emplace_back(maxValue);
        auto wrapped = std::move(wrap(std::move(vec)));
        inputs.emplace_back(std::move(wrapped));
    }

    s_ptr->Init();

    for (auto i = 0; i < test_repeats; i++) {
        outputs.emplace_back(std::move(s_ptr->Compute(std::move(inputs[i]))));
    }

    for (auto i = 0; i < test_repeats; i++) {
        auto &output = outputs[i];
        auto val = output.get();
        if (val != validation[i]) {
            std::cout << "Find Max element false" << std::endl;
            return;
        }
    }

    s_ptr->Dispose();
    std::cout << "Test finding Max successfully" << std::endl;
}

void testMin(PatIntPtr<std::vector<int>, int> s_ptr) {
    auto create = [](int size) {
        auto vec = std::vector<int>(size);
        std::generate(vec.begin(), vec.end(), rand);
        return vec;
    };

    auto wrap = [](std::vector<int> vec) {
        std::promise <std::vector<int>> prom;
        prom.set_value(std::move(vec));
        return prom.get_future();
    };

    std::cout << "Testing: " << s_ptr->Name() << std::endl;


    std::vector < std::future < std::vector < int>>> inputs;
    std::vector <std::future<int>> outputs;
    std::vector<int> validation;

    for (auto i = 0; i < test_repeats; i++) {
        auto vec = create(size);
        auto maxValue = *min_element(vec.begin(), vec.end());
        validation.emplace_back(maxValue);
        auto wrapped = std::move(wrap(std::move(vec)));
        inputs.emplace_back(std::move(wrapped));
    }

    s_ptr->Init();

    for (auto i = 0; i < test_repeats; i++) {
        outputs.emplace_back(std::move(s_ptr->Compute(std::move(inputs[i]))));
    }

    for (auto i = 0; i < test_repeats; i++) {
        auto &output = outputs[i];
        auto val = output.get();
        if (val != validation[i]) {
            std::cout << "Find Min element false" << std::endl;
            return;
        }
    }

    s_ptr->Dispose();
    std::cout << "Test finding Min successfully" << std::endl;

}

void testNop(PatIntPtr<int, int> s_ptr) {

    auto wrap = [](int num) {
        std::promise<int> prom;
        prom.set_value(std::move(num));
        return prom.get_future();
    };

    std::cout << "Testing: " << s_ptr->Name() << std::endl;


    std::vector <std::future<int>> inputs;
    std::vector <std::future<int>> outputs;

    for (auto i = 0; i < test_repeats; i++) {
        auto num = rand();
        auto wrapped = std::move(wrap(std::move(num)));
        inputs.emplace_back(std::move(wrapped));
    }

    s_ptr->Init();
    for (auto i = 0; i < test_repeats; i++) {
        outputs.emplace_back(std::move(s_ptr->Compute(std::move(inputs[i]))));
    }
    s_ptr->Dispose();
    std::cout << "Test finding Nop successfully" << std::endl;
}


void testDotPro(PatIntPtr<std::pair<std::vector<int>, std::vector<int>>, int> s_ptr)
{
	auto create = [](int size) {
		auto vec = std::vector<int>(size);
		std::generate(vec.begin(), vec.end(), randNum);
		return vec;
	};

	auto wrap = [](std::pair<std::vector<int>, std::vector<int>> pair) {
		std::promise<std::pair<std::vector<int>, std::vector<int>>> prom;
		prom.set_value(std::move(pair));
		return prom.get_future();
	};

	std::cout << "Testing: " << s_ptr->Name() << std::endl;


	std::vector<std::future<std::pair<std::vector<int>, std::vector<int>>>> inputs;
	std::vector<std::future<int>> outputs;
	std::vector<int> validation;

	for (auto i = 0; i < test_repeats; i++)
	{
		auto vec1 = create(size);
//		for (auto i : vec1)
//			std::cout << i << ' ';
//		std::cout<<std::endl;
		auto vec2 = create(size);
//		for (auto i : vec2)
//			std::cout << i << ' ';
//		std::cout<<std::endl;
		int dot_product = inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0);
		validation.emplace_back(dot_product);
		std::pair<std::vector<int>, std::vector<int>> pair = {vec1, vec2};
		auto wrapped = std::move(wrap(std::move(pair)));
		inputs.emplace_back(std::move(wrapped));
	}

	s_ptr->Init();

	for (auto i = 0; i < test_repeats; i++)
	{
		outputs.emplace_back(std::move(s_ptr->Compute(std::move(inputs[i]))));
	}

	for (auto i = 0; i < test_repeats; i++)
	{
		auto &output = outputs[i];
		auto val = output.get();
		if(val != validation[i]){
			std::cout<<"In "<< i<<". test "<<val<<" and "<<validation[i]<<std::endl;
			std::cout<<"Scalar Product false"<<std::endl;
			return;
		}
	}

	s_ptr->Dispose();
	std::cout<<"Test scalar product successfully"<<std::endl;

}

void timer1(PatIntPtr<int, int> s_ptr) {

    auto wrap = [](int num) {
        std::promise<int> prom;
        prom.set_value(std::move(num));
        return prom.get_future();
    };

    std::cout << "Testing Time of : " << s_ptr->Name() << std::endl;
    std::cout << "repeats: " << repeats << std::endl;
    std::cout << "size: " << 1 << std::endl;


    std::vector <std::future<int>> inputs;
    std::vector <std::future<int>> outputs;

    for (auto i = 0; i < repeats; i++) {
        auto num = rand();
        auto wrapped = std::move(wrap(std::move(num)));
        inputs.emplace_back(std::move(wrapped));
    }
	auto start = system_clock::now();
    s_ptr->Init();
    for (auto i = 0; i < repeats; i++) {
        outputs.emplace_back(std::move(s_ptr->Compute(std::move(inputs[i]))));
    }
    s_ptr->Dispose();
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    std::cout << "average time of " << s_ptr->Name() << " is "
              << (double(duration.count()) * microseconds::period::num / microseconds::period::den) / repeats
              << " sec" << std::endl;
}

void timer2(PatIntPtr<std::vector<int>, int> s_ptr) {
    auto create = [](int size) {
        auto vec = std::vector<int>(size);
        std::generate(vec.begin(), vec.end(), rand);
        return vec;
    };

    auto wrap = [](std::vector<int> vec) {
        std::promise <std::vector<int>> prom;
        prom.set_value(std::move(vec));
        return prom.get_future();
    };

    std::cout << "Testing Time of : " << s_ptr->Name() << std::endl;
    std::cout << "repeats: " << repeats << std::endl;
    std::cout << "size: " << size << std::endl;



    std::vector < std::future < std::vector < int>>> inputs;
    std::vector <std::future<int>> outputs;

    for (auto i = 0; i < repeats; i++) {
        auto vec = create(size);
        auto wrapped = std::move(wrap(std::move(vec)));
        inputs.emplace_back(std::move(wrapped));
    }
	auto start = system_clock::now();
    s_ptr->Init();


    for (auto i = 0; i < repeats; i++) {
        outputs.emplace_back(std::move(s_ptr->Compute(std::move(inputs[i]))));
    }


    for (auto i = 0; i < repeats; i++) {
        auto &output = outputs[i];
        auto val = output.get();
    }

    s_ptr->Dispose();
	auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    std::cout << "average time of " << s_ptr->Name() << " is "
              << (double(duration.count()) * microseconds::period::num / microseconds::period::den) / repeats
              << " sec" << std::endl;
}

void timer3(PatIntPtr <std::pair<std::vector < int>, std::vector<int>>, int> s_ptr)
{
	auto create = [](int size) {
		auto vec = std::vector<int>(size);
		std::generate(vec.begin(), vec.end(), randNum);
		return vec;
	};

	auto wrap = [](std::pair <std::vector<int>, std::vector<int>> pair) {
		std::promise < std::pair < std::vector < int > , std::vector < int>>> prom;
		prom.set_value(std::move(pair));
		return prom.get_future();
	};

	std::cout << "Testing Time of : " << s_ptr->Name() << std::endl;
	std::cout << "repeats: " << repeats << std::endl;
	std::cout << "size: " << size << std::endl;




	std::vector <std::future<std::pair < std::vector < int>, std::vector<int>>>> inputs;
	std::vector <std::future<int>> outputs;

	for (auto i = 0;i<repeats;i++)
	{
		auto vec1 = create(size);
//		for (auto i : vec1)
//			std::cout << i << ' ';
//		std::cout<<std::endl;
		auto vec2 = create(size);
//		for (auto i : vec2)
//			std::cout << i << ' ';
//		std::cout<<std::endl;
		std::pair <std::vector<int>, std::vector<int>> pair = {vec1, vec2};
		auto wrapped = std::move(wrap(std::move(pair)));
		inputs.emplace_back(std::move(wrapped));
	}
	auto start = system_clock::now();
	s_ptr->Init();

	for (auto i = 0;i<repeats;i++)
	{
		outputs.emplace_back(std::move(s_ptr->Compute(std::move(inputs[i]))));
	}

	for (auto i = 0;i<repeats;i++)
	{
		auto &output = outputs[i];
		auto val = output.get();
	}

	s_ptr->Dispose();

	auto end = system_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	std::cout << "average time of " << s_ptr->Name() << " is "
		<< (double(duration.count()) * microseconds::period::num / microseconds::period::den) / repeats
		<< " sec" << std::endl;
}


int main(int argument_count, char **arguments) {
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

//GPU
        auto nop = std::make_shared < Nop_GPU < int >> ();
        auto nop_w = AlgorithmWrapper<int, int>::create(nop);
        auto max = std::make_shared < Max_GPU < int >> ();
        auto max_w = AlgorithmWrapper < std::vector < int >,int > ::create(max);
        auto min = std::make_shared < Min_GPU < int >> ();
        auto min_w = AlgorithmWrapper < std::vector < int >,int > ::create(min);
        auto dot_pro = std::make_shared < DotPro_GPU < int >> ();
        auto dot_pro_w = AlgorithmWrapper < std::pair < std::vector < int >, std::vector<int>>, int > ::create(dot_pro);
        const auto num_threads_1 = 1;
        auto tp_nop = TaskPool<int, int>::create(nop_w, num_threads_1);
        auto tp_max = TaskPool < std::vector < int >,int > ::create(max_w, num_threads_1);
        auto tp_min = TaskPool < std::vector < int >,int > ::create(min_w, num_threads_1);
        auto tp_dot_pro = TaskPool < std::pair < std::vector < int >, std::vector<int>>, int > ::create(dot_pro_w,
                                                                                                        num_threads_1);
//CPU
//TODO
		auto qs = std::make_shared<QuickSorter<int>>();
		auto qs_w = AlgorithmWrapper<std::vector<int>, std::vector<int>>::create(qs);
		auto inc = std::make_shared<Increaser<int>>();
		auto inc_w = AlgorithmWrapper<std::vector<int>, std::vector<int>>::create(inc);


//test
//		testNop(tp_nop);
//		testMax(tp_max);
//		testMin(tp_min);
//		testDotPro(tp_dot_pro);

//timer for single function
//		timer1(tp_nop);
//      timer2(tp_min);
//      timer2(tp_max);
	  timer3(tp_dot_pro);

//timer for composition
//TODO


    }
    std::cout << "Finished" << std::endl;

    return 0;
}
