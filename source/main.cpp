

#include <iostream>
#include <thread>
#include <memory>
#include <vector>
#include <numeric>
#include <chrono>

#include "algorithms/Increaser.hpp"
#include "algorithms/Nopper.hpp"
#include "algorithms/Reorderer.hpp"
#include "algorithms/ReduceAdd.hpp"
#include "algorithms/ReduceMin.hpp"
#include "algorithms/SelectionSorter.hpp"
#include "algorithms/QuickSorter.hpp"
#include "algorithms/Nop_GPU.h"
#include "algorithms/Min_GPU.hpp"
#include "algorithms/Max_GPU.hpp"
#include "algorithms/DotPro_GPU.hpp"
#include "algorithms/MatDouble_GPU.hpp"
#include "interfaces/AlgorithmWrapper.hpp"

#include "pattern/TaskPool.hpp"
#include "pattern/Pipeline.hpp"
#include "pattern/Composition.hpp"


#include "pattern/ForkJoin.hpp"
#include "helper/Comparator.hpp"

using namespace std::chrono;
const int test_repeats = 100;
const int repeats = 100;
const int size = 1000;//size must equal to N in gpuCommon.cu

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
            std::cout<<std::endl;
            std::cout <<val<<" "<<validation[i]<<" "<<i<<" "<< "Find Min element false" << std::endl;
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
    std::cout << "Test Nop successfully" << std::endl;
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
void testMatDb(PatIntPtr<std::vector<std::vector<int>>, std::vector<std::vector<int>>> s_ptr)
{
    auto create = [](int size) {
        auto res = std::vector<std::vector<int>>(size);
        for(int i = 0;i<size;i++) {
            auto vec = std::vector<int>(size);
            std::generate(vec.begin(), vec.end(), rand);
            res[i] = vec;
        }
        return res;
    };

    auto mat2double = [](std::vector<std::vector<int>> vec){
        std::vector<std::vector<int>> res ;
        for(int i = 0;i<size;i++){
            std::vector<int> temp;
            for(int j = 0;j<size;j++){
                int num = vec[i][j]+vec[i][j];
                temp.push_back(num);
            }
            res.push_back(temp);
        }
        return res;
    };

    auto wrap = [](std::vector<std::vector<int>> vec) {
        std::promise<std::vector<std::vector<int>>> prom;
        prom.set_value(std::move(vec));
        return prom.get_future();
    };

    std::cout << "Testing: " << s_ptr->Name() << std::endl;


    std::vector<std::future<std::vector<std::vector<int>>>> inputs;
    std::vector<std::future<std::vector<std::vector<int>>>> outputs;
    std::vector<std::vector<std::vector<int>>> validation;

    for (auto i = 0; i < test_repeats; i++)
    {
        auto vec = create(size);
        auto valid = mat2double(vec);
        auto wrapped = std::move(wrap(std::move(vec)));
        inputs.emplace_back(std::move(wrapped));
        validation.emplace_back(valid);
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
        auto validate = validation[i];
        if(val[0][0] != validate[0][0] || val[size-1][size-1] != validate[size-1][size-1]){
            std::cout<<"In "<< i<<". test "<<std::endl;
            std::cout<<"MatDouble false"<<std::endl;
            return;
        }
    }

    s_ptr->Dispose();
    std::cout<<"Test MatDouble GPU successfully"<<std::endl;
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

void timer4(PatIntPtr<std::vector<int>, std::vector<int>> s_ptr) {
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
    std::vector <std::future<std::vector < int>>> outputs;

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

void timer5(PatIntPtr<std::tuple<int, int>, int> s_ptr) {
    auto wrap = [](std::tuple<int,int> tuple) {
        std::promise <std::tuple<int,int>> prom;
        prom.set_value(std::move(tuple));
        return prom.get_future();
    };

    std::cout << "Testing Time of : " << s_ptr->Name() << std::endl;
    std::cout << "repeats: " << repeats << std::endl;
    std::cout << "size: " << size << std::endl;



    std::vector < std::future < std::tuple<int,int>>> inputs;
    std::vector <std::future<int>> outputs;

    for (auto i = 0; i < repeats; i++) {
        int i1 = rand();
        int i2 = rand();
        auto tuple = std::make_tuple(i1,i2);
        auto wrapped = std::move(wrap(std::move(tuple)));
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
void timer6(PatIntPtr<std::vector<std::vector<int>>, std::vector<std::vector<int>>> s_ptr)
{
    auto create = [](int size) {
        auto res = std::vector<std::vector<int>>(size);
        for(int i = 0;i<size;i++) {
            auto vec = std::vector<int>(size);
            std::generate(vec.begin(), vec.end(), rand);
            res[i] = vec;
        }
        return res;
    };

    auto wrap = [](std::vector<std::vector<int>> vec) {
        std::promise<std::vector<std::vector<int>>> prom;
        prom.set_value(std::move(vec));
        return prom.get_future();
    };
    std::cout << "Testing Time of : " << s_ptr->Name() << std::endl;
    std::cout << "repeats: " << repeats << std::endl;
    std::cout << "size: " << size << std::endl;

    std::vector<std::future<std::vector<std::vector<int>>>> inputs;
    std::vector<std::future<std::vector<std::vector<int>>>> outputs;

    for (auto i = 0; i < repeats; i++)
    {
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

int main(int argument_count, char **arguments) {
    {
        const auto num_threads_1 = 1;
//GPU
        auto nop = std::make_shared < Nop_GPU < int >> ();
        auto nop_w = AlgorithmWrapper<int, int>::create(nop);
        auto max = std::make_shared < Max_GPU < int >> ();
        auto max_w = AlgorithmWrapper < std::vector < int >,int > ::create(max);
        auto min = std::make_shared < Min_GPU < int >> ();
        auto min_w = AlgorithmWrapper < std::vector < int >,int > ::create(min);
        auto dot_pro = std::make_shared < DotPro_GPU < int >> ();
        auto dot_pro_w = AlgorithmWrapper < std::pair < std::vector < int >, std::vector<int>>, int > ::create(dot_pro);
        auto matdb_gpu = std::make_shared<MatDouble_GPU<int>>();
        auto matdb_gpu_w = AlgorithmWrapper<std::vector<std::vector<int>>, std::vector<std::vector<int>>> ::create(matdb_gpu);


        auto tp_nop = TaskPool<int, int>::create(nop_w, num_threads_1);
        auto tp_max = TaskPool < std::vector < int >,int > ::create(max_w, num_threads_1);
        auto tp_min = TaskPool < std::vector < int >,int > ::create(min_w, num_threads_1);
        auto tp_dot_pro = TaskPool < std::pair < std::vector < int >, std::vector<int>>, int > ::create(dot_pro_w,num_threads_1);
        auto tp_matdb_gpu = TaskPool<std::vector<std::vector<int>>, std::vector<std::vector<int>>>::create(matdb_gpu_w, num_threads_1);
//CPU
//TODO
        const auto num_threads_2 = 8;
		auto qs = std::make_shared<QuickSorter<int>>();
		auto qs_w = AlgorithmWrapper<std::vector<int>, std::vector<int>>::create(qs);
		auto inc = std::make_shared<Increaser<int>>();
		auto inc_w = AlgorithmWrapper<std::vector<int>, std::vector<int>>::create(inc);
        auto tp_qs = TaskPool<std::vector<int>, std::vector<int>>::create(qs_w,num_threads_2);
        auto tp_inc = TaskPool<std::vector<int>, std::vector<int>>::create(inc_w,num_threads_2);

        auto nopper = std::make_shared < Nopper < int >> ();
        auto nopper_w = AlgorithmWrapper<int, int>::create(nopper);
        auto tp_nopper = TaskPool<int, int>::create(nopper_w, num_threads_2);

        auto ro = std::make_shared < Reorderer < int >> ();
        auto ro_w = AlgorithmWrapper<std::vector < int >,std::vector < int >>::create(ro);
        auto tp_ro = TaskPool< std::vector < int >,std::vector < int > >::create(ro_w, num_threads_2);

        auto ss = std::make_shared < SelectionSorter < int >> ();
        auto ss_w = AlgorithmWrapper<std::vector < int >,std::vector < int >>::create(ss);
        auto tp_ss = TaskPool< std::vector < int >,std::vector < int > >::create(ss_w, num_threads_2);

        auto ra = std::make_shared < ReduceAddVector < int >> ();
        auto ra_w = AlgorithmWrapper < std::vector < int >,int > ::create(ra);
        auto tp_ra = TaskPool < std::vector < int >,int > ::create(ra_w, num_threads_2);

        auto rm = std::make_shared < ReduceMin < int,int >> ();
        auto rm_w = AlgorithmWrapper < std::tuple < int,int >,int > ::create(rm);
        auto tp_rm = TaskPool < std::tuple < int,int >,int > ::create(rm_w, num_threads_2);



//test
//		testNop(tp_nop);
//		testMax(tp_max);
//		testMin(tp_min);
//		testDotPro(tp_dot_pro);
        //testMatDb(tp_matdb_gpu);


//timer for single function
 //   timer1(tp_nop);
//      timer2(tp_min);
//      timer2(tp_max);
//	  timer3(tp_dot_pro);
//        timer4(tp_qs);
//      timer4(tp_inc);
//        timer1(tp_nopper);
//        timer4(tp_ro);
//        timer4(tp_ss);
//        timer2(tp_ra);
//        timer5(tp_rm);
        timer6(tp_matdb_gpu);
//timer for composition
//TODO


    }
    std::cout << "Finished" << std::endl;

    return 0;
}
