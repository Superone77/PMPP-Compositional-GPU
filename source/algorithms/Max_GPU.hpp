#pragma once

#include "../Commons.hpp"
#include "../interfaces/AlgorithmInterface.hpp"
#include "gpuCommon.h"

#include <cassert>
#include <vector>
#include <string>

template<typename T>
class Max_GPU:public AlgorithmInterface<std::vector<T>, T>{
public:
    Max_GPU() = default;

    Max_GPU(Max_GPU& other) = default;
    Max_GPU(Max_GPU&& other) = default;

    Max_GPU& operator=(const Max_GPU& other) = default;
    Max_GPU& operator=(Max_GPU&& other) = default;
    virtual ~Max_GPU() = default;

    T Compute(std::vector<T>&& vector) const override{
        T&& res = max_run(vector);
        return std::move(res);
    }

    std::string Name() const override {
        return std::string("Max_GPU");
    }
};