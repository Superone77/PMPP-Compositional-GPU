#pragma once

#include "../Commons.hpp"
#include "../interfaces/AlgorithmInterface.hpp"
#include "gpuCommon.h"

#include <cassert>
#include <vector>
#include <string>

template<typename T>
class Min_GPU:public AlgorithmInterface<std::vector<T>, T>{
public:
    Min_GPU() = default;

    Min_GPU(Min_GPU& other) = default;
    Min_GPU(Min_GPU&& other) = default;

    Min_GPU& operator=(const Min_GPU& other) = default;
    Min_GPU& operator=(Min_GPU&& other) = default;
    virtual ~Min_GPU() = default;

    T Compute(std::vector<T>&& vector) const override{
        T&& res = min_run(vector);
        return std::move(res);
    }

    std::string Name() const override {
        return std::string("Min_GPU");
    }
};