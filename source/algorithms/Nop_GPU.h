#pragma once

#include "../Commons.hpp"
#include "../interfaces/AlgorithmInterface.hpp"
#include "gpuCommon.h"

#include <cassert>
#include <vector>
#include <string>


template<typename T>
class Nop_GPU:public AlgorithmInterface<T, T>{
public:
    Nop_GPU() = default;

    Nop_GPU(Nop_GPU& other) = default;
    Nop_GPU(Nop_GPU&& other) = default;

    Nop_GPU& operator=(const Nop_GPU& other) = default;
    Nop_GPU& operator=(Nop_GPU&& other) = default;
    virtual ~Nop_GPU() = default;

    T Compute(T&& num) const override{
        nop_run();
        return std::move(num);
    }

    std::string Name() const override {
        return std::string("Nop_GPU");
    }
};


