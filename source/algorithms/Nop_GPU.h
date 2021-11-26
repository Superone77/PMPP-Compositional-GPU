#pragma once

#include "../Commons.hpp"
#include "../interfaces/AlgorithmInterface.hpp"
#include "gpuCommon.h"

#include <cassert>
#include <vector>
#include <string>


template<typename T>
class Nop_GPU:public AlgorithmInterface<std::vector<T>, std::vector<T>>{
public:
    Nop_GPU() = default;

    Nop_GPU(Nop_GPU& other) = default;
    Nop_GPU(Nop_GPU&& other) = default;

    Nop_GPU& operator=(const Nop_GPU& other) = default;
    Nop_GPU& operator=(Nop_GPU&& other) = default;
    virtual ~Nop_GPU() = default;

    std::vector<T> Compute(std::vector<T>&& vector) const override{
        nop_run();
        return std::move(vector);
    }

    std::string Name() const override {
        return std::string("Nop_GPU");
    }
};


