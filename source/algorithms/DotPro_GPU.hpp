#pragma once

#include "../Commons.hpp"
#include "../interfaces/AlgorithmInterface.hpp"
#include "gpuCommon.h"

#include <cassert>
#include <vector>
#include <string>

template<typename T>
class DotPro_GPU:public AlgorithmInterface<std::vector<T>, T>{
public:
    DotPro_GPU() = default;

    DotPro_GPU(DotPro_GPU& other) = default;
    DotPro_GPU(DotPro_GPU&& other) = default;

    DotPro_GPU& operator=(const DotPro_GPU& other) = default;
    DotPro_GPU& operator=(DotPro_GPU&& other) = default;
    virtual ~DotPro_GPU() = default;

    T Compute(std::vector<T>&& vec1,std::vector<T>&& vec2) const override{
        T&& res = dot_product_run(vec1,vec2);
        return std::move(res);
    }

    std::string Name() const override {
        return std::string("DotPro_GPU");
    }
};