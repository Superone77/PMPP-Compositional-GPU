#pragma once

#include "../Commons.hpp"
#include "../interfaces/AlgorithmInterface.hpp"
#include "gpuCommon.h"

#include <cassert>
#include <vector>
#include <string>

template<typename T>
class DotPro_GPU:public AlgorithmInterface<std::pair<std::vector<T>,std::vector<T>>, int>{
public:
    DotPro_GPU() = default;

    DotPro_GPU(DotPro_GPU& other) = default;
    DotPro_GPU(DotPro_GPU&& other) = default;

    DotPro_GPU& operator=(const DotPro_GPU& other) = default;
    DotPro_GPU& operator=(DotPro_GPU&& other) = default;
    virtual ~DotPro_GPU() = default;

    T Compute(std::pair<std::vector<T>,std::vector<T>>&& vec) const override{
        T&& res = dot_product_run(vec.first,vec.second);
        return std::move(res);
    }

    std::string Name() const override {
        return std::string("DotPro_GPU");
    }
};
