#pragma once

#include "../Commons.hpp"
#include "../interfaces/AlgorithmInterface.hpp"
#include "gpuCommon.h"

#include <cassert>
#include <vector>
#include <string>


template<typename T>
class DotProSmVec_GPU:public AlgorithmInterface<std::vector<T>, int>{
public:
    DotProSmVec_GPU() = default;

    DotProSmVec_GPU(DotProSmVec_GPU& other) = default;
    DotProSmVec_GPU(DotProSmVec_GPU&& other) = default;

    DotProSmVec_GPU& operator=(const DotProSmVec_GPU& other) = default;
    DotProSmVec_GPU& operator=(DotProSmVec_GPU&& other) = default;
    virtual ~DotProSmVec_GPU() = default;

    T Compute(std::vector<T>&& vec) const override{
        T&& res = dot_product_run(vec,vec);
        return std::move(res);
    }

    std::string Name() const override {
        return std::string("DotProSmVec_GPU");
    }
};