#pragma once

#include "../Commons.hpp"
#include "../interfaces/AlgorithmInterface.hpp"
#include "gpuCommon.h"

#include <cassert>
#include <tuple>
#include <vector>
#include <string>

template<typename T>
class MatDouble_GPU : public AlgorithmInterface<std::vector<std::vector<T>>, std::vector<std::vector<T>>> {
public:
    MatDouble_GPU() = default;

    MatDouble_GPU(MatDouble_GPU& other) = default;
    MatDouble_GPU(MatDouble_GPU&& other) = default;

    MatDouble_GPU& operator=(const MatDouble_GPU& other) = default;
    MatDouble_GPU& operator=(MatDouble_GPU&& other) = default;

    virtual ~MatDouble_GPU() = default;

    std::vector<std::vector<T>> Compute(std::vector<std::vector<T>>&& vector) const override{

        matrix_double_gpu(vector);

        return std::move(vector);
    }

    std::string Name() const override {
        return std::string("MatDouble_GPU");
    }
};

