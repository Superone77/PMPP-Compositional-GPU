#include "../commen.h"
#include "../interfaces/KernelInterface.hpp"
#include "cuda_runtime.h"
#include <iostream>
#include <vector>

class MinCUDA:cKernel{
public:
    MinCUDA()=default;

    MinCUDA(const MinCUDA& other) = default;
    MinCUDA(MinCUDA&& other) = default;

    MinCUDA& operator=(const MinCUDA& other) = default;
    MinCUDA& operator=(MinCUDA& other) = default;

    virtual ~MinCUDA() = default;

    cudaError_t Compute(double *min, const double *a, float &time);

    std::string Name() const{
        return "minCUDA";
    }

};