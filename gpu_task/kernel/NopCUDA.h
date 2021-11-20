#include "../commen.h"
#include "../interfaces/KernelInterface.hpp"
#include <iostream>
#include <vector>

class NopCUDA:cKernel{
public:
    NopCUDA() = default;

    NopCUDA(const NopCUDA& other) = default;
    NopCUDA(NopCUDA&& other) = default;

    NopCUDA& operator=(const NopCUDA& other) = default;
    NopCUDA& operator=(NopCUDA&& other) = default;

    virtual ~NopCUDA() = default;

    void Compute(float& time);

    std::string Name() const{
        return "NopCUDA";
    }

};