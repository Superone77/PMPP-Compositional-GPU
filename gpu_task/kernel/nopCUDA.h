#include "../commen.h"
#include <iostream>
#include <vector>

class NopCUDA{
public:
    NopCUDA() = default;

    NopCUDA(const NopCUDA& other) = default;
    NopCUDA(NopCUDA&& other) = default;

    NopCUDA& operator=(const NopCUDA& other) = default;
    NopCUDA& operator=(NopCUDA&& other) = default;

    virtual ~NopCUDA() = default;

    void Compute();

    std::string Name() const{
        return "NopCUDA";
    }

};