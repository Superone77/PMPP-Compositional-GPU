#pragma once

#include "../commen.h"
#include <iostream>
#include <string>

class cKernel {
public:
    cKernel() = default;

    cKernel(const cKernel& other) = default;
    cKernel(cKernel&& other) = default;

    cKernel& operator=(const cKernel& other) = default;
    cKernel& operator=(cKernel&& other) = default;

    virtual ~cKernel() = default;

    virtual std::string Name() const = 0;
};
