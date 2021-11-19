#include "../commen.h"
#include <iostream>
#include <string>

template<typename T_input, typename T_output>
class Kernel {
public:
    Kernel() = default;

    Kernel(const Kernel& other) = default;
    Kernel(Kernel&& other) = default;

    Kernel& operator=(const Kernel& other) = default;
    Kernel& operator=(Kernel&& other) = default;

    virtual ~Kernel() = default;

    virtual T_output Compute(T_input input)= 0;

    virtual std::string Name() const = 0;
};
