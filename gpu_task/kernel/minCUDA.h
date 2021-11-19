#include "../commen.h"
#include "../interfaces/KernelInterface.hpp"
#include <iostream>
#include <vector>

template<typename T>
class minCUDA:Kernel<std::vector<T>,T> {
public:
    minCUDA(){
        T* res = (T*)malloc(sizeof(T));
    }

    minCUDA(const minCUDA& other) = default;
    minCUDA(minCUDA&& other) = default;

    minCUDA& operator=(const minCUDA& other) = default;
    minCUDA& operator=(minCUDA& other) = default;

    ~minCUDA(){
        free(res);
    }

    T Compute(std::vector<T> input);

    std::string Name() const{
        return "minCUDA";
    }
    void load__gpu(std::vector<T>& input);
    void free__gpu();
private:
    T* deviceNums;
    T* minimum;
    T* res;

};