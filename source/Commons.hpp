#pragma once

#include <future>
#include <memory>
#include <vector>
#include <iostream>
#include <stdint.h>
#include <cuda_runtime.h>

#define CUDA_CHECK_ERROR                                                       \
    do {                                                                       \
        const cudaError_t err = cudaGetLastError();                            \
        if (err != cudaSuccess) {                                              \
            const char *const err_str = cudaGetErrorString(err);               \
            std::cerr << "Cuda error in " << __FILE__ << ":" << __LINE__ - 1   \
                      << ": " << err_str << " (" << err << ")" << std::endl;   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while(0)

template<typename T1, typename T2>
class PatternInterface;

template<typename T1, typename T2>
using PatIntPtr = std::shared_ptr<PatternInterface<T1, T2>>;


template<typename T1, typename T2>
class AlgorithmInterface;

template<typename T1, typename T2>
using AlgoIntPtr = std::shared_ptr<AlgorithmInterface<T1, T2>>;


template<typename T1, typename T2>
class AlgorithmWrapper;

template<typename T1, typename T2>
using AlgoWrapPtr = std::shared_ptr<AlgorithmWrapper<T1, T2>>;


template<typename T>
using FutVec = std::vector<std::future<T>>;
