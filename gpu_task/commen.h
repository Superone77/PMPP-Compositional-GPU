#pragma once

#include <iostream>
#include <stdint.h>
#include <cuda_runtime.h>
#define TIME_TEST 0


//#define TIME_TEST_FIRST
//{
//cudaEvent_t evStart, evStop;
//cudaEventCreate(&evStart);
//cudaEventCreate(&evStop);
//cudaEventRecord(evStart,0);
//}
//#define TIME_TEST_LAST
//{};
//    cudaEventRecord(evStop,0); \
//    cudaEventSynchronize(evStop); \
//    float elapsedTime_ms;   \
//    cudaEventElapsedTime(&elapsedTime_ms, evStart, evStop); \
//    std::cout<<"CUDA with global memory horizontal processing took "<<elapsedTime_ms<<"ms"<<std::endl; \
//    cudaEventDestroy(evStart);  \
//    cudaEventDestroy(evStop); \



inline unsigned int div_up(unsigned int numerator, unsigned int denominator)
{
    unsigned int result = numerator / denominator;
    if (numerator % denominator) ++result;
    return result;
}