#include <stdio.h>
#include "minCUDA.h"
#define BLOCK_SIZE 16
#define SQRT_BLOCK_SIZE 4

template <typename T>
__global__ void minKernel(T* nums, T* min, int N);

template <typename T>
void minCUDA<T>::load__gpu(std::vector<T>& input){
    cudaMalloc((void**)&minimum,sizeof(T));
    cudaMalloc((void**)&deviceNums, input.size()*sizeof(T));
    cudaMemcpy(deviceNums, input, input.size()*sizeof(T),cudaMemcpyHostToDevice);
}
template <typename T>
void minCUDA<T>::free__gpu(){
    cudaFree(minimum);
    cudaFree(deviceNums);
}
template <typename T>
T minCUDA<T>::Compute(std::vector<T> input){
    load__gpu(input);
    int N = input.size();
    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;
    minKernel<T><<<blocksPerGrid, threadsPerBlock>>>(deviceNums, minimum, N);
    cudaMemcpy(res, minimum, input.size()*sizeof(T),cudaMemcpyDeviceToHost);
    free__gpu();
    return *res;
}
template <typename T>
__global__ void minKernel(T* nums, T* min, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N){
        if(min>nums[i]){
            min = nums[i];
        }
    }
}