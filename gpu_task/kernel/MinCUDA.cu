#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>
#include "MinCUDA.h"

#define N 16
#define BLOCKSIZE 16

__global__ void minKernel(double *min, const double *a){
    __shared__ double mintile[BLOCKSIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    mintile[tid] = a[i];
    __syncthreads();

    // strided index and non-divergent branch
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            if (mintile[tid + s] < mintile[tid])
                mintile[tid] = mintile[tid + s];
        }
        __syncthreads();
}

    if (tid == 0) {
        min[blockIdx.x] = mintile[0];
    }
}
__global__ void finalminKernel(double *min) {
    __shared__ double mintile[BLOCKSIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    mintile[tid] = min[i];
    __syncthreads();

    // strided index and non-divergent branch
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            if (mintile[tid + s] < mintile[tid])
                mintile[tid] = mintile[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        min[blockIdx.x] = mintile[0];
    }
}

cudaError_t MinCUDA::Compute(double*min, const double *a, float &time){
    double *dev_a = 0;
    double *dev_min = 0;
    float milliseconds = 0;
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid(N);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaSetDevice(0);
    CUDA_CHECK_ERROR;

    cudaMalloc((void**)&dev_min, N * sizeof(float));
    CUDA_CHECK_ERROR;

    cudaMalloc((void**)&dev_a, N * N * sizeof(float));
    CUDA_CHECK_ERROR;

    cudaMemcpy(dev_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR;

    cudaEventRecord(start);
    minKernel<<<dimGrid, dimBlock>>>(dev_min, dev_a);
    CUDA_CHECK_ERROR;
    cudaThreadSynchronize();
    finalminKernel<<<1, dimBlock>>>(dev_min);
    CUDA_CHECK_ERROR;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //cudaDeviceSynchronize();
    CUDA_CHECK_ERROR;
    cudaMemcpy(min, dev_min, sizeof(double), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR;
    cudaEventElapsedTime(&milliseconds, start, stop);
    CUDA_CHECK_ERROR;
    time = milliseconds;


    return cudaSuccess;
}



