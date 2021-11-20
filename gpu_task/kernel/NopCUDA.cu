/*
 * nop-kernel
 */
#include <stdio.h>
#include "NopCUDA.h"
#define BLOCK_SIZE 32


__global__ void nopKernel(){
    //just nop-kernel
}

void NopCUDA::Compute(float& time){
    float milliseconds = 0;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(1,1);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    nopKernel<<<dimGrid, dimBlock>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    time = milliseconds;
    //printf("nop-kernel\n");
    CUDA_CHECK_ERROR;
}