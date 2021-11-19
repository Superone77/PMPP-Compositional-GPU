/*
 * nop-kernel
 */
#include <stdio.h>
#include "nopCUDA.h"
#define BLOCK_SIZE 32

__global__ void nopKernel(){
    //just nop-kernel
}

void NopCUDA::Compute(){
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(1,1);
    nopKernel<<<dimGrid, dimBlock>>>();
    printf("nop-kernel\n");
    CUDA_CHECK_ERROR;
}