/*
 * nop-kernel
 */

#define BLOCK_SIZE 32

__global__ void nopKernel(){
    //just nop-kernel
}

void nopKernel_test(){
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(1,1);

    nopKernel<<<dimGrid, dimBlock>>>();
//    CUDA_CHECK_ERROR;
}

