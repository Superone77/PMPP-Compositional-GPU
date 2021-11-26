#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gpuCommon.h"

__global__ void nopKernel(){
    //just nop-kernel
}

void nop_run(){
    nopKernel<<<1, 1>>>();
    CUDA_CHECK_ERROR;
}
