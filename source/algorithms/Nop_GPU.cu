//#pragma once
//
//#include <stdio.h>
//#include "Nop_GPU.h"
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#define BLOCK_SIZE 1
//
//__global__ void nopKernel(){
//    //just nop-kernel
//}
//
//template<typename T>
//std::vector<T> Nop_GPU<T>::Compute(std::vector<T>&& vector) const override{
//    runtest();
//    return std::move(vector);
//}
////    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
////    dim3 dimGrid(1,1);
////    nopKernel<<<dimGrid, dimBlock>>>();
////    CUDA_CHECK_ERROR;
////    return std::move(vector);
//
//extern "C" void runtest(){
//    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//    dim3 dimGrid(1,1);
//    nopKernel<<<dimGrid, dimBlock>>>();
//    CUDA_CHECK_ERROR;
//}
//
//
