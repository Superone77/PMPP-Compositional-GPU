///*
// * nop-kernel
// */
//#include <stdio.h>
//#include "kernel.h"
//#define BLOCK_SIZE 32
//
//
////__global__ void nopKernel(){
////    //just nop-kernel
////}
//
//void nopKernel_test(){
//    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//    dim3 dimGrid(1,1);
//
//    nopKernel<<<dimGrid, dimBlock>>>();
//    printf("nop-kernel finished\n");
//    CUDA_CHECK_ERROR;
//}
//
////template <typename T>
//__global__ void minKernel(std::vector<int> vec){
//    //TODO
//}
//
////template <typename T>
//int minKernel_test(std::vector<int>&& vec){//TODO
//    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//    dim3 dimGrid(1,1);
//    nopKernel<<<dimGrid, dimBlock>>>();
//    CUDA_CHECK_ERROR;
//
//
//    return 0;
//}
//
//
//int maxKernel_test(std::vector<int>&& vec){
//    //TODO
//    return 0;
//}
//
//int scalar_product(std::vector<int>&& vec1, std::vector<int>&& vec2){
//    //TODO
//    return 0;
//}