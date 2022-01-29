#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include "gpuCommon.h"

const int threadsPerBlock=16;
const int N = 10;
const int blocksPerGrid = (N + threadsPerBlock -1) / threadsPerBlock;

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

__global__ void nopKernel(){
    //just nop-kernel
}

void nop_run(){
    nopKernel<<<1, 1>>>();
    CUDA_CHECK_ERROR;
}


__global__ void ReductionMin(int *d_a, int *d_partial_min)
{
    //申请共享内存，存在于每个block中
    __shared__ int partialMin[threadsPerBlock];

    //确定索引
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= N) return;
    int tid = threadIdx.x;

    //传global memory数据到shared memory
    partialMin[tid]=d_a[i];
    //printf("%d:%d", tid, partialMin[tid]);

    //传输同步
    __syncthreads();

    //在共享存储器中进行规约
    for(int stride = 1; stride < blockDim.x; stride*=2)
    {
        if(tid%(2*stride)==0 && tid+stride<N) partialMin[tid] = partialMin[tid]>partialMin[tid+stride]?partialMin[tid+stride]:partialMin[tid];
        __syncthreads();
    }
    //将当前block的计算结果写回输出数组
    if(tid==0)
        d_partial_min[blockIdx.x] = partialMin[0];

}


template<typename T>
T min_run(std::vector<T> &vector){
    //申请host端内存及初始化
    T   *h_a,*h_partial_min;
    h_a = (T*)malloc( N*sizeof(T) );
    h_partial_min = (T*)malloc( blocksPerGrid*sizeof(T));

    for (int i=0; i < vector.size(); ++i)  h_a[i] = vector[i];
    //for(int i = 0; i< N;i++) printf("%d ", h_a[i]);


    //分配显存空间
    int size = sizeof(T);
    T *d_a;
    T *d_partial_min;
    cudaMalloc((void**)&d_a,N*size);
    cudaMalloc((void**)&d_partial_min,blocksPerGrid*size);

    //把数据从Host传到Device
    cudaMemcpy(d_a, h_a, size*vector.size(), cudaMemcpyHostToDevice);

    //调用内核函数
    ReductionMin<<<blocksPerGrid,threadsPerBlock>>>(d_a,d_partial_min);

    //将结果传回到主机端
    cudaMemcpy(h_partial_min, d_partial_min, size*blocksPerGrid, cudaMemcpyDeviceToHost);

    //将部分和求和
    int min=INT_MAX;
    for (int i=0; i < blocksPerGrid; ++i)  min= std::min(min, h_partial_min[i]);
    //printf("%d ", min);
    cudaFree(d_a);
    cudaFree(d_partial_min);
    free(h_a);
    free(h_partial_min);
    return std::move(min);
}
template int min_run(std::vector<int> &vector);

__global__ void ReductionMax(int *d_a, int *d_partial_max)
{
    //申请共享内存，存在于每个block中
    __shared__ int partialMax[threadsPerBlock];

    //确定索引
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= N) return;
    int tid = threadIdx.x;

    //传global memory数据到shared memory
    partialMax[tid]=d_a[i];
    //printf("%d:%d", tid, partialMin[tid]);

    //传输同步
    __syncthreads();

    //在共享存储器中进行规约
    for(int stride = 1; stride < blockDim.x; stride*=2)
    {
        if(tid%(2*stride)==0&& tid+stride<N) partialMax[tid] = partialMax[tid]<partialMax[tid+stride]?partialMax[tid+stride]:partialMax[tid];
        __syncthreads();
    }
    //将当前block的计算结果写回输出数组
    if(tid==0)
        d_partial_max[blockIdx.x] = partialMax[0];

}


template<typename T>
T max_run(std::vector<T> &vector){
    //申请host端内存及初始化
    T   *h_a,*h_partial_max;
    h_a = (T*)malloc( N*sizeof(T) );
    h_partial_max = (T*)malloc( blocksPerGrid*sizeof(T));

    for (int i=0; i < vector.size(); ++i)  h_a[i] = vector[i];
//    for(int i = 0; i< N;i++) printf("%d ", h_a[i]);


    //分配显存空间
    int size = sizeof(T);
    T *d_a;
    T *d_partial_max;
    cudaMalloc((void**)&d_a,N*size);
    cudaMalloc((void**)&d_partial_max,blocksPerGrid*size);

    //把数据从Host传到Device
    cudaMemcpy(d_a, h_a, size*vector.size(), cudaMemcpyHostToDevice);

    //调用内核函数
    ReductionMax<<<blocksPerGrid,threadsPerBlock>>>(d_a,d_partial_max);

    //将结果传回到主机端
    cudaMemcpy(h_partial_max, d_partial_max, size*blocksPerGrid, cudaMemcpyDeviceToHost);

    //将部分和求和
    int max=INT_MIN;
    for (int i=0; i < blocksPerGrid; ++i)  max= std::max(max, h_partial_max[i]);
    //printf("%d ", min);
    cudaFree(d_a);
    cudaFree(d_partial_max);
    free(h_a);
    free(h_partial_max);
    return std::move(max);
}
template int max_run(std::vector<int> &vector);

__global__
void dot2(int* a, int* b, int* c)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;
    float sum = 0.0;

    while (i < N){
        sum += a[i] * b[i];
        i += blockDim.x * gridDim.x;
    }
    // c[cacheIdx] += sum;  //会出现不同步的问题
    atomicAdd(&c[cacheIdx], sum);
    __syncthreads();
    // printf("%d %d %d %d\n", i, threadIdx.x, blockIdx.x, blockDim.x);
}

template<typename T>
T dot_product_run(std::vector<T> &vec1, std::vector<T> &vec2){
    T* a;
    T* b;
    T* c;

    T* A;
    T* B;
    T* C;

    int sz = N * sizeof(T);
    a = (T*)malloc(sz);
    b = (T*)malloc(sz);
    c = (T*)malloc(sz);

    cudaMalloc(&A, sz);
    cudaMalloc(&B, sz);
    cudaMalloc(&C, sz);
    for (int i=0; i<N; i++)
    {
        a[i] = vec1[i];
        b[i] = vec2[i];
    }
    cudaMemcpy(A, a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(B, b, sz, cudaMemcpyHostToDevice);

    dot2<<<blocksPerGrid,threadsPerBlock>>>(A, B, C);

    cudaMemcpy(c, C, sz, cudaMemcpyDeviceToHost);

    T sum_c = 0;
    for (int i=0; i<N; i++)
    {
        sum_c += c[i];
    }
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(a);
    free(b);
    free(c);
    return sum_c;

}
template int dot_product_run(std::vector<int> &vec1, std::vector<int> &vec2);