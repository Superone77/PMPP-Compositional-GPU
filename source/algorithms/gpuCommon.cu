#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <limits>
#include "gpuCommon.h"


const int N = 10000;
using namespace std;

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
inline unsigned int div_up(unsigned int numerator, unsigned int denominator)
{
    unsigned int result = numerator / denominator;
    if (numerator % denominator) ++result;
    return result;
}

const int threadsPerBlock=16;

const int blocksPerGrid = (N + threadsPerBlock -1) / threadsPerBlock;

__global__ void nopKernel(){
    //just nop-kernel
}

void nop_run(){
    nopKernel<<<1, 1>>>();
    CUDA_CHECK_ERROR;
}


__global__ void ReductionMin(int *d_a, int *d_partial_min)
{
    __shared__ int partialMin[threadsPerBlock];

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= N) return;
    int tid = threadIdx.x;

    partialMin[tid]=d_a[i];
    //printf("%d:%d", tid, partialMin[tid]);

    __syncthreads();

    for(int stride = 1; stride < blockDim.x; stride*=2)
    {
        if(tid%(2*stride)==0 && tid+stride<N && partialMin[tid+stride] != 0) partialMin[tid] = partialMin[tid]>partialMin[tid+stride]?partialMin[tid+stride]:partialMin[tid];
        __syncthreads();
    }
    if(tid==0)
        d_partial_min[blockIdx.x] = partialMin[0];

}


template<typename T>
T min_run(std::vector<T> &vector){
    T   *h_a,*h_partial_min;
    h_a = (T*)malloc( N*sizeof(T) );
    h_partial_min = (T*)malloc( blocksPerGrid*sizeof(T));

    for (int i=0; i < vector.size(); ++i)  h_a[i] = vector[i];
    //for(int i = 0; i< N;i++) printf("%d ", h_a[i]);


    int size = sizeof(T);
    T *d_a;
    T *d_partial_min;
    cudaMalloc((void**)&d_a,N*size);
    cudaMalloc((void**)&d_partial_min,blocksPerGrid*size);

    cudaMemcpy(d_a, h_a, size*vector.size(), cudaMemcpyHostToDevice);

    ReductionMin<<<blocksPerGrid,threadsPerBlock>>>(d_a,d_partial_min);

    cudaMemcpy(h_partial_min, d_partial_min, size*blocksPerGrid, cudaMemcpyDeviceToHost);

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
    __shared__ int partialMax[threadsPerBlock];

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= N) return;
    int tid = threadIdx.x;

    partialMax[tid]=d_a[i];
    //printf("%d:%d", tid, partialMin[tid]);

    __syncthreads();

    for(int stride = 1; stride < blockDim.x; stride*=2)
    {
        if(tid%(2*stride)==0&& tid+stride<N) partialMax[tid] = partialMax[tid]<partialMax[tid+stride]?partialMax[tid+stride]:partialMax[tid];
        __syncthreads();
    }
    if(tid==0)
        d_partial_max[blockIdx.x] = partialMax[0];

}


template<typename T>
T max_run(std::vector<T> &vector){
    T   *h_a,*h_partial_max;
    h_a = (T*)malloc( N*sizeof(T) );
    h_partial_max = (T*)malloc( blocksPerGrid*sizeof(T));

    for (int i=0; i < vector.size(); ++i)  h_a[i] = vector[i];
//    for(int i = 0; i< N;i++) printf("%d ", h_a[i]);


    int size = sizeof(T);
    T *d_a;
    T *d_partial_max;
    cudaMalloc((void**)&d_a,N*size);
    cudaMalloc((void**)&d_partial_max,blocksPerGrid*size);

    cudaMemcpy(d_a, h_a, size*vector.size(), cudaMemcpyHostToDevice);

    ReductionMax<<<blocksPerGrid,threadsPerBlock>>>(d_a,d_partial_max);

    cudaMemcpy(h_partial_max, d_partial_max, size*blocksPerGrid, cudaMemcpyDeviceToHost);

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


__global__ void MatDoubleKernel(int* Md, int* Pd) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by * 1+ ty;
    int col = bx *1 + tx;
    int m_value = Md[row*N+col];
    Pd[row*N+col]=2*m_value;
}


template<typename T>
vector<vector<T>> matrix_double_gpu(vector<vector<T>> &vec)
{
    T* m;
    T* p;

    T* Md;
    T* Pd;

    int sz = N*N * sizeof(T);
    m = (T*)malloc(sz);
    p = (T*)malloc(sz);

    cudaMalloc(&Md, sz);
    cudaMalloc(&Pd, sz);

    for(int i = 0;i<N;i++){
        for(int j = 0;j<N;j++){
            m[i*N+j] = vec[i][j];
        }
    }
    cudaMemcpy(Md, m, sz, cudaMemcpyHostToDevice);

    dim3 dimBlock(1, 1);
    dim3 dimGrid(N,N);
    MatDoubleKernel<<<dimGrid, dimBlock>>>(Md, Pd);

    cudaMemcpy(p, Pd, sz, cudaMemcpyDeviceToHost);

    for(int i = 0;i<N;i++){
        for(int j = 0;j<N;j++){
            vec[i][j] = p[i*N+j];
        }
    }
    cudaFree(Pd);
    cudaFree(Md);
    free(m);
    free(p);
    return vec;
}

template vector<vector<int>> matrix_double_gpu(vector<vector<int>> &vec);