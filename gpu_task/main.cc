#include "kernel/NopCUDA.h"
#include "kernel/MinCUDA.h"
#include <iostream>
#include <vector>
#define N 16


int main() {
    NopCUDA *nop = new NopCUDA();
    float time = 0.0f;
    nop->Compute(time);
    std::cout << nop->Name() << " ";
    std::cout << "finished" << std::endl;
    std::cout<<"time: "<<time<<std::endl;
    delete nop;
    MinCUDA *minCuda = new MinCUDA();
    const double a[N] = {-8.5, -8.4, -6.8, -4.5, -4.2, -3.9, -3.4, -2.3, 1.5, 3.3, 4.3, 4.7, 6.5, 6.7, 8.0, 9.4};


    double *min;
    time = 0.0f;

    min = (double *) malloc((N) * sizeof(double));

    cudaError_t cudaStatus = minCuda->Compute(min, a, time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "minmaxCuda failed!");
        return 1;
    }
    std::cout << minCuda->Name() << " ";
    std::cout << "finished" << std::endl;
    std::cout<<"min: "<<min[0]<<std::endl;
    std::cout<<"time: "<<time<<std::endl;

}