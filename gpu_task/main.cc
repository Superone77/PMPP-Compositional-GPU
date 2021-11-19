#include "kernel/nopCUDA.h"
#include "kernel/minCUDA.h"
#include <iostream>
#include <vector>


int main(){
    NopCUDA* nop = new NopCUDA();
    nop->Compute();
    std::cout<<nop->Name()<<" ";
    std::cout<<"finished"<<std::endl;
    delete nop;
    minCUDA<float>* min = new minCUDA<float>();
    std::vector<float> vec;
    vec.push_back(1.0);
    vec.push_back(2.0);
    float x = min->Compute(vec);
    std::cout<<"minimum is " << x <<std::endl;
    std::cout<<min->Name()<<" ";
    std::cout<<"finished"<<std::endl;
}