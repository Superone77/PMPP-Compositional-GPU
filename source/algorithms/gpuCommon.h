//
// Created by Superone77 on 2021/11/26.
//
#include "../Commons.hpp"
#include <vector>

#ifndef EXERCISE_GPUCOMMON_H
#define EXERCISE_GPUCOMMON_H
extern "C"
void nop_run();


template<typename T>
T min_run(std::vector<T> &vector);

template<typename T>
T max_run(std::vector<T> &vector);

template<typename T>
T dot_product_run(std::vector<T> &vec1, std::vector<T> &vec2);

#endif //EXERCISE_GPUCOMMON_H
