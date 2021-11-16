#pragma once

#include <future>
#include <memory>
#include <vector>


template<typename T1, typename T2>
class PatternInterface;

template<typename T1, typename T2>
using PatIntPtr = std::shared_ptr<PatternInterface<T1, T2>>;


template<typename T1, typename T2>
class AlgorithmInterface;

template<typename T1, typename T2>
using AlgoIntPtr = std::shared_ptr<AlgorithmInterface<T1, T2>>;


template<typename T1, typename T2>
class AlgorithmWrapper;

template<typename T1, typename T2>
using AlgoWrapPtr = std::shared_ptr<AlgorithmWrapper<T1, T2>>;


template<typename T>
using FutVec = std::vector<std::future<T>>;
