#pragma once

#include "../Commons.hpp"

#include "../interfaces/PatternInterface.hpp"
#include "../interfaces/ThreadSafeQueue.hpp"
#include "../interfaces/Executor.hpp"

template <typename T_input1, typename T_input2>
class ComparatorSorted : public AlgorithmInterface<std::tuple<T_input1&, T_input2&>, bool>
{
public:
    ComparatorSorted() = default;

    ComparatorSorted(ComparatorSorted &other) = default;
    ComparatorSorted(ComparatorSorted &&other) = default;

    ComparatorSorted &operator=(const ComparatorSorted &other) = default;
    ComparatorSorted &operator=(ComparatorSorted &&other) = default;

    virtual ~ComparatorSorted() = default;

    bool Compute(std::tuple<T_input1&, T_input2&> &&value) const override
    {
        return std::is_sorted(std::get<0>(value).begin(),std::get<0>(value).end());
    }
    
    std::string Name() const override {
		return std::string("sorted comparator");
	}
};
