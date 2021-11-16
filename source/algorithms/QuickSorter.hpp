#pragma once

#include "../Commons.hpp"
#include "../interfaces/AlgorithmInterface.hpp"

#include "../helper/quicksort.hpp"

#include <cassert>
#include <tuple>
#include <vector>
#include <string>

template<typename T>
class QuickSorter : public AlgorithmInterface<std::vector<T>, std::vector<T>> {
public:
	QuickSorter() = default;

	QuickSorter(QuickSorter& other) = default;
	QuickSorter(QuickSorter&& other) = default;

	QuickSorter& operator=(const QuickSorter& other) = default;
	QuickSorter& operator=(QuickSorter&& other) = default;

	virtual ~QuickSorter() = default;

	std::vector<T> Compute(std::vector<T>&& vector) const override {
		size_t length = vector.size();
		assert(length > 0);

		quicksort(vector);

		return std::move(vector);
	}

	std::string Name() const override {
		return std::string("quicksorter");
	}
};

template <typename T_value, typename T_key>
class QuickSorterKeyed : public AlgorithmInterface<std::vector<std::tuple<T_value, T_key>>, std::vector<std::tuple<T_value, T_key>>> {
public:
	QuickSorterKeyed() = default;

	QuickSorterKeyed(QuickSorterKeyed& other) = default;
	QuickSorterKeyed(QuickSorterKeyed&& other) = default;

	QuickSorterKeyed& operator=(const QuickSorterKeyed& other) = default;
	QuickSorterKeyed& operator=(QuickSorterKeyed&& other) = default;

	virtual ~QuickSorterKeyed() = default;

	std::vector<std::tuple<T_value, T_key>> Compute(std::vector<std::tuple<T_value, T_key>>&& vector) const override {
		size_t length = vector.size();
		assert(length > 0);

		quicksort_keyed(vector);

		return std::move(vector);
	}

	std::string Name() const override {
		return std::string("quicksorter_key");
	}
};
