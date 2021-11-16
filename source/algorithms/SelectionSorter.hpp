#pragma once

#include "../Commons.hpp"
#include "../interfaces/AlgorithmInterface.hpp"

#include <cassert>
#include <vector>
#include <string>

template<typename T>
class SelectionSorter : public AlgorithmInterface<std::vector<T>, std::vector<T>> {
public:
	SelectionSorter() = default;

	SelectionSorter(SelectionSorter& other) = default;
	SelectionSorter(SelectionSorter&& other) = default;

	SelectionSorter& operator=(const SelectionSorter& other) = default;
	SelectionSorter& operator=(SelectionSorter&& other) = default;

	virtual ~SelectionSorter() = default;

	std::vector<T> Compute(std::vector<T>&& vector) const override {
		size_t length = vector.size();
		assert(length > 0);

		for (int i = 0; i < length - 1; i++) {
			int min_idx = i;
			for (int j = i + 1; j < length; j++) {
				if (vector[j] < vector[min_idx]) {
					min_idx = j;
				}
			}

			std::swap(vector[min_idx], vector[i]);
		}

		return std::move(vector);
	}

	std::string Name() const override {
		return std::string("selectionsorter");
	}
};