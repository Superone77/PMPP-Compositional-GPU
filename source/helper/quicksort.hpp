#pragma once

#include "../Commons.hpp"

#include <tuple>

template<typename T>
long long int partition(T* const __restrict ptr, long long int low, long long int high) {
	T pivot = ptr[low + (high - low) / 2];
	long long int i = low - 1;
	long long int j = high + 1;

	while (true) {
		do {
			i++;
		} while (ptr[i] < pivot);

		do {
			j--;
		} while (ptr[j] > pivot);

		if (i >= j) {
			return j;
		}

		std::swap(ptr[i], ptr[j]);
	}
}

template<typename T> 
void quicksort(T* const __restrict ptr, long long int low, long long int high) {
	if (low >= high) {
		return;
	}

	long long int p = partition(ptr, low, high);
	quicksort(ptr, low, p);
	quicksort(ptr, p + 1, high);
}

template<typename T>
void quicksort(std::vector<T>& vector) {
	quicksort(vector.data(), 0, vector.size() - 1);
}

template <typename T_value, typename T_key>
long long int partition_keyed(std::tuple<T_value, T_key>* const __restrict ptr, long long int low, long long int high) {
	std::tuple<T_value, T_key>& pivot = ptr[low + (high - low) / 2];

	T_key pivot_key = std::get<1>(pivot);

	long long int i = low - 1;
	long long int j = high + 1;

	while (true) {
		do {
			i++;
		} while (std::get<1>(ptr[i]) < pivot_key);

		do {
			j--;
		} while (std::get<1>(ptr[j]) > pivot_key);

		if (i >= j) {
			return j;
		}

		std::swap(ptr[i], ptr[j]);
	}
}

template <typename T_value, typename T_key>
void quicksort_keyed(std::tuple<T_value, T_key>* const __restrict ptr, long long int low, long long int high) {
	if (low >= high) {
		return;
	}

	long long int p = partition_keyed(ptr, low, high);
	quicksort_keyed(ptr, low, p);
	quicksort_keyed(ptr, p + 1, high);
}

template <typename T_value, typename T_key>
void quicksort_keyed(std::vector<std::tuple<T_value, T_key>>& vector) {
	long long int low = 0;
	long long int high = static_cast<long long int>(vector.size()) - 1;

	quicksort_keyed(vector.data(), low, high);
}

