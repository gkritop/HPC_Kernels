#pragma once
#include <vector>
#include <type_traits>

namespace hpc {

// Compute the sum of a vector using Kahan compensated summation.

template <typename T>
T kahan_sum(const std::vector<T>& x) {
    
    static_assert(std::is_floating_point<T>::value,
                  "kahan_sum: T must be float or double");

    T sum = 0;
    T c   = 0;

    for (auto v : x) {
        T y = v - c;
        T t = sum + y;
        c   = (t - sum) - y;
        sum = t;
    }
    return sum;
}

}