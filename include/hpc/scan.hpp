#pragma once
#include <vector>
#include <cstddef>
#include <type_traits>

namespace hpc {

/// In-place inclusive scan (prefix sum).
/// After call, x[i] = sum_{j=0..i} original_x[j].
template <typename T>
void inclusive_scan_inplace(std::vector<T>& x) {

    static_assert(std::is_arithmetic<T>::value,
                  "inclusive_scan_inplace: T must be arithmetic");

    T acc = 0;
    
    for (std::size_t i = 0; i < x.size(); ++i) {
        acc += x[i];
        x[i] = acc;
    }
}

}