#pragma once
#include <vector>
#include <random>
#include <type_traits>

namespace hpc {

/// Generate a reproducible random vector of length n in [-1, 1].

template <typename T>
std::vector<T> make_random(std::size_t n, unsigned seed) {

    static_assert(std::is_floating_point<T>::value,
                  "make_random: T must be float/double");

    std::mt19937 rng(seed); // PRNG engine
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<T> v(n);
    
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = static_cast<T>(dist(rng));
    }

    return v;
}

}