#pragma once
#include <vector>
#include <cstddef>
#include <type_traits>
#include <cassert>
#include <algorithm>

namespace hpc {

/// Naive matrix multiply (i–k–j loop).
/// A(M×K) · B(K×N) = C(M×N), row-major.

template <typename T>
void matmul_naive(std::size_t M, std::size_t N, std::size_t K,
                  const std::vector<T>& A,
                  const std::vector<T>& B,
                  std::vector<T>& C)
                  
{
    static_assert(std::is_floating_point<T>::value, "matmul_naive: T must be float or double");

    assert(A.size() == M * K);
    assert(B.size() == K * N);

    C.assign(M * N, T(0));

    for (std::size_t i = 0; i < M; ++i) {
        // NOTE: using i-k-j loop order for better cache reuse; tried i-j-k but slower in my tests
        for (std::size_t k = 0; k < K; ++k) {
            const T aik = A[i * K + k];

            for (std::size_t j = 0; j < N; ++j) {
                C[i * N + j] += aik * B[k * N + j];
            }
        }
    }
}

/// Cache-blocked matrix multiply (ijk with block tiling).
/// BS = block size (default 128).
template <typename T>
void matmul_blocked(std::size_t M, std::size_t N, std::size_t K,
                    const std::vector<T>& A,
                    const std::vector<T>& B,
                    std::vector<T>& C,
                    std::size_t BS = 128)
{
    static_assert(std::is_floating_point<T>::value,
                  "matmul_blocked: T must be float or double");

    assert(A.size() == M * K);
    assert(B.size() == K * N);

    C.assign(M * N, T(0));

    for (std::size_t ii = 0; ii < M; ii += BS) {
        const std::size_t iimax = std::min(ii + BS, M);

        for (std::size_t kk = 0; kk < K; kk += BS) {
            const std::size_t kkmax = std::min(kk + BS, K);

            for (std::size_t jj = 0; jj < N; jj += BS) {
                const std::size_t jjmax = std::min(jj + BS, N);

                // NOTE: Parallelize with OpenMP,
                // #pragma omp parallel for
                for (std::size_t i = ii; i < iimax; ++i) {
                    for (std::size_t k = kk; k < kkmax; ++k) {
                        const T aik = A[i * K + k];
                        for (std::size_t j = jj; j < jjmax; ++j) {
                            C[i * N + j] += aik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

}