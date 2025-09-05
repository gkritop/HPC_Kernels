#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cmath>

#include "hpc/matmul.hpp"
#include "hpc/reduction.hpp"
#include "hpc/scan.hpp"


TEST(Matmul, Small3x4x2) {
    using T = double;
    std::size_t M = 3, N = 4, K = 2;

    std::vector<T> A = {
        1,2,
        3,4,
        5,6
    }; // MxK = 3x2

    std::vector<T> B = {
        7,8,9,10,
        11,12,13,14
    }; // KxN = 2x4

    std::vector<T> C;
    hpc::matmul_naive<T>(M, N, K, A, B, C);

    std::vector<T> ref = {
        29, 32, 35, 38,
        65, 72, 79, 86,
        101,112,123,134
    }; // expected MxN = 3x4

    ASSERT_EQ(C.size(), ref.size());
    for (size_t i = 0; i < C.size(); ++i) {
        EXPECT_DOUBLE_EQ(C[i], ref[i]);
    }
}

TEST(Reduction, KahanVsStd) {
    using T = double;

    // Construct adversarial input alternating large ± values + tiny offsets
    std::vector<T> x(1000);
    for (int i = 0; i < 1000; ++i) {
        x[i] = (i % 2 == 0 ? 1e8 : -1e8) + 1.0 / (i + 1);
    }

    double s_kahan = hpc::kahan_sum<T>(x);
    // std::cout << "debug: kahan=" << s_kahan << " naive=" << s_naive << std::endl; // debug leftover
    double s_naive = std::accumulate(x.begin(), x.end(), 0.0);

    EXPECT_TRUE(std::isfinite(s_kahan));
    EXPECT_TRUE(std::isfinite(s_naive));

    // They should be close, but not necessarily bitwise equal.
    // Kahan should reduce catastrophic cancellation compared to naive sum.
    EXPECT_NEAR(s_kahan, s_naive, 1e4);
}

TEST(Scan, InclusiveSmall) {
    using T = int;

    std::vector<T> x = {1, 2, 3, 4, 5};
    hpc::inclusive_scan_inplace<T>(x);

    std::vector<T> ref = {1, 3, 6, 10, 15};
    EXPECT_EQ(x, ref);
}