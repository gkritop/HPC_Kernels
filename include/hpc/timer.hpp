#pragma once
#include <chrono>

namespace hpc {

struct Timer {
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;

    time_point t0{};

    void start() { t0 = clock::now(); }

    double stop_s() const {
        const auto t1 = clock::now();
        return std::chrono::duration<double>(t1 - t0).count();
    }
};

}