#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <ctime>

#include "hpc/matmul.hpp"
#include "hpc/reduction.hpp"
#include "hpc/scan.hpp"
#include "hpc/timer.hpp"
#include "hpc/csv.hpp"
#include "hpc/rand.hpp"


struct Args {
    std::string op = "matmul";           // operation: matmul, reduction, scan
    size_t M = 1024, N = 1024, K = 1024; // matrix dimensions
    size_t size = 1 << 24;               // vector size (for reduction/scan)
    int reps = 7;                        // repetitions (median taken)
    std::string dtype = "float";         // float or double
    unsigned seed = 42u;                 // RNG seed
    std::string out = "results.csv";     // output CSV
    bool blocked = false;                // use blocked matmul
};

static bool starts_with(const char* s, const char* k) {
    return std::strncmp(s, k, std::strlen(k)) == 0;
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (starts_with(argv[i], "--op=")) a.op = std::string(argv[i] + 5);
        else if (starts_with(argv[i], "--M=")) a.M = std::stoull(argv[i] + 4);
        else if (starts_with(argv[i], "--N=")) a.N = std::stoull(argv[i] + 4);
        else if (starts_with(argv[i], "--K=")) a.K = std::stoull(argv[i] + 4);
        else if (starts_with(argv[i], "--size=")) a.size = std::stoull(argv[i] + 7);
        else if (starts_with(argv[i], "--reps=")) a.reps = std::stoi(argv[i] + 7);
        else if (starts_with(argv[i], "--dtype=")) a.dtype = std::string(argv[i] + 8);
        else if (starts_with(argv[i], "--seed=")) a.seed = static_cast<unsigned>(std::stoul(argv[i] + 7));
        else if (starts_with(argv[i], "--out=")) a.out = std::string(argv[i] + 6);
        else if (std::strcmp(argv[i], "--blocked") == 0) a.blocked = true;
        else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: hpc_bench --op=matmul|reduction|scan "
                         "[--M=] [--N=] [--K=] [--size=] "
                         "[--reps=] [--dtype=float|double] "
                         "[--seed=] [--out=path] [--blocked]\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown arg: " << argv[i] << "\n";
            std::exit(1);
        }
    }
    return a;
}

template <class T>
double checksum_vec(const std::vector<T>& v) {
    long double s = 0.0L;
    for (auto x : v) s += static_cast<long double>(x);
    return static_cast<double>(s);
}

template <class T>
void bench_matmul(const Args& a) {
    using namespace hpc;

    auto A = make_random<T>(a.M * a.K, a.seed);
    auto B = make_random<T>(a.K * a.N, a.seed + 1);
    std::vector<T> C(a.M * a.N);

    const char* op_label = a.blocked ? "matmul_blocked" : "matmul_naive";

    // warm-up
    if (a.blocked) matmul_blocked<T>(a.M, a.N, a.K, A, B, C, 128);
    else matmul_naive<T>(a.M, a.N, a.K, A, B, C);

    // measure
    std::vector<double> times(a.reps);

    for (int r = 0; r < a.reps; ++r) {
        Timer t; t.start();

        if (a.blocked) matmul_blocked<T>(a.M, a.N, a.K, A, B, C, 128);
        else matmul_naive<T>(a.M, a.N, a.K, A, B, C);

        times[r] = t.stop_s();
    }

    std::sort(times.begin(), times.end());
    double t_med = times[times.size() / 2];

    // metrics
    double flops = 2.0 * (double)a.M * (double)a.N * (double)a.K;
    double gflops = (flops / t_med) / 1e9;
    double bytes = sizeof(T) * ((double)a.M * a.K + (double)a.K * a.N + 2.0 * (double)a.M * a.N);
    double gbps = (bytes / t_med) / 1e9;
    double sumC = checksum_vec(C);

    // csv
    std::time_t ts = std::time(nullptr);

    csv_write_header_if_new(a.out,
        "timestamp,op,M,N,K,size,dtype,reps,ns_per_rep,gflops,gbps,checksum");

    char line[512];

    std::snprintf(line, sizeof(line),
        "%lld,%s,%zu,%zu,%zu,0,%s,%d,%.0f,%.6f,%.6f,%.17g",
        (long long)ts, op_label, a.M, a.N, a.K, a.dtype.c_str(), a.reps,
        t_med * 1e9, gflops, gbps, sumC);

    csv_append_line(a.out, line);

    std::cout << "[" << op_label << "] median " << (t_med * 1e3) << " ms, "
              << gflops << " GF/s, " << gbps << " GB/s, checksum=" << sumC << "\n";
}

template <class T>
void bench_reduction(const Args& a) {
    using namespace hpc;

    auto x = make_random<T>(a.size, a.seed);
    volatile T sink = 0; // avoid DCE

    // warm-up
    sink = kahan_sum<T>(x);

    // measure
    std::vector<double> times(a.reps);

    for (int r = 0; r < a.reps; ++r) {
        Timer t; t.start();

        sink = kahan_sum<T>(x);
        times[r] = t.stop_s();
    }

    std::sort(times.begin(), times.end());
    double t_med = times[times.size() / 2];

    double flops = (double)a.size - 1.0;
    double gflops = (flops / t_med) / 1e9;
    double bytes = sizeof(T) * (double)a.size;
    double gbps = (bytes / t_med) / 1e9;
    double chk = (double)sink;

    std::time_t ts = std::time(nullptr);

    csv_write_header_if_new(a.out,
        "timestamp,op,M,N,K,size,dtype,reps,ns_per_rep,gflops,gbps,checksum");
    
    char line[512];

    std::snprintf(line, sizeof(line),
        "%lld,reduction,0,0,0,%zu,%s,%d,%.0f,%.6f,%.6f,%.17g",
        (long long)ts, a.size, a.dtype.c_str(), a.reps,
        t_med * 1e9, gflops, gbps, chk);
    
    csv_append_line(a.out, line);

    std::cout << "[reduction] median " << (t_med * 1e3) << " ms, "
              << gflops << " GF/s, " << gbps << " GB/s, checksum=" << chk << "\n";
}

template <class T>
void bench_scan(const Args& a) {
    using namespace hpc;

    auto x = make_random<T>(a.size, a.seed);

    // warm-up
    inclusive_scan_inplace<T>(x);

    // measure
    std::vector<double> times(a.reps);

    for (int r = 0; r < a.reps; ++r) {
        auto tmp = x; // fresh copy each rep to simulate write traffic
        Timer t; t.start();
        inclusive_scan_inplace<T>(tmp);
        times[r] = t.stop_s();
    }

    std::sort(times.begin(), times.end());
    double t_med = times[times.size() / 2];

    double flops  = (double)a.size; // approx
    double gflops = (flops / t_med) / 1e9;
    double bytes  = sizeof(T) * 2.0 * (double)a.size; // read+write
    double gbps   = (bytes / t_med) / 1e9;
    double chk    = std::accumulate(x.begin(), x.end(), 0.0);

    std::time_t ts = std::time(nullptr);

    csv_write_header_if_new(a.out,
        "timestamp,op,M,N,K,size,dtype,reps,ns_per_rep,gflops,gbps,checksum");

    char line[512];

    std::snprintf(line, sizeof(line),
        "%lld,scan,0,0,0,%zu,%s,%d,%.0f,%.6f,%.6f,%.17g",
        (long long)ts, a.size, a.dtype.c_str(), a.reps,
        t_med * 1e9, gflops, gbps, chk);

    csv_append_line(a.out, line);

    std::cout << "[scan] median " << (t_med * 1e3) << " ms, "
              << gflops << " GF/s, " << gbps << " GB/s, checksum=" << chk << "\n";
}

int main(int argc, char** argv) {
    auto a = parse(argc, argv);
    bool is_float = (a.dtype == "float");

    if (a.op == "matmul") {
        if (is_float) bench_matmul<float>(a);
        else bench_matmul<double>(a);
    } else if (a.op == "reduction") {
        if (is_float) bench_reduction<float>(a);
        else bench_reduction<double>(a);
    } else if (a.op == "scan") {
        if (is_float) bench_scan<float>(a);
        else bench_scan<double>(a);
    } else {
        std::cerr << "Unknown --op: " << a.op << "\n";
        return 2;
    }

    return 0;
}
