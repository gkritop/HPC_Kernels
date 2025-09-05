cmake_minimum_required(VERSION 3.16)
project(hpc_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Default build type ----
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Main benchmark binary ----
add_executable(hpc_bench src/bench.cpp)
target_include_directories(hpc_bench PRIVATE ${CMAKE_SOURCE_DIR}/include)

if (MSVC)
    target_compile_options(hpc_bench PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(hpc_bench PRIVATE
        -O3 -march=native -mtune=native
        -ffast-math -funroll-loops
    )
endif()

# ---- Optional OpenMP support ----
option(USE_OPENMP "Enable OpenMP for parallel matmul" OFF)
if (USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(hpc_bench OpenMP::OpenMP_CXX)
        if (MSVC)
            target_compile_options(hpc_bench PRIVATE /openmp)
        endif()
    endif()
endif()

# ---- Unit tests (GoogleTest via FetchContent) ----
include(CTest)
if (BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(hpc_tests tests/test_kernels.cpp)
    target_include_directories(hpc_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(hpc_tests GTest::gtest_main)

    add_test(NAME hpc_tests COMMAND hpc_tests)
endif()
