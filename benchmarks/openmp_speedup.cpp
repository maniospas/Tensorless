#include <iostream>
#include <chrono>
#include <omp.h>

// Function to perform 16 bitwise operations
void bitwise_operations(volatile int &value) {
    value ^= 0x5A5A5A5A;
    value |= 0xA5A5A5A5;
    value &= 0x3C3C3C3C;
    value ^= 0xC3C3C3C3;
    value |= 0x5A5A5A5A;
    value &= 0xA5A5A5A5;
    value ^= 0x3C3C3C3C;
    value |= 0xC3C3C3C3;
    value &= 0x5A5A5A5A;
    value ^= 0xA5A5A5A5;
    value |= 0x3C3C3C3C;
    value &= 0xC3C3C3C3;
    value ^= 0x5A5A5A5A;
    value |= 0xA5A5A5A5;
    value &= 0x3C3C3C3C;
    value ^= 0xC3C3C3C3;
}

void benchmark_traditional(int num_iterations) {
    volatile int value = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < 64; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < 8; i++) {
            bitwise_operations(value);
            bitwise_operations(value);
            volatile int value2 = 0;
            bitwise_operations(value2);
            bitwise_operations(value2);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Traditional C++ duration: " << duration.count() << " seconds\n";
}

void benchmark_openmp(int num_iterations) {
    volatile int value = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < num_iterations; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < 64; i++) {
            bitwise_operations(value);
            bitwise_operations(value);
            volatile int value2 = 0;
            bitwise_operations(value2);
            bitwise_operations(value2);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "OpenMP duration: " << duration.count() << " seconds\n";
}

int main() {
    int num_iterations = 8;

    benchmark_traditional(num_iterations);
    benchmark_openmp(num_iterations);

    return 0;
}
