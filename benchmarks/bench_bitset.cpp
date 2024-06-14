#include <iostream>
#include <bitset>
#include <chrono>

const long long ITERATIONS = 10000000;

void benchmark_bitset() {
    std::bitset<64> b1("1100110011001100110011001100110011001100110011001100110011001100");
    std::bitset<64> b2("1010101010101010101010101010101010101010101010101010101010101010");
    std::bitset<64> result;

    auto start = std::chrono::high_resolution_clock::now();
    for (long long i = 0; i < ITERATIONS; ++i) {
        result = b1 & b2;  // AND operation
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Bitset AND operation took: " << duration.count() << " seconds" << std::endl;
}

void benchmark_long_long() {
    unsigned long long b1 = 0xCCCCCCCCCCCCCCCC;  // binary: 1100110011001100110011001100110011001100110011001100110011001100
    unsigned long long b2 = 0xAAAAAAAAAAAAAAAA;  // binary: 1010101010101010101010101010101010101010101010101010101010101010
    unsigned long long result;

    auto start = std::chrono::high_resolution_clock::now();
    for (long long i = 0; i < ITERATIONS; ++i) {
        result = b1 & b2;  // AND operation
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Long long AND operation took: " << duration.count() << " seconds" << std::endl;
}

int main() {
    benchmark_bitset();
    benchmark_long_long();
    return 0;
}
