#include <iostream>
#include <chrono>

const int ITERATIONS = 10000000000;

int multiplyBy8(int x) {
    return static_cast<int>(static_cast<long long>(x) * 8);
}

int shiftBy3(int x) {
    return static_cast<int>(static_cast<long long>(x) << 3);
}

int main() {
    int x = 42; // Example integer
    volatile int result;

    // Benchmark multiplication
    auto start = std::chrono::high_resolution_clock::now();
    for (long i = 0; i < ITERATIONS; ++i) {
        result = multiplyBy8(x);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationMultiply = end - start;
    std::cout << "Multiplication duration: " << durationMultiply.count() << " seconds" << std::endl;

    // Benchmark shift
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        result = shiftBy3(x);
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationShift = end - start;
    std::cout << "Shift duration: " << durationShift.count() << " seconds" << std::endl;

   
}