#include <iostream>
#include <chrono>
#include "../tensorless/types/all.h"

// Define the functions using #define
#define INT_TO_BOOL(n) ((bool)(n))
#define POPCOUNT_NONZERO(n) (bitcount(n) != 0)

int main() {
    VECTOR num1 = 100000000;
    VECTOR num2 = 0;
    VECTOR num3 = 100000000; // 42 in binary
    VECTOR num4 = 0;

    const int repetitions = 10000000;
    volatile bool result1, result2;

    // Timing the int to bool cast
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repetitions; ++i) {
        result1 = INT_TO_BOOL(num1);
        result2 = INT_TO_BOOL(num2);
        result1 = INT_TO_BOOL(num1);
        result2 = INT_TO_BOOL(num2);
        result1 = INT_TO_BOOL(num1);
        result2 = INT_TO_BOOL(num2);
        result1 = INT_TO_BOOL(num1);
        result2 = INT_TO_BOOL(num2);
        result1 = INT_TO_BOOL(num1);
        result2 = INT_TO_BOOL(num2);
        result1 = INT_TO_BOOL(num1);
        result2 = INT_TO_BOOL(num2);
        result1 = INT_TO_BOOL(num1);
        result2 = INT_TO_BOOL(num2);
        result1 = INT_TO_BOOL(num1);
        result2 = INT_TO_BOOL(num2);
        result1 = INT_TO_BOOL(num1);
        result2 = INT_TO_BOOL(num2);
        result1 = INT_TO_BOOL(num1);
        result2 = INT_TO_BOOL(num2);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Casting int to bool took: " << duration.count() << " seconds for " << repetitions << " repetitions" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repetitions; ++i) {
        result1 = POPCOUNT_NONZERO(num3);
        result2 = POPCOUNT_NONZERO(num4);
        result1 = POPCOUNT_NONZERO(num1);
        result2 = POPCOUNT_NONZERO(num2);
        result1 = POPCOUNT_NONZERO(num1);
        result2 = POPCOUNT_NONZERO(num2);
        result1 = POPCOUNT_NONZERO(num1);
        result2 = POPCOUNT_NONZERO(num2);
        result1 = POPCOUNT_NONZERO(num1);
        result2 = POPCOUNT_NONZERO(num2);
        result1 = POPCOUNT_NONZERO(num3);
        result2 = POPCOUNT_NONZERO(num4);
        result1 = POPCOUNT_NONZERO(num1);
        result2 = POPCOUNT_NONZERO(num2);
        result1 = POPCOUNT_NONZERO(num1);
        result2 = POPCOUNT_NONZERO(num2);
        result1 = POPCOUNT_NONZERO(num1);
        result2 = POPCOUNT_NONZERO(num2);
        result1 = POPCOUNT_NONZERO(num1);
        result2 = POPCOUNT_NONZERO(num2);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "__builtin_popcountll took: " << duration.count() << " seconds for " << repetitions << " repetitions" << std::endl;

    return 0;
}
