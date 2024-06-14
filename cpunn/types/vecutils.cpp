#ifndef VECUTILS_H
#define VECUTILS_H

#include <iostream>
#include <vector>
#include <bitset>
#include <cstdlib>
#include <random>

#define VECTOR __uint128_t // long long
#define ONEHOT(i) ((VECTOR)1 << i)


std::random_device rd;
std::mt19937_64 generator(rd());
std::uniform_int_distribution<VECTOR> distribution(0, LONG_LONG_MAX);
VECTOR lrand() {
    return distribution(generator);
}

#endif  // VECUTILS_H
