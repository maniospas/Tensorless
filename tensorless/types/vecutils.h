/*
Copyright 2024 Emmanouil Krasanakis

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef VECUTILS_H
#define VECUTILS_H

#include <iostream>
#include <vector>
#include <bitset>
#include <cstdlib>
#include <random>

namespace tensorless {

#ifdef __SIZEOF_INT128__ 
    #define VECTOR __int128 
#else
    #define VECTOR long long int
#endif

#define ONEHOT(i) ((VECTOR)1 << i)

std::random_device rd;
std::mt19937_64 generator(rd());
std::uniform_int_distribution<VECTOR> distribution(0, LONG_LONG_MAX);
inline VECTOR lrand() {
    return distribution(generator);
}

}
#endif  // VECUTILS_H
