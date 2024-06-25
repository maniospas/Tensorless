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

//#define DEBUG_OVERFLOWS  // enable for a slow but logically safe execution environment
//#define SUPERLONG

#include <iostream>
#include <vector>
#include <bitset>
#include <cstdlib>
#include <random>

namespace tensorless {

std::random_device rd;
std::mt19937_64 generator(rd());
std::uniform_int_distribution<long long> distribution(0, LONG_LONG_MAX);

#ifdef SUPERLONG
    #ifdef __SIZEOF_INT128__
        #define INTERNALVECTOR __int128 
    #else
        #define INTERNALVECTOR long long
    #endif
    class FourLongs {
        private:
            INTERNALVECTOR l1;
            INTERNALVECTOR l2;
            INTERNALVECTOR l3;
            INTERNALVECTOR l4;
        public:
            inline FourLongs(INTERNALVECTOR val): l1(val), l2(0), l3(0), l4(0) {}
            inline FourLongs(): l1(0), l2(0), l3(0), l4(0) {}
            inline FourLongs(const FourLongs& other): l1(other.l1), l2(other.l2), l3(other.l3), l4(other.l4) {}
            inline FourLongs& operator=(const FourLongs& other) {
                if (this == &other) {
                    return *this; // handle self assignment
                }
                l1 = other.l1;
                l2 = other.l2;
                l3 = other.l3;
                l4 = other.l4;
                return *this;
            }
            inline FourLongs(INTERNALVECTOR l1, INTERNALVECTOR l2, INTERNALVECTOR l3, INTERNALVECTOR l4): l1(l1), l2(l2), l3(l3), l4(l4) {}
            inline int count() const {
                return __builtin_popcountll(l1) + __builtin_popcountll(l2)+__builtin_popcountll(l3)+__builtin_popcountll(l4);
            }
            inline bool any() const {
                return l1 || l2 || l3 || l4;
            }
            inline FourLongs operator&(const FourLongs &other) const {
                return FourLongs(l1 & other.l1, l2 & other.l2, l3 & other.l3, l4 & other.l4);
            }
            inline FourLongs operator|(const FourLongs &other) const {
                return FourLongs(l1 | other.l1, l2 | other.l2, l3 | other.l3, l4 | other.l4);
            }
            inline FourLongs operator^(const FourLongs &other) const {
                return FourLongs(l1 ^ other.l1, l2 ^ other.l2, l3 ^ other.l3, l4 ^ other.l4);
            }
            inline FourLongs operator~() const {
                return FourLongs(~l1, ~l2, ~l3, ~l4);
            }
            inline FourLongs& operator&=(const FourLongs &other) {
                l1 &= other.l1;
                l2 &= other.l2;
                l3 &= other.l3;
                l4 &= other.l4;
                return *this;
            }
            inline FourLongs& operator|=(const FourLongs &other) {
                l1 |= other.l1;
                l2 |= other.l2;
                l3 |= other.l3;
                l4 |= other.l4;
                return *this;
            }
            inline FourLongs& operator^=(const FourLongs &other) {
                l1 ^= other.l1;
                l2 ^= other.l2;
                l3 ^= other.l3;
                l4 ^= other.l4;
                return *this;
            }
            inline int operator[](int index) const {
                /*if (index < 0 || index >= 256) {
                    throw std::out_of_range("Index out of range");
                }*/
                if (index < 64) 
                    return (l1 >> index) & 1;
                else if (index < 128) 
                    return (l2 >> (index - 64)) & 1;
                else if (index < 192) 
                    return (l3 >> (index - 128)) & 1;
                else 
                    return (l4 >> (index - 192)) & 1;
            }
            inline const FourLongs& toggleOn(int index) {
                /*if (index < 0 || index >= 256) {
                    throw std::out_of_range("Index out of range");
                }*/
                if (index < 64) 
                    l1 |= ((INTERNALVECTOR)1) << index;
                else if (index < 128) 
                    l2 |= ((INTERNALVECTOR)1) << (index-64);
                else if (index < 192) 
                    l3 |= ((INTERNALVECTOR)1) << (index-128);
                else 
                    l3 |= ((INTERNALVECTOR)1) << (index-192);
                return *this;
            }
    };


    #define VECTOR FourLongs 
    #define bitcount(x) ((x).count())  
    #define GETAT(x, i) (x)[i]
    #define ANY(x) (x).any()
    inline VECTOR lrand() {
        return FourLongs(distribution(generator),
                         distribution(generator),
                         distribution(generator),
                         distribution(generator));
    }
    #define ONEHOT(i) (FourLongs().toggleOn(i))
#else
#ifdef __SIZEOF_INT128__
    #define VECTOR __int128 
    #define GETAT(x, i) ((int)((x >> i) & 1))
    #define bitcount(x) (__builtin_popcountll(static_cast<uint64_t>(x))+__builtin_popcountll(static_cast<uint64_t>((x) >> 64)))
    #define ANY(x) (x)
    inline VECTOR lrand() {
        return (((VECTOR)distribution(generator))<<64 | (VECTOR)distribution(generator));
    }
    #define ONEHOT(i) (((VECTOR)1) << i)
#else
    #define VECTOR long long int
    #define GETAT(x, i) ((x >> i) & 1)
    #define bitcount(x) __builtin_popcountll(x)
    #define ANY(x) (x)
    inline VECTOR lrand() {
        return distribution(generator);
    }
    #define ONEHOT(i) (((VECTOR)1) << i)
#endif
#endif

#define VECTOR_SIZE (sizeof(VECTOR)*8);

}
#endif  // VECUTILS_H
