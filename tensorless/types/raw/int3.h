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

#ifndef INT3_H
#define INT3_H

#include <iostream>
#include <vector>
#include <bitset>
#include <cstdlib>
#include <random>
#include "../vecutils.h"

namespace tensorless {

class Int3 {
private:
    VECTOR value;
    VECTOR value1;
    VECTOR value2;
    explicit Int3(VECTOR v, VECTOR v1, VECTOR v2) : value(v), value1(v1), value2(v2) {}

public:
    static Int3 random() {
        return Int3(lrand(), 0, 0);
    }

    inline __attribute__((always_inline)) static Int3 broadcastOnes(const VECTOR &mask) {
        return Int3(mask, 0, 0);
    }

    static Int3 broadcast(int val) {
        if (val < 0 || val > 7) {
            throw std::logic_error("can only set values in range [0,7], given " + std::to_string(val));
        }
        VECTOR value = 0, value1 = 0, value2 = 0;
        if (val >= 4) {
            value2 = ~value2;
            val -= 4;
        }
        if (val >= 2) {
            value1 = ~value1;
            val -= 2;
        }
        if (val) {
            value = ~value;
        }
        return Int3(value, value1, value2);
    }

    static int num_params() {
        return 3;
    }
    
    static int num_bits() {
        return 3 * VECTOR_SIZE;
    }

    Int3(const std::vector<int>& vec) : value(0), value1(0), value2(0) {
        for (int i = 0; i < vec.size(); ++i) {
            if (vec[i]) {
                set(i, vec[i]);
            }
        }
    }
    
    Int3(const Int3 &other) : value(other.value), value1(other.value1), value2(other.value2) {}
    
    Int3() : value(0), value1(0), value2(0) {}

    Int3 zerolike() const {
        return Int3();
    }

    inline __attribute__((always_inline)) Int3 zerolike(const VECTOR& mask) const {
        VECTOR notmask = ~mask;
        return Int3(value & notmask, value1 & notmask, value2 & notmask);
    }

    int size() const {
        return sizeof(VECTOR) * 8;
    }

    Int3& operator=(const Int3 &other) {
        if (this != &other) {
            value = other.value;
            value1 = other.value1;
            value2 = other.value2;
        }
        return *this;
    }

    explicit operator bool() const {
        return ANY(value) || ANY(value1) || ANY(value2);
    }

    friend std::ostream& operator<<(std::ostream &os, const Int3 &si) {
        os << "[" << si.get(0);
        for(int i = 1; i < si.size(); ++i) {
            os << "," << si.get(i);
        }
        os << "]";
        return os;
    }

    const Int3& print(const std::string& text = "") const {
        std::cout << text << *this << "\n";
        return *this;
    }

    inline __attribute__((always_inline)) bool isZeroAt(int i) const {
        VECTOR a = value | value1 | value2;
        return GETAT(a, i);
    }

    inline __attribute__((always_inline)) int sum() const {
        return bitcount(value) + (bitcount(value1) * 2) + (bitcount(value2) * 4);
    }

    inline __attribute__((always_inline)) int sum(VECTOR mask) const {
        return bitcount(value & mask) + (bitcount(value1 & mask) * 2) + (bitcount(value2 & mask) * 4);
    }
    
    int get(int i) const {
        return GETAT(value, i) + (GETAT(value1, i) * 2) + (GETAT(value2, i) * 4);
    }


    inline __attribute__((always_inline)) Int3 half() const {
        return Int3(value1, value2, 0);
    }

    template <typename RetNumber>
    inline __attribute__((always_inline)) RetNumber applyHalf(const RetNumber &number) const {
        return number.eighth(value2).quarter(value1).half(value);
    }

    template <typename RetNumber>
    inline __attribute__((always_inline)) RetNumber applyHalf(const RetNumber &number, const VECTOR &mask) const {
        return number.eighth(value2 & mask).quarter(value1 & mask).half(value & mask);
    }
    
    template <typename RetNumber>
    inline __attribute__((always_inline)) RetNumber applyTimes2(const RetNumber &number) const {
        return number.times2(value2).times2(value2).times2(value1);
    }

    template <typename RetNumber>
    inline __attribute__((always_inline)) RetNumber applyTimes2(const RetNumber &number, const VECTOR &mask) const {
        return number.times2(value2 & mask).times2(value2 & mask).times2(value1 & mask);
    }

    inline __attribute__((always_inline)) Int3& set(int i, int val) {
        if (size() <= i || i < 0) {
            throw std::logic_error("out of range");
        }
        if (val < 0 || val > 7) {
            throw std::logic_error("can only set values in range [0,7], given " + std::to_string(val));
        }
        if (val & 1) {
            // #pragma omp atomic
            value |= ONEHOT(i);
        } else {
            // #pragma omp atomic
            value &= ~ONEHOT(i);
        }
        if (val & 2) {
            // #pragma omp atomic
            value1 |= ONEHOT(i);
        } else {
            // #pragma omp atomic
            value1 &= ~ONEHOT(i);
        }
        if (val & 4) {
            // #pragma omp atomic
            value2 |= ONEHOT(i);
        } else {
            // #pragma omp atomic
            value2 &= ~ONEHOT(i);
        }
        return *this;
    }

    int operator[](int i) const {
        return get(i);
    }
    
    Int3& operator[](std::pair<int, int> p) {
        set(p.first, p.second);
        return *this;
    }

    inline __attribute__((always_inline)) int countNonZeros() const { 
        VECTOR n = value | value1 | value2;
        return bitcount(n);
    }
    
    inline __attribute__((always_inline)) Int3 operator~() const {
        return Int3(~value, ~value1, ~value2);
    }
    
    inline __attribute__((always_inline)) Int3 operator!=(const Int3 &other) const {
        return Int3((other.value ^ value), (other.value1 ^ value1), (other.value2 ^ value2));
    }

    inline __attribute__((always_inline)) Int3 operator*(const Int3 &other) const {
        return Int3(other.value & value, 
                    (other.value1 & value) | (other.value & value1),
                    (other.value2 & value) | (other.value & value2) | (other.value1 & value1));
    }
 
    inline __attribute__((always_inline)) Int3 twosComplement(const VECTOR &mask) const {
        VECTOR notmask = ~mask;
        return Int3(mask & ~value | (notmask & value), 
                    mask & ~value1 | (notmask & value1),
                    mask & ~value2 | (notmask & value2)
        ).addWithoutCarry(Int3(mask, 0, 0));
    }
    
    inline __attribute__((always_inline)) Int3 twosComplement() const {
        return Int3(~value, ~value1, ~value2).addWithoutCarry(Int3(~(VECTOR)0, 0, 0));
    }

    inline __attribute__((always_inline)) Int3 twosComplementWithCarry(const VECTOR &mask, VECTOR &carry) const {
        VECTOR notmask = ~mask;
        return Int3(mask & ~value | (notmask & value), 
                    mask & ~value1 | (notmask & value1),
                    mask & ~value2 | (notmask & value2)
        ).addWithCarry(Int3(mask, 0, 0), carry);
    }
    
    inline __attribute__((always_inline)) Int3 twosComplementWithCarry(VECTOR &carry) const {
        return Int3(~value, ~value1, ~value2).addWithCarry(Int3(~(VECTOR)0, 0, 0), carry);
    }

    inline __attribute__((always_inline)) Int3 addWithoutCarry(const Int3 &other) const {
        VECTOR carry = value & other.value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        return Int3(value ^ other.value, 
                    value1 ^ other.value1 ^ carry,
                    value2 ^ other.value2 ^ carry1);
    }

    inline __attribute__((always_inline)) Int3 addWithCarry(const Int3 &other, VECTOR &lastcarry) const {
        VECTOR carry = value & other.value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        lastcarry = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        return Int3(value ^ other.value, 
                    value1 ^ other.value1 ^ carry,
                    value2 ^ other.value2 ^ carry1);
    }

    inline __attribute__((always_inline)) Int3 operator+(const Int3 &other) const {
        VECTOR carry = value & other.value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        return Int3(value ^ other.value, 
                    value1 ^ other.value1 ^ carry,
                    value2 ^ other.value2 ^ carry1);
    }

    inline __attribute__((always_inline)) Int3& operator+=(const Int3 &other) {
        VECTOR carry = value & other.value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        value ^= other.value;
        value1 ^= other.value1 ^ carry;
        value2 ^= other.value2 ^ carry1;
        return *this;
    }

    inline __attribute__((always_inline)) Int3& operator*=(const Int3 &other) {
        value2 = (other.value2 & value) | (other.value & value2) | (other.value1 & value1);
        value1 = (other.value1 & value) | (other.value & value1);
        value &= other.value; 
        return *this;
    }
    
    inline __attribute__((always_inline)) Int3 merge(const Int3 &other, const VECTOR &mask) const {
        VECTOR notmask = ~mask;
        return Int3((value&mask) | (other.value & notmask), 
                    (value1&mask) | (other.value1 & notmask),
                    (value2&mask) | (other.value2 & notmask)
                    );
    }

    inline __attribute__((always_inline)) static int sup() {
        return 7;
    }

    inline __attribute__((always_inline)) static int eps() {
        return 1;
    }

    inline __attribute__((always_inline)) static int inf() {
        return 0;
    }

    inline __attribute__((always_inline)) int absmax() const {
        int ret = 0;
        VECTOR v1 = value1;
        VECTOR v = value;
        if (ANY(value2)) {
            ret += 4;
            v1 &= value2;
            v &= value2;
        }
        if (ANY(v1)) {
            ret += 2;
            v &= v1;
        }
        if (ANY(v)) {
            ret += 1;
        }
        return ret;
    }
};

} // namespace tensorless

#endif // INT3_H
