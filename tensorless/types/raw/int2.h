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

#ifndef Int2_H
#define Int2_H

#include <iostream>
#include <vector>
#include <bitset>
#include <cstdlib>
#include <random>
#include "../vecutils.h"


namespace tensorless {
class Int2 {
private:
    VECTOR value;
    VECTOR value1;
    explicit Int2(VECTOR v, VECTOR v1) : value(v), value1(v1) {}
public:
    static Int2 random() {return Int2(lrand(), 0);}
    static Int2 broadcast(double value) {
        int val = (int)value;
        return Int2(val&1?~(VECTOR)0:0, val&2?~(VECTOR)0:0);
    }

    inline __attribute__((always_inline)) static Int2 broadcastOnes(const VECTOR &mask) {
        return Int2(mask, 0);
    }

    static int num_params() {
        return 2;
    }

    static int num_bits() {
        return 2*VECTOR_SIZE;
    }

    Int2(const std::vector<int>& vec) : value(0), value1(0) {
        for (int i = 0; i < vec.size(); ++i) 
            if (vec[i]) 
                set(i, vec[i]);
    }
    
    Int2(const Int2 &other) : value(other.value), value1(other.value1) {}
    
    Int2() : value(0), value1(0) {}

    const int size() const {
        return sizeof(VECTOR)*8;
    }

    Int2& operator=(const Int2 &other) {
        if (this != &other) {
            value = other.value;
            value1 = other.value1;
        }
        return *this;
    }

    explicit operator bool() const {
        return ANY(value) || ANY(value1);
    }
    
    friend std::ostream& operator<<(std::ostream &os, const Int2 &si) {
        os << "[" << si.get(0);
        for(int i=1;i<si.size();i++)
            os << "," << si.get(i);
        os << "]";
        return os;
    }

    const Int2& print(const std::string& text="") const {
        std::cout << text << *this << "\n";
        return *this;
    }

    const bool isZeroAt(int i) {
        VECTOR a = value | value1;
        return GETAT(a, i);
    }

    const int sum() {
        return bitcount(value) + bitcount(value1)*2;
    }

    const int sum(VECTOR mask) {
        return bitcount(value&mask) + bitcount(value1&mask)*2;
    }
    
    const int get(int i) const {
        return GETAT(value, i) + GETAT(value1, i)*2;
    }

    const Int2& set(int i, int val) {
        if(size()<=i || i<0)
            throw std::logic_error("out of of range");
        if(val<0 || val>3)
            throw std::logic_error("can only set values in range [0,3]");
        if(val>=2) {
            // #pragma omp atomic
            value1 |= ONEHOT(i);
            val -= 2;
        }
        else {
            // #pragma omp atomic
            value1 &= ~ONEHOT(i);
        }
        if(val) {
            // #pragma omp atomic
            value |= ONEHOT(i);
        }
        else {
            // #pragma omp atomic
            value &= ~ONEHOT(i);
        }
        return *this;
    }

    int operator[](int i) {
        return get(i);
    }

    int operator[](int i) const {
        return get(i);
    }
    
    Int2& operator[](std::pair<int, int> p) {
        set(p.first, p.second);
        return *this;
    }

    inline __attribute__((always_inline)) Int2 zerolike() const {
        return Int2();
    }

    inline __attribute__((always_inline)) Int2 zerolike(const VECTOR& mask) const {
        VECTOR notmask = ~mask;
        return Int2(value&notmask, value1&notmask);
    }

    inline __attribute__((always_inline)) int countNonZeros() { 
        VECTOR n = value | value1;
        return bitcount(n);
    }
    
    inline __attribute__((always_inline)) Int2 operator~() const {
        return Int2(~(value | value1), 0);
    }
    
    inline __attribute__((always_inline)) Int2 operator!=(const Int2 &other) const {
        return Int2((other.value^value) | (other.value1^value1), 0);
    }

    inline __attribute__((always_inline)) Int2 operator*(const Int2 &other) const {
        return Int2(other.value&value, (other.value1&value) | (other.value&value1));
    }

    inline __attribute__((always_inline)) Int2 addWithoutCarry(const Int2 &other) const {
        VECTOR carry = other.value&value;
        return Int2(other.value^value, 
                    other.value1^value1^carry
                    );
    }
 
    inline __attribute__((always_inline)) Int2 twosComplement(const VECTOR &mask) const {
        VECTOR notmask = ~mask;
        return Int2(mask&~value | (notmask&value), 
                      mask&~value1 | (notmask&value1)
        ).addWithoutCarry(Int2(mask,0));
    }
    
    inline __attribute__((always_inline)) Int2 twosComplement() const {
        return Int2(~value, ~value1).addWithoutCarry(Int2(~(VECTOR)0,0));
    }

    inline __attribute__((always_inline)) Int2 twosComplementWithCarry(const VECTOR &mask, VECTOR &carry) const {
        VECTOR notmask = ~mask;
        return Int2(mask&~value | (notmask&value), 
                      mask&~value1 | (notmask&value1)
        ).addWithCarry(Int2(mask,0), carry);
    }
    
    inline __attribute__((always_inline)) Int2 twosComplementWithCarry(VECTOR &carry) const {
        return Int2(~value, ~value1).addWithCarry(Int2(~(VECTOR)0,0), carry);
    }

    inline __attribute__((always_inline)) Int2 addWithCarry(const Int2 &other, VECTOR &lastcarry) const {
        VECTOR carry = other.value&value;
        lastcarry = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        return Int2(other.value^value, 
                    other.value1^value1^carry
                    );
    }

    inline __attribute__((always_inline)) Int2 maximum(const Int2 &other) const {
        VECTOR vSelector = value1 & ~other.value1;
        VECTOR v1 = value1 | other.value1;
        VECTOR v = (vSelector&value) | (other.value & ~vSelector);
        return Int2(v, v1);
    }

    inline __attribute__((always_inline)) Int2 minimum(const Int2 &other) const {
        VECTOR vSelector = ~value1 & other.value1;
        VECTOR v1 = value1 & other.value1;
        VECTOR v = (vSelector&value) | (other.value & ~vSelector);
        return Int2(v, v1);
    }

    inline __attribute__((always_inline)) Int2 merge(const Int2 &other, const VECTOR &mask) const {
        VECTOR notmask = ~mask;
        return Int2((value&mask) | (other.value & notmask), 
                    (value1&mask) | (other.value1 & notmask)
                    );
    }

    inline __attribute__((always_inline)) Int2 operator+(const Int2 &other) const {
        VECTOR carry = other.value&value;
        return Int2(other.value^value, 
                    other.value1^value1^carry
                    );
    }

    inline __attribute__((always_inline)) const Int2& operator+=(const Int2 &other) {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        value = other.value^value;
        value1 = other.value1^value1^carry;
        return *this;
    }

    inline __attribute__((always_inline)) const Int2& operator*=(const Int2 &other) {
        value1 = (other.value1&value) | (other.value&value1);
        value = other.value&value; 
        return *this;
    }

    inline __attribute__((always_inline)) static int sup() {
        return 3;
    }
    
    inline __attribute__((always_inline)) static int eps() {
        return 1;
    }

    inline __attribute__((always_inline)) static int inf() {
        return 0;
    }

    template <typename RetNumber> inline __attribute__((always_inline)) RetNumber applyHalf(const RetNumber &number) const {
        return number.quarter(value1).half(value);
    }

    template <typename RetNumber> inline __attribute__((always_inline)) RetNumber applyHalf(const RetNumber &number, const VECTOR &mask) const {
        return number.quarter(value1&mask).half(value&mask);
    }

    template <typename RetNumber> RetNumber inline __attribute__((always_inline)) applyTimes2(const RetNumber &number) const {
        return number.times2(value1);
    }

    template <typename RetNumber> RetNumber inline __attribute__((always_inline)) applyTimes2(const RetNumber &number, const VECTOR &mask) const {
        return number.times2(value1&mask);
    }
    
    inline __attribute__((always_inline)) Int2 half() const {
        return Int2(value1, 0);
    }

    inline __attribute__((always_inline)) const int absmax() const {
        int ret = 0;
        VECTOR v = value;
        if(ANY(value1)) {
            ret += 2;
            v &= value1;
        }
        if(ANY(v))
            ret += 1;
        return ret;
    }
};


}

#endif // Int2_H
