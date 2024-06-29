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

#ifndef Int4_H
#define Int4_H

#include <iostream>
#include <vector>
#include <bitset>
#include <cstdlib>
#include <random>
#include "../vecutils.h"


namespace tensorless {
class Int4 {
private:
    VECTOR value;
    VECTOR value1;
    VECTOR value2;
    VECTOR value3;
    explicit Int4(VECTOR v, VECTOR v1, VECTOR v2, VECTOR v3) : value(v), value1(v1), value2(v2), value3(v3) {}
public:
    static Int4 random() {return Int4(lrand(), lrand(), lrand(), lrand());}
    static Int4 broadcast(int val) {
        if(val<0 || val>15)
            throw std::logic_error("can only set values in range [0,15], given "+std::to_string(val));
        VECTOR value;
        VECTOR value1;
        VECTOR value2;
        VECTOR value3;
        if(val>=8) {
            value3 = ~value3;
            val -= 8;
        }
        if(val>=4) {
            value2 = ~value2;
            val -= 4;
        }
        if(val>=2) {
            value1 = ~value1;
            val -= 2;
        }
        if(val)
            value = ~value;
        return Int4(value, value1, value2, value3);
    }

    static int num_params() {
        return 4;
    }
    
    static int num_bits() {
        return 4*VECTOR_SIZE;
    }

    Int4(const std::vector<int>& vec) : value(0), value1(0), value2(0), value3(0) {
        for (int i = 0; i < vec.size(); ++i) 
            if (vec[i]) 
                set(i, vec[i]);
    }
    
    Int4(const Int4 &other) : value(other.value), value1(other.value1), value2(other.value2), value3(other.value3) {}
    
    Int4() : value(0), value1(0), value2(0), value3(0) {}

    Int4 zerolike() const {
        return Int4();
    }

    Int4 zerolike(const VECTOR& mask) const {
        VECTOR notmask = ~mask;
        return Int4(value&notmask, value1&notmask, value2&notmask, value3&notmask);
    }

    const int size() const {
        return sizeof(VECTOR)*8;
    }

    Int4& operator=(const Int4 &other) {
        if (this != &other) {
            value = other.value;
            value1 = other.value1;
            value2 = other.value2;
            value3 = other.value3;
        }
        return *this;
    }

    explicit operator bool() const {
        return ANY(value) || ANY(value1) || ANY(value2) || ANY(value3);
    }

    friend std::ostream& operator<<(std::ostream &os, const Int4 &si) {
        os << "[" << si.get(0);
        for(int i=1;i<si.size();i++)
            os << "," << si.get(i);
        os << "]";
        return os;
    }

    const Int4& print(const std::string& text="") const {
        std::cout << text << *this << "\n";
        return *this;
    }

    const bool isZeroAt(int i) {
        VECTOR a = value | value1 | value2 | value3;
        return GETAT(a, i);
    }

    const int sum() {
        return bitcount(value) + bitcount(value1)<<1 + bitcount(value2)<<2 + bitcount(value3)<<3;
    }

    const int sum(VECTOR mask) {
        return bitcount(value&mask) + bitcount(value1&mask)<<1 + bitcount(value2&mask)<<2 + bitcount(value3&mask)<<3;
    }
    
    const int get(int i) const {
        int ret = GETAT(value, i);
        ret += GETAT(value1, i)*2;
        ret += GETAT(value2, i)*4;
        ret += GETAT(value3, i)*8;
        return ret;
    }

    Int4 half() const {
        return Int4(value1, value2, value3, 0);
    }
    
    inline static Int4 broadcastOnes(const VECTOR &mask) {
        return Int4(mask, 0, 0, 0);
    }

    template <typename RetNumber> inline __attribute__((always_inline)) RetNumber applyHalf(const RetNumber &number) const {
        return number.half(value3).eighth(value3).eighth(value2).quarter(value1).half(value);
    }

    template <typename RetNumber> inline __attribute__((always_inline)) RetNumber applyHalf(const RetNumber &number, const VECTOR &mask) const {
        return number.half(value3&mask).eighth(value3&mask).eighth(value2&mask).quarter(value1&mask).half(value&mask);
    }
    
    template <typename RetNumber> inline __attribute__((always_inline)) RetNumber applyTimes2(const RetNumber &number) const {
        return number.times2(value3).times2(value3).times2(value3).times2(value2).times2(value2).times2(value1);
    }

    template <typename RetNumber> inline __attribute__((always_inline)) RetNumber applyTimes2(const RetNumber &number, const VECTOR &mask) const {
        return number.times2(value3&mask).times2(value3&mask).times2(value3&mask).times2(value2&mask).times2(value2&mask).times2(value1&mask);
    }

    const Int4& set(int i, int val) {
        #ifdef DEBUG_SET
        if(size()<=i || i<0)
            throw std::logic_error("out of of range");
        if(val<0 || val>15)
            throw std::logic_error("can only set values in range [0,15], given "+std::to_string(val));
        #endif
        if(val>=8) {
            // #pragma omp atomic
            value3 |= ONEHOT(i);
            val -= 8;
        }
        else {
            // #pragma omp atomic
            value3 &= ~ONEHOT(i);
        }
        if(val>=4) {
            // #pragma omp atomic
            value2 |= ONEHOT(i);
            val -= 4;
        }
        else {
            // #pragma omp atomic
            value2 &= ~ONEHOT(i);
        }
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
    
    Int4& operator[](std::pair<int, int> p) {
        set(p.first, p.second);
        return *this;
    }

    inline __attribute__((always_inline)) int countNonZeros() { 
        VECTOR n = value | value1 | value2;
        return bitcount(n);
    }
    
    inline __attribute__((always_inline)) Int4 operator~() const {
        return Int4(~(value | value1 | value2 | value3), 0, 0, 0);
    }
    
    inline __attribute__((always_inline)) Int4 operator!=(const Int4 &other) const {
        return Int4((other.value^value) | (other.value1^value1) | (other.value2^value2) | (other.value3^value3), 0, 0, 0);
    }
 
    inline __attribute__((always_inline)) Int4 twosComplement(const VECTOR &mask) const {
        VECTOR notmask = ~mask;
        return Int4(mask&~value | (notmask&value), 
                      mask&~value1 | (notmask&value1),
                      mask&~value2 | (notmask&value2),
                      mask&~value3 | (notmask&value3)
        ).selfAdd(mask);
    }
    
    inline __attribute__((always_inline)) Int4 twosComplement() const {
        return Int4(~value, ~value1, ~value2, ~value3).addWithoutCarry(Int4(~(VECTOR)0,0,0,0));
    }

    inline __attribute__((always_inline)) Int4 twosComplementWithCarry(const VECTOR &mask, VECTOR &carry) const {
        VECTOR notmask = ~mask;
        return Int4(mask&~value | (notmask&value), 
                      mask&~value1 | (notmask&value1),
                      mask&~value2 | (notmask&value2),
                      mask&~value3 | (notmask&value3)
        ).addWithCarry(Int4(mask,0,0, 0), carry);
    }
    
    inline __attribute__((always_inline)) Int4 twosComplementWithCarry(VECTOR &carry) const {
        return Int4(~value, ~value1, ~value2, ~value3).addWithCarry(Int4(~(VECTOR)0,0,0,0), carry);
    }

    inline __attribute__((always_inline)) Int4 addWithoutCarry(const Int4 &other) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        return Int4(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1,
                    other.value3^value3^carry2
                    );
    }

    inline __attribute__((always_inline)) Int4 addWithCarry(const Int4 &other, VECTOR &lastcarry) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        lastcarry = (value3 & other.value3) | (carry2 & (value3 ^ other.value3));
        return Int4(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1,
                    other.value3^value3^carry2
                    );
    }

    inline __attribute__((always_inline)) Int4 operator+(const Int4 &other) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        return Int4(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1,
                    other.value3^value3^carry2
                    );
    }

    inline __attribute__((always_inline)) const Int4& operator+=(const Int4 &other) {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        value = other.value^value;
        value1 = other.value1^value1^carry;
        value2 = other.value2^value2^carry1;
        value3 = other.value3^value3^carry2;
        return *this;
    }

    inline __attribute__((always_inline)) const Int4& selfAdd(const VECTOR &other, const VECTOR &other1, const VECTOR &other2, const VECTOR &other3) {
        VECTOR carry = other&value;
        VECTOR carry1 = (value1 & other1) | (carry & (value1 ^ other1));
        VECTOR carry2 = (value2 & other2) | (carry1 & (value2 ^ other2));
        value = other^value;
        value1 = other1^value1^carry;
        value2 = other2^value2^carry1;
        value3 = other3^value3^carry2;
        return *this;
    }

    inline __attribute__((always_inline)) const Int4& selfAdd(const VECTOR &other, const VECTOR &other1, const VECTOR &other2) {
        VECTOR carry = other&value;
        VECTOR carry1 = (value1 & other1) | (carry & (value1 ^ other1));
        VECTOR carry2 = (value2 & other2) | (carry1 & (value2 ^ other2));
        value = other^value;
        value1 = other1^value1^carry;
        value2 = other2^value2^carry1;
        value3 = value3^carry2;
        return *this;
    }

    inline __attribute__((always_inline)) const Int4& selfAdd(const VECTOR &other, const VECTOR &other1) {
        VECTOR carry = other&value;
        VECTOR carry1 = (value1 & other1) | (carry & (value1 ^ other1));
        VECTOR carry2 = carry1 & value2;
        value = other^value;
        value1 = other1^value1^carry;
        value2 = value2^carry1;
        value3 = value3^carry2;
        return *this;
    }

    inline __attribute__((always_inline)) const Int4& selfAdd(const VECTOR &other) {
        VECTOR carry = other&value;
        VECTOR carry1 = carry & value1;
        VECTOR carry2 = carry1 & value2;
        value = other^value;
        value1 = value1^carry;
        value2 = value2^carry1;
        value3 = value3^carry2;
        return *this;
    }

    inline __attribute__((always_inline)) static int sup() {
        return 15;
    }

    inline __attribute__((always_inline)) static int eps() {
        return 1;
    }

    inline __attribute__((always_inline)) static int inf() {
        return 0;
    }
    
    inline __attribute__((always_inline)) Int4 merge(const Int4 &other, const VECTOR &mask) const {
        VECTOR notmask = ~mask;
        return Int4((value&mask) | (other.value & notmask), 
                    (value1&mask) | (other.value1 & notmask),
                    (value2&mask) | (other.value2 & notmask),
                    (value3&mask) | (other.value3 & notmask)
                    );
    }

    inline __attribute__((always_inline)) const int absmax() const {
        int ret = 0;
        VECTOR v2 = value2;
        VECTOR v1 = value1;
        VECTOR v = value;
        if(ANY(value3)) {
            ret += 8;
            v2 &= value3;
            v1 &= value3;
            v &= value3;
        }
        if(ANY(v2)) {
            ret += 4;
            v1 &= v2;
            v &= v2;
        }
        if(ANY(v1)) {
            ret += 2;
            v &= v1;
        }
        if(ANY(v))
            ret += 1;
        return ret;
    }
};


}

#endif // Int4_H
