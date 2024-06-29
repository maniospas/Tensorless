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

#ifndef Float7_H
#define Float7_H

#include <iostream>
#include <vector>
#include <bitset>
#include <cstdlib>
#include <random>
#include "../vecutils.h"

namespace tensorless {

class Float7 {
private:
    VECTOR value;
    VECTOR value1;
    VECTOR value2;
    VECTOR value3;
    VECTOR value4;
    VECTOR value5;
    VECTOR value6;
    explicit Float7(VECTOR v, VECTOR v1, VECTOR v2, VECTOR v3, VECTOR v4, VECTOR v5, VECTOR v6) : 
        value(v), value1(v1), value2(v2), value3(v3), value4(v4), value5(v5), value6(v6) {}
public:
    static inline __attribute__((always_inline)) Float7 random() {return Float7(lrand(), lrand(), lrand(), lrand(), lrand(), lrand(), 0);}
    
    static inline __attribute__((always_inline)) Float7 broadcast(double val) {
        if(val<0 || val>1)
            throw std::logic_error("can only set values in range [0,1]");
        VECTOR value6 = 0;
        VECTOR value5 = 0;
        VECTOR value4 = 0;
        VECTOR value3 = 0;
        VECTOR value2 = 0;
        VECTOR value1 = 0;
        VECTOR value = 0;
        if(val>=0.5) {
            value6 = ~value6;
            val -= 0.5;
        }
        if(val>=0.25) {
            value5 = ~value5;
            val -= 0.25;
        }
        if(val>=0.125) {
            value4 = ~value4;
            val -= 0.125;
        }
        if(val>=0.0625) {
            value3 = ~value3;
            val -= 0.0625;
        }
        if(val>=0.03125) {
            value2 = ~value2;
            val -= 0.03125;
        }
        if(val>=0.015625) {
            value1 = ~value1;
            val -= 0.015625;
        }
        if(val>=0.0078125/2)
            value = ~value;
        return Float7(value, value1, value2, value3, value4, value5, value6);
    }


    inline __attribute__((always_inline)) Float7(const std::vector<double>& vec) : value(0), value1(0), value2(0), value3(0), value4(0), value5(0), value6(0) {
        for (int i = 0; i < vec.size(); ++i) 
            if (vec[i]) 
                set(i, vec[i]);
    }
    
    inline __attribute__((always_inline)) Float7(const Float7 &other) : 
        value(other.value), value1(other.value1), value2(other.value2), value3(other.value3), value4(other.value4), value5(other.value5), value6(other.value6) {}
    
    inline __attribute__((always_inline)) Float7() : value(0), value1(0), value2(0), value3(0), value4(0), value5(0), value6(0) {}

    inline __attribute__((always_inline)) Float7& operator=(const Float7 &other) {
        if (this != &other) {
            value = other.value;
            value1 = other.value1;
            value2 = other.value2;
            value3 = other.value3;
            value4 = other.value4;
            value5 = other.value5;
            value6 = other.value6;
        }
        return *this;
    }

    inline __attribute__((always_inline)) Float7 times2() const {
        #ifdef DEBUG_OVERFLOWS
        if(ANY(value6))
            throw std::logic_error("arithmetic overflow");
        #endif
        return Float7(0, value, value1, value2, value3, value4, value5);
    }

    inline __attribute__((always_inline)) Float7 times2(const VECTOR &mask) const {
        #ifdef DEBUG_OVERFLOWS
        if(ANY(value6 & mask))
            throw std::logic_error("arithmetic overflow");
        #endif
        VECTOR notmask = ~mask;
        return Float7(                   notmask & value, 
                      (mask & value)  | (notmask & value1),
                      (mask & value1) | (notmask & value2),
                      (mask & value2) | (notmask & value3),
                      (mask & value3) | (notmask & value4),
                      (mask & value4) | (notmask & value5),
                      (mask & value5) | (notmask & value6)
                      );
    }

    inline __attribute__((always_inline)) Float7 half() const {
        return Float7(value1, value2, value3, value4, value5, value6, 0);
    }

    inline __attribute__((always_inline)) Float7 half(const VECTOR &mask) const {
        VECTOR notmask = ~mask;
        return Float7(
                      (mask & value1) | (value & notmask), 
                      (mask & value2) | (value1 & notmask), 
                      (mask & value3) | (value2 & notmask), 
                      (mask & value4) | (value3 & notmask), 
                      (mask & value5) | (value4 & notmask), 
                      (mask & value6) | (value5 & notmask), 
                                         value6 & notmask);
    }
    
    inline __attribute__((always_inline)) Float7 quarter() const {
        return Float7(value2, value3, value4, value5, value6, 0, 0);
    }

    inline __attribute__((always_inline)) Float7 quarter(const VECTOR &mask) const {
        VECTOR notmask = ~mask;
        return Float7(
                      (mask & value2) | (value & notmask), 
                      (mask & value3) | (value1 & notmask), 
                      (mask & value4) | (value2 & notmask), 
                      (mask & value5) | (value3 & notmask), 
                      (mask & value6) | (value4 & notmask), 
                                         value5 & notmask, 
                                         value6 & notmask);
    }
    
    inline __attribute__((always_inline)) Float7 eighth() const {
        return Float7(value3, value4, value5, value6, 0, 0, 0);
    }

    inline __attribute__((always_inline)) Float7 eighth(const VECTOR &mask) const {
        VECTOR notmask = ~mask;
        return Float7(
                      (mask & value3) | (value & notmask), 
                      (mask & value4) | (value1 & notmask), 
                      (mask & value5) | (value2 & notmask), 
                      (mask & value6) | (value3 & notmask), 
                                         value4 & notmask, 
                                         value5 & notmask, 
                                         value6 & notmask);
    }

    inline __attribute__((always_inline)) static int num_params() {
        return 7;
    }

    inline __attribute__((always_inline)) static int num_bits() {
        return 7*VECTOR_SIZE;
    }
    
    inline __attribute__((always_inline)) explicit operator bool() const {
        return ANY(value) || ANY(value1) || ANY(value2) || ANY(value3) 
            || ANY(value4) || ANY(value5) || ANY(value6);
    }

    inline __attribute__((always_inline)) friend std::ostream& operator<<(std::ostream &os, const Float7 &si) {
        os << "[" << si.get(0);
        for(int i=1;i<si.size();i++)
            os << "," << si.get(i);
        os << "]";
        return os;
    }

    inline __attribute__((always_inline)) const Float7& print(const std::string& text="") const {
        std::cout << text << *this << "\n";
        return *this;
    }

    inline __attribute__((always_inline)) const int size() const {
        return VECTOR_SIZE;
    }

    inline __attribute__((always_inline)) const bool isZeroAt(int i) {
        VECTOR a = value | value1 | value2 | value3 | value4 | value5 | value6;
        return GETAT(a, i);
    }
    
    inline __attribute__((always_inline)) const double absmax() const {
        int ret = 0;
        VECTOR mask = 0;
        if(ANY(value6)) {
            ret += 64;
            mask = value6;
        } 
        else {
            mask = ~mask;
        }
        VECTOR v5 = value5 & mask;
        if(ANY(v5)) {
            ret += 32;
            mask = v5;
        }
        VECTOR v4 = value4 & mask;
        if(ANY(v4)) {
            ret += 16;
            mask = v4;
        }
        VECTOR v3 = value3 & mask;
        if(ANY(v3)) {
            ret += 8;
            mask = v3;
        }
        VECTOR v2 = value2 & mask;
        if(ANY(v2)) {
            ret += 4;
            mask = v2;
        }
        VECTOR v1 = value1 & mask;
        VECTOR v = value;
        if(ANY(v1)) {
            ret += 2;
            v &= v1;
        }
        if(ANY(v)) {
            ret += 1;
        }
        return ret/64.0;
    }

    inline __attribute__((always_inline)) const double sum() const {
        int ret = bitcount(value);
        ret += bitcount(value1)*2;
        ret += bitcount(value2)*4;
        ret += bitcount(value3)*8;
        ret += bitcount(value4)*16;
        ret += bitcount(value5)*32;
        ret += bitcount(value6)*64;
        return ret/64.0;
    }

    inline __attribute__((always_inline)) const double sum(const VECTOR &mask) const {
        int ret = bitcount(value & mask);
        ret += bitcount(value1 & mask)*2;
        ret += bitcount(value2 & mask)*4;
        ret += bitcount(value3 & mask)*8;
        ret += bitcount(value4 & mask)*16;
        ret += bitcount(value5 & mask)*32;
        ret += bitcount(value6 & mask)*64;
        return ret/64.0;
    }
    
    inline __attribute__((always_inline)) const double get(int i) const {
        int ret = GETAT(value, i);
        ret += GETAT(value1, i)*2;
        ret += GETAT(value2, i)*4;
        ret += GETAT(value3, i)*8;
        ret += GETAT(value4, i)*16;
        ret += GETAT(value5, i)*32;
        ret += GETAT(value6, i)*64;
        return ret/64.0;
    }

    inline __attribute__((always_inline)) const Float7& set(int i, double val) {
        #ifdef DEBUG_SET
        if(size()<=i || i<0)
            throw std::logic_error("out of of range");
        if(val<0 || val>1)
            throw std::logic_error("can only set values in range [0,1]");
        #endif
        if(val>=0.5) {
            // #pragma omp atomic
            value6 |= ONEHOT(i);
            val -= 0.5;
        }
        else {
            // #pragma omp atomic
            value6 &= ~ONEHOT(i);
        }
        if(val>=0.25) {
            // #pragma omp atomic
            value5 |= ONEHOT(i);
            val -= 0.25;
        }
        else {
            // #pragma omp atomic
            value5 &= ~ONEHOT(i);
        }
        if(val>=0.125){
            // #pragma omp atomic
            value4 |= ONEHOT(i);
            val -= 0.125;
        }
        else {
            // #pragma omp atomic
            value4 &= ~ONEHOT(i);
        }
        if(val>=0.0625) {
            // #pragma omp atomic
            value3 |= ONEHOT(i);
            val -= 0.0625;
        }
        else {
            // #pragma omp atomic
            value3 &= ~ONEHOT(i);
        }
        if(val>=0.03125) {
            // #pragma omp atomic
            value2 |= ONEHOT(i);
            val -= 0.03125;
        }
        else {
            // #pragma omp atomic
            value2 &= ~ONEHOT(i);
        }
        if(val>=0.015625) {
            // #pragma omp atomic
            value1 |= ONEHOT(i);
            val -= 0.015625;
        }
        else {
            // #pragma omp atomic
            value1 &= ~ONEHOT(i);
        }
        if(val>=0.0078125/2) {
            // #pragma omp atomic
            value |= ONEHOT(i);
        }
        else {
            // #pragma omp atomic
            value &= ~ONEHOT(i);
        }
        return *this;
    }

    inline __attribute__((always_inline)) double operator[](int i) {
        return get(i);
    }

    inline __attribute__((always_inline)) double operator[](int i) const {
        return get(i);
    }
    
    inline __attribute__((always_inline)) Float7& operator[](std::pair<int, double> p) {
        set(p.first, p.second);
        return *this;
    }

    inline __attribute__((always_inline)) int countNonZeros() { 
        VECTOR n = value | value1 | value2 | value3 | value4 | value5 | value6;
        return bitcount(n);
    }

    inline __attribute__((always_inline)) Float7 operator~() const {
        return Float7(0, 0, 0, 0, 0, 0, ~(value | value1 | value2 | value3 | value4 | value5 | value6));
    }

    inline __attribute__((always_inline)) Float7 operator!=(const Float7 &other) const {
        return Float7(0, 0, 0, 0, 0, 0, 
            (other.value^value) | (other.value1^value1) | (other.value2^value2) | (other.value3^value3) | (other.value4^value4) | (other.value5^value5) | (other.value6^value6));
    }

    inline __attribute__((always_inline)) Float7 operator*(const Float7 &other) const {
        Float7 ret = Float7(value6&other.value, value6&other.value1, value6&other.value2, value6&other.value3, value6&other.value4, value6&other.value5, value6&other.value6);
        ret.selfAdd(value5&other.value1, value5&other.value2, value5&other.value3, value5&other.value4, value5&other.value5, value5&other.value6);
        ret.selfAdd(value4&other.value2, value4&other.value3, value4&other.value4, value4&other.value5, value4&other.value6);
        ret.selfAdd(value3&other.value3, value3&other.value4, value3&other.value5, value3&other.value6);
        ret.selfAdd(value2&other.value4, value2&other.value5, value2&other.value6);
        ret.selfAdd(value1&other.value5, value1&other.value6);
        ret.selfAdd(value&other.value6);

        return ret;
    }

    inline __attribute__((always_inline)) Float7 addWithoutCarry(const Float7 &other) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        VECTOR carry3 = (value3 & other.value3) | (carry2 & (value3 ^ other.value3));
        VECTOR carry4 = (value4 & other.value4) | (carry3 & (value4 ^ other.value4));
        VECTOR carry5 = (value5 & other.value5) | (carry4 & (value5 ^ other.value5));
        return Float7(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1,
                    other.value3^value3^carry2,
                    other.value4^value4^carry3,
                    other.value5^value5^carry4,
                    other.value6^value6^carry5
                    );
    }

    inline __attribute__((always_inline)) Float7 addWithCarry(const Float7 &other, VECTOR &lastcarry) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        VECTOR carry3 = (value3 & other.value3) | (carry2 & (value3 ^ other.value3));
        VECTOR carry4 = (value4 & other.value4) | (carry3 & (value4 ^ other.value4));
        VECTOR carry5 = (value5 & other.value5) | (carry4 & (value5 ^ other.value5));
        lastcarry = (value6 & other.value6) | (carry5 & (value6 ^ other.value6));
        return Float7(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1,
                    other.value3^value3^carry2,
                    other.value4^value4^carry3,
                    other.value5^value5^carry4,
                    other.value6^value6^carry5
                    );
    }

    inline __attribute__((always_inline)) Float7 twosComplement(const VECTOR &mask) const {
        VECTOR notmask = ~mask;
        return Float7((mask&~value) | (notmask&value), 
                      (mask&~value1) | (notmask&value1),
                      (mask&~value2) | (notmask&value2),
                      (mask&~value3) | (notmask&value3),
                      (mask&~value4) | (notmask&value4),
                      (mask&~value5) | (notmask&value5),
                      (mask&~value6) | (notmask&value6)
        ).selfAdd(mask);
    }

    inline __attribute__((always_inline)) Float7 twosComplement() const {
        return Float7(~value, ~value1, ~value2, ~value3, ~value4, ~value5, ~value6)
            .addWithoutCarry(Float7(~(VECTOR)0,0,0,0,0,0,0));
    }

    inline __attribute__((always_inline)) Float7 twosComplementWithCarry(VECTOR& carry) const {
        return Float7(~value, ~value1, ~value2, ~value3, ~value4, ~value5, ~value6)
            .addWithCarry(Float7(~(VECTOR)0,0,0,0,0,0,0), carry);
    }
    
    inline __attribute__((always_inline)) Float7 twosComplementWithCarry(const VECTOR &mask, VECTOR& carry) const {
        VECTOR notmask = ~mask;
        return Float7((mask&~value) | (notmask&value), 
                      (mask&~value1) | (notmask&value1),
                      (mask&~value2) | (notmask&value2),
                      (mask&~value3) | (notmask&value3),
                      (mask&~value4) | (notmask&value4),
                      (mask&~value5) | (notmask&value5),
                      (mask&~value6) | (notmask&value6)
        ).addWithCarry(Float7(mask,0,0,0,0,0,0), carry);
    }

    inline __attribute__((always_inline)) Float7 operator+(const Float7 &other) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        VECTOR carry3 = (value3 & other.value3) | (carry2 & (value3 ^ other.value3));
        VECTOR carry4 = (value4 & other.value4) | (carry3 & (value4 ^ other.value4));
        VECTOR carry5 = (value5 & other.value5) | (carry4 & (value5 ^ other.value5));

        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = (value6 & other.value6) | (carry5 & (value6 ^ other.value6));
        if(ANY(lastcarry)) 
            throw std::logic_error("arithmetic overflow");
        #endif

        return Float7(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1,
                    other.value3^value3^carry2,
                    other.value4^value4^carry3,
                    other.value5^value5^carry4,
                    other.value6^value6^carry5
                    );
    }

    inline __attribute__((always_inline)) const Float7& selfAdd(const VECTOR &other, const VECTOR &other1, const VECTOR &other2, const VECTOR &other3, const VECTOR &other4, const VECTOR &other5, const VECTOR &other6) {
        VECTOR carry = other&value;
        VECTOR carry1 = (value1 & other1) | (carry & (value1 ^ other1));
        VECTOR carry2 = (value2 & other2) | (carry1 & (value2 ^ other2));
        VECTOR carry3 = (value3 & other3) | (carry2 & (value3 ^ other3));
        VECTOR carry4 = (value4 & other4) | (carry3 & (value4 ^ other4));
        VECTOR carry5 = (value5 & other5) | (carry4 & (value5 ^ other5));

        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = (value6 & other6) | (carry5 & (value6 ^ other.6));
        if(ANY(lastcarry)) 
            throw std::logic_error("arithmetic overflow");
        #endif

        value = other^value;
        value1 = other1^value1^carry;
        value2 = other2^value2^carry1;
        value3 = other3^value3^carry2;
        value4 = other4^value4^carry3;
        value5 = other5^value5^carry4;
        value6 = other6^value6^carry5;
        return *this;
    }
    

    inline __attribute__((always_inline)) const Float7& selfAdd(const VECTOR &other, const VECTOR &other1, const VECTOR &other2, const VECTOR &other3, const VECTOR &other4, const VECTOR &other5) {
        VECTOR carry = other&value;
        VECTOR carry1 = (value1 & other1) | (carry & (value1 ^ other1));
        VECTOR carry2 = (value2 & other2) | (carry1 & (value2 ^ other2));
        VECTOR carry3 = (value3 & other3) | (carry2 & (value3 ^ other3));
        VECTOR carry4 = (value4 & other4) | (carry3 & (value4 ^ other4));
        VECTOR carry5 = (value5 & other5) | (carry4 & (value5 ^ other5));

        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = (carry5 & value6);
        if(ANY(lastcarry)) 
            throw std::logic_error("arithmetic overflow");
        #endif

        value = other^value;
        value1 = other1^value1^carry;
        value2 = other2^value2^carry1;
        value3 = other3^value3^carry2;
        value4 = other4^value4^carry3;
        value5 = other5^value5^carry4;
        value6 = value6^carry5;
        return *this;
    }

    
    inline __attribute__((always_inline)) const Float7& selfAdd(const VECTOR &other, const VECTOR &other1, const VECTOR &other2, const VECTOR &other3, const VECTOR &other4) {
        VECTOR carry = other&value;
        VECTOR carry1 = (value1 & other1) | (carry & (value1 ^ other1));
        VECTOR carry2 = (value2 & other2) | (carry1 & (value2 ^ other2));
        VECTOR carry3 = (value3 & other3) | (carry2 & (value3 ^ other3));
        VECTOR carry4 = (value4 & other4) | (carry3 & (value4 ^ other4));
        VECTOR carry5 = carry4 & value5;

        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = (carry5 & value6);
        if(ANY(lastcarry)) 
            throw std::logic_error("arithmetic overflow");
        #endif

        value = other^value;
        value1 = other1^value1^carry;
        value2 = other2^value2^carry1;
        value3 = other3^value3^carry2;
        value4 = other4^value4^carry3;
        value5 = value5^carry4;
        value6 = value6^carry5;
        return *this;
    }

    inline __attribute__((always_inline)) const Float7& selfAdd(const VECTOR &other, const VECTOR &other1, const VECTOR &other2, const VECTOR &other3) {
        VECTOR carry = other&value;
        VECTOR carry1 = (value1 & other1) | (carry & (value1 ^ other1));
        VECTOR carry2 = (value2 & other2) | (carry1 & (value2 ^ other2));
        VECTOR carry3 = (value3 & other3) | (carry2 & (value3 ^ other3));
        VECTOR carry4 = carry3 & value4;
        VECTOR carry5 = carry4 & value5;

        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = (carry5 & value6);
        if(ANY(lastcarry)) 
            throw std::logic_error("arithmetic overflow");
        #endif

        value = other^value;
        value1 = other1^value1^carry;
        value2 = other2^value2^carry1;
        value3 = other3^value3^carry2;
        value4 = value4^carry3;
        value5 = value5^carry4;
        value6 = value6^carry5;
        return *this;
    }

    inline __attribute__((always_inline)) const Float7& selfAdd(const VECTOR &other, const VECTOR &other1, const VECTOR &other2) {
        VECTOR carry = other&value;
        VECTOR carry1 = (value1 & other1) | (carry & (value1 ^ other1));
        VECTOR carry2 = (value2 & other2) | (carry1 & (value2 ^ other2));
        VECTOR carry3 = carry2 & value3;
        VECTOR carry4 = carry3 & value4;
        VECTOR carry5 = carry4 & value5;

        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = (carry5 & value6);
        if(ANY(lastcarry)) 
            throw std::logic_error("arithmetic overflow");
        #endif

        value = other^value;
        value1 = other1^value1^carry;
        value2 = other2^value2^carry1;
        value3 = value3^carry2;
        value4 = value4^carry3;
        value5 = value5^carry4;
        value6 = value6^carry5;
        return *this;
    }

    inline __attribute__((always_inline)) const Float7& selfAdd(const VECTOR &other, const VECTOR &other1) {
        VECTOR carry = other&value;
        VECTOR carry1 = (value1 & other1) | (carry & (value1 ^ other1));
        VECTOR carry2 = carry1 & value2;
        VECTOR carry3 = carry2 & value3;
        VECTOR carry4 = carry3 & value4;
        VECTOR carry5 = carry4 & value5;

        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = (carry5 & value6);
        if(ANY(lastcarry)) 
            throw std::logic_error("arithmetic overflow");
        #endif

        value = other^value;
        value1 = other1^value1^carry;
        value2 = value2^carry1;
        value3 = value3^carry2;
        value4 = value4^carry3;
        value5 = value5^carry4;
        value6 = value6^carry5;
        return *this;
    }

    inline __attribute__((always_inline)) const Float7& selfAdd(const VECTOR &other) {
        VECTOR carry = other&value;
        VECTOR carry1 = carry & value1;
        VECTOR carry2 = carry1 & value2;
        VECTOR carry3 = carry2 & value3;
        VECTOR carry4 = carry3 & value4;
        VECTOR carry5 = carry4 & value5;

        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = (carry5 & value6);
        if(ANY(lastcarry)) 
            throw std::logic_error("arithmetic overflow");
        #endif

        value = other^value;
        value1 = value1^carry;
        value2 = value2^carry1;
        value3 = value3^carry2;
        value4 = value4^carry3;
        value5 = value5^carry4;
        value6 = value6^carry5;
        return *this;
    }

    inline __attribute__((always_inline)) const Float7& operator+=(const Float7 &other) {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        VECTOR carry3 = (value3 & other.value3) | (carry2 & (value3 ^ other.value3));
        VECTOR carry4 = (value4 & other.value4) | (carry3 & (value4 ^ other.value4));
        VECTOR carry5 = (value5 & other.value5) | (carry4 & (value5 ^ other.value5));

        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = (value6 & other.value6) | (carry5 & (value6 ^ other.value6));
        if(ANY(lastcarry)) 
            throw std::logic_error("arithmetic overflow");
        #endif

        value = other.value^value;
        value1 = other.value1^value1^carry;
        value2 = other.value2^value2^carry1;
        value3 = other.value3^value3^carry2;
        value4 = other.value4^value4^carry3;
        value5 = other.value5^value5^carry4;
        value6 = other.value6^value6^carry5;
        return *this;
    }
    
    inline __attribute__((always_inline)) const Float7& operator*=(const Float7 &other) {
        Float7 ret = Float7(value6&other.value, value6&other.value1, value6&other.value2, value6&other.value3, value6&other.value4, value6&other.value5, value6&other.value6);
        ret.selfAdd(value5&other.value1, value5&other.value2, value5&other.value3, value5&other.value4, value5&other.value5, value5&other.value6);
        ret.selfAdd(value4&other.value2, value4&other.value3, value4&other.value4, value4&other.value5, value4&other.value6);
        ret.selfAdd(value3&other.value3, value3&other.value4, value3&other.value5, value3&other.value6);
        ret.selfAdd(value2&other.value4, value2&other.value5, value2&other.value6);
        ret.selfAdd(value1&other.value5, value1&other.value6);
        ret.selfAdd(value&other.value6);
        value = ret.value;
        value1 = ret.value1;
        value2 = ret.value2;
        value3 = ret.value3;
        value4 = ret.value4;
        value5 = ret.value5;
        value6 = ret.value6;
        return *this;
    }

    inline __attribute__((always_inline)) static double sup() {
        return 0.9921875;
    }
    
    static double eps() {
        return 0.0078125;
    }

    inline __attribute__((always_inline)) static double inf() {
        return 0;
    }

    inline __attribute__((always_inline)) Float7 zerolike() const {
        return Float7();
    }

    inline __attribute__((always_inline)) Float7 zerolike(const VECTOR& mask) const {
        VECTOR notmask = ~mask;
        return Float7(value&notmask, value1&notmask, value2&notmask, value3&notmask, value4&notmask, value5&notmask, value6&notmask);
    }
};
}

#endif // Float7_H
