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

#ifndef Float8_H
#define Float8_H

#include <iostream>
#include <vector>
#include <bitset>
#include <cstdlib>
#include <random>
#include "../vecutils.h"

namespace tensorless {

class Float8 {
private:
    VECTOR value;
    VECTOR value1;
    VECTOR value2;
    VECTOR value3;
    VECTOR value4;
    VECTOR value5;
    VECTOR value6;
    VECTOR value7;
    explicit Float8(VECTOR v, VECTOR v1, VECTOR v2, VECTOR v3, VECTOR v4, VECTOR v5, VECTOR v6, VECTOR v7) : 
        value(v), value1(v1), value2(v2), value3(v3), value4(v4), value5(v5), value6(v6), value7(v7) {}
public:
    static inline Float8 random() {return Float8(lrand(), lrand(), lrand(), lrand(), lrand(), lrand(), lrand(), 0);}
    
    static inline Float8 broadcast(double val) {
        if(val<0 || val>2)
            throw std::logic_error("can only set values in range [0,2]");
        VECTOR value7 = 0;
        VECTOR value6 = 0;
        VECTOR value5 = 0;
        VECTOR value4 = 0;
        VECTOR value3 = 0;
        VECTOR value2 = 0;
        VECTOR value1 = 0;
        VECTOR value = 0;
        if(val>=1) {
            value7 = ~value7;
            val -= 1;
        }
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
        return Float8(value, value1, value2, value3, value4, value5, value6, value7);
    }


    inline Float8(const std::vector<double>& vec) : value(0), value1(0), value2(0), value3(0), value4(0), value5(0), value6(0), value7(0) {
        for (int i = 0; i < vec.size(); ++i) 
            if (vec[i]) 
                set(i, vec[i]);
    }
    
    inline Float8(const Float8 &other) : 
        value(other.value), value1(other.value1), value2(other.value2), value3(other.value3), value4(other.value4), value5(other.value5), value6(other.value6), value7(other.value7) {}
    
    inline Float8() : value(0), value1(0), value2(0), value3(0), value4(0), value5(0), value6(0), value7(0) {}

    inline Float8& operator=(const Float8 &other) {
        if (this != &other) {
            value = other.value;
            value1 = other.value1;
            value2 = other.value2;
            value3 = other.value3;
            value4 = other.value4;
            value5 = other.value5;
            value6 = other.value6;
            value7 = other.value7;
        }
        return *this;
    }

    inline Float8 times2() const {
        #ifdef DEBUG_OVERFLOWS
        if(ANY(value7))
            throw std::logic_error("arithmetic overflow");
        #endif
        return Float8(0, value, value1, value2, value3, value4, value5, value6);
    }

    inline Float8 times2(const VECTOR &mask) const {
        #ifdef DEBUG_OVERFLOWS
        if(ANY(value7 & mask))
            throw std::logic_error("arithmetic overflow");
        #endif
        VECTOR notmask = ~mask;
        return Float8(                   notmask & value, 
                      (mask & value)  | (notmask & value1),
                      (mask & value1) | (notmask & value2),
                      (mask & value2) | (notmask & value3),
                      (mask & value3) | (notmask & value4),
                      (mask & value4) | (notmask & value5),
                      (mask & value5) | (notmask & value6),
                      (mask & value6) | (notmask & value7)
                      );
    }

    inline Float8 half() const {
        return Float8(value1, value2, value3, value4, value5, value6, value7, 0);
    }

    inline Float8 half(const VECTOR &mask) const {
        VECTOR notmask = ~mask;
        return Float8(
                      (mask & value1) | (value & notmask), 
                      (mask & value2) | (value1 & notmask), 
                      (mask & value3) | (value2 & notmask), 
                      (mask & value4) | (value3 & notmask), 
                      (mask & value5) | (value4 & notmask), 
                      (mask & value6) | (value5 & notmask), 
                      (mask & value7) | (value6 & notmask), 
                                         value7 & notmask);
    }

    inline static int num_params() {
        return 8;
    }

    inline static int num_bits() {
        return 8*VECTOR_SIZE;
    }
    
    inline explicit operator bool() const {
        return ANY(value) || ANY(value1) || ANY(value2) || ANY(value3) 
            || ANY(value4) || ANY(value5) || ANY(value6) || ANY(value7);
    }

    inline friend std::ostream& operator<<(std::ostream &os, const Float8 &si) {
        os << "[" << si.get(0);
        for(int i=1;i<si.size();i++)
            os << "," << si.get(i);
        os << "]";
        return os;
    }

    inline const Float8& print(const std::string& text="") const {
        std::cout << text << *this << "\n";
        return *this;
    }

    inline const int size() const {
        return VECTOR_SIZE;
    }

    inline const bool isZeroAt(int i) {
        VECTOR a = value | value1 | value2 | value3 | value4 | value5 | value6 | value7;
        return GETAT(a, i);
    }
    
    inline const double absmax() const {
        int ret = 0;
        VECTOR mask = 0;
        if(ANY(value7)) {
            ret += 128;
            mask = value7;
        } 
        else {
            mask = ~mask;
        }
        VECTOR v6 = value6 & mask;
        if(ANY(v6)) {
            ret += 64;
            mask = v6;
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
        return ret/128.0;
    }

    inline const double sum() const {
        int ret = bitcount(value);
        ret += bitcount(value1)*2;
        ret += bitcount(value2)*4;
        ret += bitcount(value3)*8;
        ret += bitcount(value4)*16;
        ret += bitcount(value5)*32;
        ret += bitcount(value6)*64;
        ret += bitcount(value7)*128;
        return ret/128.0;
    }

    inline const double sum(const VECTOR &mask) const {
        int ret = bitcount(value & mask);
        ret += bitcount(value1 & mask)*2;
        ret += bitcount(value2 & mask)*4;
        ret += bitcount(value3 & mask)*8;
        ret += bitcount(value4 & mask)*16;
        ret += bitcount(value5 & mask)*32;
        ret += bitcount(value6 & mask)*64;
        ret += bitcount(value7 & mask)*128;
        return ret/128.0;
    }
    
    inline const double get(int i) const {
        int ret = GETAT(value, i);
        ret += GETAT(value1, i)*2;
        ret += GETAT(value2, i)*4;
        ret += GETAT(value3, i)*8;
        ret += GETAT(value4, i)*16;
        ret += GETAT(value5, i)*32;
        ret += GETAT(value6, i)*64;
        ret += GETAT(value7, i)*128;
        return ret/128.0;
    }

    inline const Float8& set(int i, double val) {
        if(size()<=i || i<0)
            throw std::logic_error("out of of range");
        if(val<0 || val>2)
            throw std::logic_error("can only set values in range [0,2]");
        if(val>=1) {
            // #pragma omp atomic
            value7 |= ONEHOT(i);
            val -= 1;
        }
        else {
            // #pragma omp atomic
            value7 &= ~ONEHOT(i);
        }
        if(val>=0.5) {
            // #pragma omp atomic
            value6 |= ONEHOT(i);
            val -= 0.5;
        }
        else{
            // #pragma omp atomic
            value6 &= ~ONEHOT(i);
        }
        if(val>=0.25){
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

    inline double operator[](int i) {
        return get(i);
    }

    inline double operator[](int i) const {
        return get(i);
    }
    
    inline Float8& operator[](std::pair<int, double> p) {
        set(p.first, p.second);
        return *this;
    }

    inline int countNonZeros() { 
        VECTOR n = value | value1 | value2 | value3 | value4 | value5 | value6 | value7;
        return bitcount(n);
    }

    inline Float8 operator~() const {
        return Float8(0, 0, 0, 0, 0, 0, 0, ~(value | value1 | value2 | value3 | value4 | value5 | value6 | value7));
    }

    inline Float8 operator!=(const Float8 &other) const {
        return Float8(0, 0, 0, 0, 0, 0, 0, 
            (other.value^value) | (other.value1^value1) | (other.value2^value2) | (other.value3^value3) | (other.value4^value4) | (other.value5^value5) | (other.value6^value6) | (other.value7^value7));
    }

    inline Float8 operator*(const Float8 &other) const {
        Float8 ret = Float8();
        ret += Float8(value7&other.value, value7&other.value1, value7&other.value2, value7&other.value3, value7&other.value4, value7&other.value5, value7&other.value6, value7&other.value7);
        ret += Float8(value6&other.value1, value6&other.value2, value6&other.value3, value6&other.value4, value6&other.value5, value6&other.value6, value6&other.value7, 0);
        ret += Float8(value5&other.value2, value5&other.value3, value5&other.value4, value5&other.value5, value5&other.value6, value5&other.value7, 0, 0);
        ret += Float8(value4&other.value3, value4&other.value4, value4&other.value5, value4&other.value6, value4&other.value7, 0, 0, 0);
        ret += Float8(value3&other.value4, value3&other.value5, value3&other.value6, value3&other.value7, 0, 0, 0, 0);
        ret += Float8(value2&other.value5, value2&other.value6, value2&other.value7, 0, 0, 0, 0, 0);
        ret += Float8(value1&other.value6, value1&other.value7, 0, 0, 0, 0, 0, 0);
        ret += Float8(value&other.value7, 0, 0, 0, 0, 0, 0, 0);

        return ret;
    }

    inline Float8 addWithoutCarry(const Float8 &other) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        VECTOR carry3 = (value3 & other.value3) | (carry2 & (value3 ^ other.value3));
        VECTOR carry4 = (value4 & other.value4) | (carry3 & (value4 ^ other.value4));
        VECTOR carry5 = (value5 & other.value5) | (carry4 & (value5 ^ other.value5));
        VECTOR carry6 = (value6 & other.value6) | (carry5 & (value6 ^ other.value6));
        return Float8(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1,
                    other.value3^value3^carry2,
                    other.value4^value4^carry3,
                    other.value5^value5^carry4,
                    other.value6^value6^carry5,
                    other.value7^value7^carry6
                    );
    }

    inline Float8 addWithCarry(const Float8 &other, VECTOR &lastcarry) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        VECTOR carry3 = (value3 & other.value3) | (carry2 & (value3 ^ other.value3));
        VECTOR carry4 = (value4 & other.value4) | (carry3 & (value4 ^ other.value4));
        VECTOR carry5 = (value5 & other.value5) | (carry4 & (value5 ^ other.value5));
        VECTOR carry6 = (value6 & other.value6) | (carry5 & (value6 ^ other.value6));
        lastcarry = (value7 & other.value7) | (carry6 & (value7 ^ other.value7));
        return Float8(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1,
                    other.value3^value3^carry2,
                    other.value4^value4^carry3,
                    other.value5^value5^carry4,
                    other.value6^value6^carry5,
                    other.value7^value7^carry6
                    );
    }

    inline Float8 twosComplement(const VECTOR &mask) const {
        VECTOR notmask = ~mask;
        return Float8((mask&~value) | (notmask&value), 
                      (mask&~value1) | (notmask&value1),
                      (mask&~value2) | (notmask&value2),
                      (mask&~value3) | (notmask&value3),
                      (mask&~value4) | (notmask&value4),
                      (mask&~value5) | (notmask&value5),
                      (mask&~value6) | (notmask&value6),
                      (mask&~value7) | (notmask&value7)
        ).addWithoutCarry(Float8(mask,0,0,0,0,0,0,0));
    }

    inline Float8 twosComplement() const {
        return Float8(~value, ~value1, ~value2, ~value3, ~value4, ~value5, ~value6, ~value7)
            .addWithoutCarry(Float8(~(VECTOR)0,0,0,0,0,0,0,0));
    }

    inline Float8 twosComplementWithCarry(VECTOR& carry) const {
        return Float8(~value, ~value1, ~value2, ~value3, ~value4, ~value5, ~value6, ~value7)
            .addWithCarry(Float8(~(VECTOR)0,0,0,0,0,0,0,0), carry);
    }
    
    inline Float8 twosComplementWithCarry(const VECTOR &mask, VECTOR& carry) const {
        VECTOR notmask = ~mask;
        return Float8((mask&~value) | (notmask&value), 
                      (mask&~value1) | (notmask&value1),
                      (mask&~value2) | (notmask&value2),
                      (mask&~value3) | (notmask&value3),
                      (mask&~value4) | (notmask&value4),
                      (mask&~value5) | (notmask&value5),
                      (mask&~value6) | (notmask&value6),
                      (mask&~value7) | (notmask&value7)
        ).addWithCarry(Float8(mask,0,0,0,0,0,0,0), carry);
    }

    inline Float8 operator+(const Float8 &other) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        VECTOR carry3 = (value3 & other.value3) | (carry2 & (value3 ^ other.value3));
        VECTOR carry4 = (value4 & other.value4) | (carry3 & (value4 ^ other.value4));
        VECTOR carry5 = (value5 & other.value5) | (carry4 & (value5 ^ other.value5));
        VECTOR carry6 = (value6 & other.value6) | (carry5 & (value6 ^ other.value6));

        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = (value7 & other.value7) | (carry6 & (value7 ^ other.value7));
        if(ANY(lastcarry)) 
            throw std::logic_error("arithmetic overflow");
        #endif

        return Float8(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1,
                    other.value3^value3^carry2,
                    other.value4^value4^carry3,
                    other.value5^value5^carry4,
                    other.value6^value6^carry5,
                    other.value7^value7^carry6
                    );
    }
    inline const Float8& operator+=(const Float8 &other) {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        VECTOR carry3 = (value3 & other.value3) | (carry2 & (value3 ^ other.value3));
        VECTOR carry4 = (value4 & other.value4) | (carry3 & (value4 ^ other.value4));
        VECTOR carry5 = (value5 & other.value5) | (carry4 & (value5 ^ other.value5));
        VECTOR carry6 = (value6 & other.value6) | (carry5 & (value6 ^ other.value6));

        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = (value7 & other.value7) | (carry6 & (value7 ^ other.value7));
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
        value7 = other.value7^value7^carry6;
        return *this;
    }
    
    inline const Float8& operator*=(const Float8 &other) {
        Float8 ret = Float8();
        ret += Float8(value7&other.value, value7&other.value1, value7&other.value2, value7&other.value3, value7&other.value4, value7&other.value5, value7&other.value6, value7&other.value7);
        ret += Float8(value6&other.value1, value6&other.value2, value6&other.value3, value6&other.value4, value6&other.value5, value6&other.value6, value6&other.value7, 0);
        ret += Float8(value5&other.value2, value5&other.value3, value5&other.value4, value5&other.value5, value5&other.value6, value5&other.value7, 0, 0);
        ret += Float8(value4&other.value3, value4&other.value4, value4&other.value5, value4&other.value6, value4&other.value7, 0, 0, 0);
        ret += Float8(value3&other.value4, value3&other.value5, value3&other.value6, value3&other.value7, 0, 0, 0, 0);
        ret += Float8(value2&other.value5, value2&other.value6, value2&other.value7, 0, 0, 0, 0, 0);
        ret += Float8(value1&other.value6, value1&other.value7, 0, 0, 0, 0, 0, 0);
        ret += Float8(value&other.value7, 0, 0, 0, 0, 0, 0, 0);
        value = ret.value;
        value1 = ret.value1;
        value2 = ret.value2;
        value3 = ret.value3;
        value4 = ret.value4;
        value5 = ret.value5;
        value6 = ret.value6;
        value7 = ret.value7;
        return *this;
    }

    inline static double sup() {
        return 1.9921875;
    }
    
    static double eps() {
        return 0.0078125;
    }

    inline static double inf() {
        return 0;
    }

    inline Float8 zerolike() const {
        return Float8();
    }

    inline Float8 zerolike(const VECTOR& mask) const {
        VECTOR notmask = ~mask;
        return Float8(value&notmask, value1&notmask, value2&notmask, value3&notmask, value4&notmask, value5&notmask, value6&notmask, value7&notmask);
    }
};
}

#endif // Float8_H
