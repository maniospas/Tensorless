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

#ifndef FLOAT4_H
#define FLOAT4_H

#include <iostream>
#include <vector>
#include <bitset>
#include <cstdlib>
#include <random>
#include "../vecutils.h"


namespace tensorless {

class Float4 {
private:
    VECTOR value;
    VECTOR value1;
    VECTOR value2;
    VECTOR value3;
    explicit Float4(VECTOR v, VECTOR v1, VECTOR v2, VECTOR v3) : value(v), value1(v1), value2(v2), value3(v3) {}

public:
    static Float4 random() {return Float4(lrand(), lrand(), lrand(), 0);}
    static Float4 broadcast(double val) {
        if(val<0 || val>2)
            throw std::logic_error("can only set values in range [0,2]");
        VECTOR value3 = 0;
        VECTOR value2 = 0;
        VECTOR value1 = 0;
        VECTOR value = 0;
        if(val>=1) {
            value3 = ~value3;
            val -= 1;
        }
        if(val>=0.5) {
            value2 = ~value2;
            val -= 0.5;
        }
        if(val>=0.25) {
            value1 = ~value1;
            val -= 0.25;
        }
        if(val>=0.125/2)
            value = ~value;
        return Float4(value, value1, value2, value3);
    }
    static int num_params() {
        return 4;
    }
    
    static int num_bits() {
        return 4*VECTOR_SIZE;
    }

    Float4 times2() {
        #ifdef DEBUG_OVERFLOWS
        if(ANY(value3))
            throw std::logic_error("arithmetic overflow");
        #endif
        return Float4(0, value, value1, value2);
    }

    inline __attribute__((always_inline)) Float4 half() const {
        return Float4(value1, value2, value3, 0);
    }

    inline __attribute__((always_inline)) Float4 half(const VECTOR &mask) const {
        VECTOR notmask = ~mask;
        return Float4(
                      (mask & value1) | (value & notmask), 
                      (mask & value2) | (value1 & notmask), 
                      (mask & value3) | (value2 & notmask), 
                                         value3 & notmask);
    }
    
    inline __attribute__((always_inline)) Float4 quarter() const {
        return Float4(value2, value3, 0, 0);
    }

    inline __attribute__((always_inline)) Float4 quarter(const VECTOR &mask) const {
        VECTOR notmask = ~mask;
        return Float4(
                      (mask & value2) | (value & notmask), 
                      (mask & value3) | (value1 & notmask), 
                                         value2 & notmask, 
                                         value3 & notmask);
    }
    
    Float4 eighth() const {
        return Float4(value3, 0, 0, 0);
    }

    inline __attribute__((always_inline)) Float4 eighth(const VECTOR &mask) const {
        VECTOR notmask = ~mask;
        return Float4(
                      (mask & value3) | (value & notmask), 
                                         value1 & notmask, 
                                         value2 & notmask, 
                                         value3 & notmask);
    }

    Float4(const std::vector<double>& vec) : value(0), value1(0), value2(0), value3(0) {
        for (int i = 0; i < vec.size(); ++i) 
            if (vec[i]) 
                set(i, vec[i]);
    }
    
    Float4(const Float4 &other) : value(other.value), value1(other.value1), value2(other.value2), value3(other.value3) {}
    
    Float4() : value(0), value1(0), value2(0), value3(0) {}

    Float4& operator=(const Float4 &other) {
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

    friend std::ostream& operator<<(std::ostream &os, const Float4 &si) {
        os << "[" << si.get(0);
        for(int i=1;i<si.size();i++)
            os << "," << si.get(i);
        os << "]";
        return os;
    }

    const Float4& print(const std::string& text="") const {
        std::cout << text << *this << "\n";
        return *this;
    }

    const int size() const {
        return VECTOR_SIZE;
    }

    const bool isZeroAt(int i) {
        VECTOR a = value | value1 | value2 | value3;
        return GETAT(a, i);
    }

    const double absmax() const {
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
        return ret/8.0;
    }

    const double sum() const {
        int ret = bitcount(value);
        ret += bitcount(value1)*2;
        ret += bitcount(value2)*4;
        ret += bitcount(value3)*8;
        return ret/8.0;
    }

    const double sum(const VECTOR &mask) const {
        int ret = bitcount(value&mask);
        ret += bitcount(value1&mask)*2;
        ret += bitcount(value2&mask)*4;
        ret += bitcount(value3&mask)*8;
        return ret/8.0;
    }

    const double get(int i) const {
        int ret = GETAT(value, i);
        ret += GETAT(value1, i)*2;
        ret += GETAT(value2, i)*4;
        ret += GETAT(value3, i)*8;
        return ret/8.0;
    }

    const Float4& set(int i, double val) {
        #ifdef DEBUG_SET
        if(size()<=i || i<0)
            throw std::logic_error("out of of range");
        if(val<0 || val>2)
            throw std::logic_error("can only set values in range [0,2]");
        #endif
        if(val>=1) {
            value3 |= ONEHOT(i);
            val -= 1;
        }
        else
            value3 &= ~ONEHOT(i);
        if(val>=0.5) {
            value2 |= ONEHOT(i);
            val -= 0.5;
        }
        else
            value2 &= ~ONEHOT(i);
        if(val>=0.25){
            value1 |= ONEHOT(i);
            val -= 0.25;
        }
        else
            value1 &= ~ONEHOT(i);
        if(val>=0.125/2)
            value |= ONEHOT(i);
        else
            value &= ~ONEHOT(i);
        return *this;
    }

    double operator[](int i) {
        return get(i);
    }

    double operator[](int i) const {
        return get(i);
    }
    
    Float4& operator[](std::pair<int, double> p) {
        set(p.first, p.second);
        return *this;
    }

    inline __attribute__((always_inline)) int countNonZeros() { 
        VECTOR n = value | value1 | value2 | value3;
        return bitcount(n);
    }
    
    inline __attribute__((always_inline)) Float4 twosComplement(const VECTOR &mask) const {
        VECTOR notmask = ~mask;
        return Float4(mask&~value | (notmask&value), 
                      mask&~value1 | (notmask&value1),
                      mask&~value2 | (notmask&value2),
                      mask&~value3 | (notmask&value3)
        ).selfAdd(mask);
    }

    inline __attribute__((always_inline)) Float4 twosComplement() const {
        return Float4(~value, ~value1, ~value2, ~value3).selfAdd(~(VECTOR)0);
    }
    
    inline __attribute__((always_inline)) Float4 twosComplementWithCarry(const VECTOR &mask, VECTOR &carry) const {
        VECTOR notmask = ~mask;
        return Float4(mask&~value | (notmask&value), 
                      mask&~value1 | (notmask&value1),
                      mask&~value2 | (notmask&value2),
                      mask&~value3 | (notmask&value3)
        ).addWithCarry(Float4(mask,0,0,0), carry);
    }

    inline __attribute__((always_inline)) Float4 twosComplementWithCarry(VECTOR &carry) const {
        return Float4(~value, ~value1, ~value2, ~value3).addWithCarry(Float4(~(VECTOR)0,0,0,0), carry);
    }

    inline __attribute__((always_inline)) Float4 operator~() const {
        return Float4(0, 0, 0, ~(value | value1 | value2 | value3));
    }

    inline __attribute__((always_inline)) Float4 operator!=(const Float4 &other) const {
        return Float4(0, 0, 0, (other.value^value) | (other.value1^value1) | (other.value2^value2) | (other.value3^value3));
    }
    

    inline __attribute__((always_inline)) Float4 operator*(const Float4 &other) const {
        Float4 ret = Float4(value3&other.value, value3&other.value1, value3&other.value2, value3&other.value3);
        ret.selfAdd(value2&other.value1, value2&other.value2, value2&other.value3);
        ret.selfAdd(value1&other.value2, value1&other.value3);
        ret.selfAdd(value&other.value3);
        return ret;
    }

    
    inline __attribute__((always_inline)) const Float4& selfAdd(const VECTOR &other, const VECTOR &other1, const VECTOR &other2, const VECTOR &other3) {
        VECTOR carry = other&value;
        VECTOR carry1 = (value1 & other1) | (carry & (value1 ^ other1));
        VECTOR carry2 = (value2 & other2) | (carry1 & (value2 ^ other2));
        
        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = (value3 & other3) | (carry2 & (value3 ^ other3));
        if(ANY(lastcarry))
            throw std::logic_error("arithmetic overflow");
        #endif

        value = other^value;
        value1 = other1^value1^carry;
        value2 = other2^value2^carry1;
        value3 = other3^value3^carry2;
        return *this;
    }
    
    inline __attribute__((always_inline)) const Float4& selfAdd(const VECTOR &other, const VECTOR &other1, const VECTOR &other2) {
        VECTOR carry = other&value;
        VECTOR carry1 = (value1 & other1) | (carry & (value1 ^ other1));
        VECTOR carry2 = (value2 & other2) | (carry1 & (value2 ^ other2));
        
        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = carry2 & value3;
        if(ANY(lastcarry))
            throw std::logic_error("arithmetic overflow");
        #endif

        value = other^value;
        value1 = other1^value1^carry;
        value2 = other2^value2^carry1;
        value3 = value3^carry2;
        return *this;
    }

    inline __attribute__((always_inline)) const Float4& selfAdd(const VECTOR &other, const VECTOR &other1) {
        VECTOR carry = other&value;
        VECTOR carry1 = (value1 & other1) | (carry & (value1 ^ other1));
        VECTOR carry2 = carry1 & value2;
        
        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = carry2 & value3;
        if(ANY(lastcarry))
            throw std::logic_error("arithmetic overflow");
        #endif

        value = other^value;
        value1 = other1^value1^carry;
        value2 = value2^carry1;
        value3 = value3^carry2;
        return *this;
    }
    
    inline __attribute__((always_inline)) const Float4& selfAdd(const VECTOR &other) {
        VECTOR carry = other&value;
        VECTOR carry1 = carry & value1;
        VECTOR carry2 = carry1 & value2;
        
        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = carry2 & value3;
        if(ANY(lastcarry))
            throw std::logic_error("arithmetic overflow");
        #endif

        value = other^value;
        value1 = value1^carry;
        value2 = value2^carry1;
        value3 = value3^carry2;
        return *this;
    }

    inline __attribute__((always_inline)) Float4 addWithoutCarry(const Float4 &other) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        return Float4(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1,
                    other.value3^value3^carry2
                    );
    }

    inline __attribute__((always_inline)) Float4 addWithCarry(const Float4 &other, VECTOR &lastcarry) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        lastcarry = (value3 & other.value3) | (carry2 & (value3 ^ other.value3));
        return Float4(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1,
                    other.value3^value3^carry2
                    );
    }

    Float4 operator+(const Float4 &other) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));

        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = (value3 & other.value3) | (carry2 & (value3 ^ other.value3));
        if(ANY(lastcarry)) 
            throw std::logic_error("arithmetic overflow");
        #endif

        return Float4(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1,
                    other.value3^value3^carry2
                    );
    }
    const Float4& operator+=(const Float4 &other) {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        
        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = (value3 & other.value3) | (carry2 & (value3 ^ other.value3));
        if(ANY(lastcarry))
            throw std::logic_error("arithmetic overflow");
        #endif

        value = other.value^value;
        value1 = other.value1^value1^carry;
        value2 = other.value2^value2^carry1;
        value3 = other.value3^value3^carry2;
        return *this;
    }
    
    const Float4& operator*=(const Float4 &other) {
        Float4 ret = Float4(value3&other.value, value3&other.value1, value3&other.value2, value3&other.value3);
        ret.selfAdd(value2&other.value1, value2&other.value2, value2&other.value3);
        ret.selfAdd(value1&other.value2, value1&other.value3);
        ret.selfAdd(value&other.value3);
        value = ret.value;
        value1 = ret.value1;
        value2 = ret.value2;
        value3 = ret.value3;
        return *this;
    }

    static double sup() {
        return 1.875;
    }

    static double eps() {
        return 0.125;
    }

    static double inf() {
        return 0;
    }

    Float4 zerolike() const {
        return Float4();
    }

    Float4 zerolike(const VECTOR& mask) const {
        VECTOR notmask = ~mask;
        return Float4(value&notmask, value1&notmask, value2&notmask, value3&notmask);
    }
};

}

#endif // FLOAT4_H
