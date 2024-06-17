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

#ifndef FLOAT3_H
#define FLOAT3_H

#include <iostream>
#include <vector>
#include <bitset>
#include <cstdlib>
#include <random>
#include "../vecutils.h"

namespace tensorless {

class Float3 {
private:
    VECTOR value;
    VECTOR value1;
    VECTOR value2;
    explicit Float3(VECTOR v, VECTOR v1, VECTOR v2) : value(v), value1(v1), value2(v2) {}
public:
    static Float3 random() {return Float3(lrand(), lrand(), 0);}
    static Float3 broadcast(double val) {
        if(val<0 || val>2)
            throw std::logic_error("can only set values in range [0,2]");
        VECTOR value2 = 0;
        VECTOR value1 = 0;
        VECTOR value = 0;
        if(val>=1) {
            value2 = ~value2;
            val -= 1;
        }
        if(val>=0.5) {
            value1 = ~value1;
            val -= 0.5;
        }
        if(val>=0.25/2)
            value = ~value;
        return Float3(value, value1, value2);
    }

    static int num_params() {
        return 3;
    }

    static int num_bits() {
        return 3*VECTOR_SIZE;
    }

    Float3(const std::vector<double>& vec) : value(0), value1(0), value2(0) {
        for (int i = 0; i < vec.size(); ++i) 
            if (vec[i]) 
                set(i, vec[i]);
    }
    
    Float3(const Float3 &other) : value(other.value), value1(other.value1), value2(other.value2) {}
    
    Float3() : value(0), value1(0), value2(0) {}

    Float3& operator=(const Float3 &other) {
        if (this != &other) {
            value = other.value;
            value1 = other.value1;
            value2 = other.value2;
        }
        return *this;
    }
    
    explicit operator bool() const {
        return (bool)(value) || (bool)value1 || (bool)value2;
    }

    Float3 times2() {
        #ifdef DEBUG_OVERFLOWS
        if(value2)
            throw std::logic_error("arithmetic overflow");
        #endif
        return Float3(0, value, value1);
    }

    Float3 half() {
        return Float3(value1, value2, 0);
    }

    friend std::ostream& operator<<(std::ostream &os, const Float3 &si) {
        os << "[" << si.get(0);
        for(int i=1;i<si.size();i++)
            os << "," << si.get(i);
        os << "]";
        return os;
    }

    const Float3& print(const std::string& text="") const {
        std::cout << text << *this << "\n";
        return *this;
    }

    const int size() const {
        return sizeof(VECTOR)*8;
    }

    const bool isZeroAt(int i) {
        VECTOR a = value | value1 | value2;
        return (a >> i) & 1;
    }

    const double absmax() const {
        double ret = 0;
        VECTOR v1 = value1;
        VECTOR v = value;
        if(value2) {
            ret += 1;
            v1 &= value2;
            v &= value2;
        }
        if(v1) {
            ret += 0.5;
            v &= v1;
        }
        if(v)
            ret += 0.25;
        return ret;
    }

    const double sum() const {
        return bitcount(value)/4.0 + bitcount(value1)/2.0 + bitcount(value2);
    }

    const double sum(VECTOR mask) const {
        return bitcount(value&mask)/4.0 + bitcount(value1&mask)/2.0 + bitcount(value2&mask);
    }

    const double get(int i) {
        return ((value >> i) & 1)/4.0 + ((value1 >> i) & 1)/2.0 + ((value2 >> i) & 1);
    }

    const double get(int i) const {
        return ((value >> i) & 1)/4.0 + ((value1 >> i) & 1)/2.0 + ((value2 >> i) & 1);
    }

    const Float3& set(int i, double val) {
        if(size()<=i || i<0)
            throw std::logic_error("out of of range");
        if(val<0 || val>2)
            throw std::logic_error("can only set values in range [0,2]");
        if(val>=1) {
            value2 |= ONEHOT(i);
            val -= 1;
        }
        else
            value2 &= ~ONEHOT(i);
        if(val>=0.5) {
            value1 |= ONEHOT(i);
            val -= 0.5;
        }
        else
            value1 &= ~ONEHOT(i);
        if(val>=0.25/2)
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
    
    Float3& operator[](std::pair<int, double> p) {
        set(p.first, p.second);
        return *this;
    }

    int countNonZeros() { 
        VECTOR n = value | value1 | value2;
        return bitcount(n);
    }

    Float3 twosComplement(const VECTOR &mask) const {
        VECTOR notmask = ~mask;
        return Float3(mask&~value | (notmask&value), 
                      mask&~value1 | (notmask&value1),
                      mask&~value2 | (notmask&value2)
        ).addWithoutCarry(Float3(mask,0,0));
    }
    
    Float3 twosComplement() const {
        return Float3(~value, ~value1, ~value2).addWithoutCarry(Float3(~(VECTOR)0,0,0));
    }

    Float3 twosComplementWithCarry(const VECTOR &mask, VECTOR &carry) const {
        VECTOR notmask = ~mask;
        return Float3(mask&~value | (notmask&value), 
                      mask&~value1 | (notmask&value1),
                      mask&~value2 | (notmask&value2)
        ).addWithCarry(Float3(mask,0,0), carry);
    }
    
    Float3 twosComplement(VECTOR &carry) const {
        return Float3(~value, ~value1, ~value2).addWithCarry(Float3(~(VECTOR)0,0,0), carry);
    }
    
    Float3 operator~() const {
        return Float3(0, 0, ~(value | value1 | value2));
    }

    Float3 operator!=(const Float3 &other) const {
        return Float3(0, 0, (other.value^value) | (other.value1^value1) | (other.value2^value2));
    }

    Float3 operator*(const Float3 &other) const {
        Float3 ret = Float3();
        ret += Float3(value2&other.value, value2&other.value1, value2&other.value2);
        ret += Float3(value1&other.value1, value1&other.value2, 0);
        ret += Float3(value&other.value2, 0, 0);
        return ret;
    }

    Float3 addWithoutCarry(const Float3 &other) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        return Float3(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1
                    );
    }

    Float3 addWithCarry(const Float3 &other, VECTOR &lastcarry) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        lastcarry = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        return Float3(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1
                    );
    }

    Float3 operator+(const Float3 &other) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        
        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        if(lastcarry)
            throw std::logic_error("arithmetic overflow");
        #endif

        return Float3(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1
                    );
    }

    const Float3& operator+=(const Float3 &other) {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        
        #ifdef DEBUG_OVERFLOWS
        VECTOR lastcarry = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        if(lastcarry)
            throw std::logic_error("arithmetic overflow");
        #endif

        value = other.value^value;
        value1 = other.value1^value1^carry;
        value2 = other.value2^value2^carry1;
        return *this;
    }

    const Float3& operator*=(const Float3 &other) {
        Float3 ret = Float3();
        ret += Float3(value2&other.value, value2&other.value1, value2&other.value2);
        ret += Float3(value1&other.value1, value1&other.value2, 0);
        ret += Float3(value&other.value2, 0, 0);
        value = ret.value;
        value1 = ret.value1;
        value2 = ret.value2;
        return *this;
    }

    static double sup() {
        return 1.75;
    }

    static double inf() {
        return 0;
    }
};

}

#endif // FLOAT3_H
