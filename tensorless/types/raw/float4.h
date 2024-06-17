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
        if(value3)
            throw std::logic_error("arithmetic overflow");
        #endif
        return Float4(0, value, value1, value2);
    }

    Float4 half() {
        return Float4(value1, value2, value3, 0);
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
        return (bool)(value) || (bool)value1 || (bool)value2 || (bool)value3;
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
        return (a >> i) & 1;
    }

    const double absmax() const {
        double ret = 0;
        VECTOR v2 = value2;
        VECTOR v1 = value1;
        VECTOR v = value;
        if(value3) {
            ret += 1;
            v2 &= value3;
            v1 &= value3;
            v &= value3;
        }
        if(v2) {
            ret += 0.5;
            v1 &= v2;
            v &= v2;
        }
        if(v1) {
            ret += 0.25;
            v &= v1;
        }
        if(v)
            ret += 0.125;
        return ret;
    }

    const double sum() const {
        return bitcount(value)/8.0 + bitcount(value1)/4.0 + bitcount(value2)/2.0 + bitcount(value3);
    }

    const double sum(VECTOR mask) const {
        return bitcount(value&mask)/8.0 + bitcount(value1&mask)/4.0 + bitcount(value2&mask)/2.0 + bitcount(value3&mask);
    }

    const double get(int i) const {
        return ((value >> i) & 1)/8.0 + ((value1 >> i) & 1)/4.0 + ((value2 >> i) & 1)/2.0 + ((value3 >> i) & 1);
    }

    const Float4& set(int i, double val) {
        if(size()<=i || i<0)
            throw std::logic_error("out of of range");
        if(val<0 || val>2)
            throw std::logic_error("can only set values in range [0,2]");
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

    int countNonZeros() { 
        VECTOR n = value | value1 | value2 | value3;
        return bitcount(n);
    }
    
    Float4 twosComplement(const VECTOR &mask) const {
        VECTOR notmask = ~mask;
        return Float4(mask&~value | (notmask&value), 
                      mask&~value1 | (notmask&value1),
                      mask&~value2 | (notmask&value2),
                      mask&~value3 | (notmask&value3)
        ).addWithoutCarry(Float4(mask,0,0,0));
    }

    Float4 twosComplement() const {
        return Float4(~value, ~value1, ~value2, ~value3).addWithoutCarry(Float4(~(VECTOR)0,0,0,0));
    }
    
    Float4 twosComplementWithCarry(const VECTOR &mask, VECTOR &carry) const {
        VECTOR notmask = ~mask;
        return Float4(mask&~value | (notmask&value), 
                      mask&~value1 | (notmask&value1),
                      mask&~value2 | (notmask&value2),
                      mask&~value3 | (notmask&value3)
        ).addWithCarry(Float4(mask,0,0,0), carry);
    }

    Float4 twosComplementWithCarry(VECTOR &carry) const {
        return Float4(~value, ~value1, ~value2, ~value3).addWithCarry(Float4(~(VECTOR)0,0,0,0), carry);
    }

    Float4 operator~() const {
        return Float4(0, 0, 0, ~(value | value1 | value2 | value3));
    }

    Float4 operator!=(const Float4 &other) const {
        return Float4(0, 0, 0, (other.value^value) | (other.value1^value1) | (other.value2^value2) | (other.value3^value3));
    }

    Float4 operator*(const Float4 &other) const {
        Float4 ret = Float4();
        ret += Float4(value3&other.value, value3&other.value1, value3&other.value2, value3&other.value3);
        ret += Float4(value2&other.value1, value2&other.value2, value2&other.value3, 0);
        ret += Float4(value1&other.value2, value1&other.value3, 0, 0);
        ret += Float4(value&other.value3, 0, 0, 0);
        return ret;
    }

    Float4 addWithoutCarry(const Float4 &other) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        return Float4(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1,
                    other.value3^value3^carry2
                    );
    }

    Float4 addWithCarry(const Float4 &other, VECTOR &lastcarry) const {
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
        if(lastcarry)
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
        if(lastcarry)
            throw std::logic_error("arithmetic overflow");
        #endif

        value = other.value^value;
        value1 = other.value1^value1^carry;
        value2 = other.value2^value2^carry1;
        value3 = other.value3^value3^carry2;
        return *this;
    }
    const Float4& operator*=(const Float4 &other) {
        Float4 ret = Float4();
        ret += Float4(value3&other.value, value3&other.value1, value3&other.value2, value3&other.value3);
        ret += Float4(value2&other.value1, value2&other.value2, value2&other.value3, 0);
        ret += Float4(value1&other.value2, value1&other.value3, 0, 0);
        ret += Float4(value&other.value3, 0, 0, 0);
        value = ret.value;
        value1 = ret.value1;
        value2 = ret.value2;
        value3 = ret.value3;
        return *this;
    }

    static double sup() {
        return 1.875;
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
