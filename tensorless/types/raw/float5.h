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

#ifndef FLOAT5_H
#define FLOAT5_H

#include <iostream>
#include <vector>
#include <bitset>
#include <cstdlib>
#include <random>
#include "../vecutils.h"

namespace tensorless {

class Float5 {
private:
    VECTOR value;
    VECTOR value1;
    VECTOR value2;
    VECTOR value3;
    VECTOR value4;
    explicit Float5(VECTOR v, VECTOR v1, VECTOR v2, VECTOR v3, VECTOR v4) : value(v), value1(v1), value2(v2), value3(v3), value4(v4) {}
public:
    static Float5 random() {return Float5(lrand(), lrand(), lrand(), lrand(), lrand());}

    Float5(const std::vector<double>& vec) : value(0), value1(0), value2(0), value3(0) {
        for (int i = 0; i < vec.size(); ++i) 
            if (vec[i]) 
                set(i, vec[i]);
    }
    
    Float5(const Float5 &other) : value(other.value), value1(other.value1), value2(other.value2), value3(other.value3), value4(other.value4) {}
    
    Float5() : value(0), value1(0), value2(0), value3(0), value4(0) {}

    Float5& operator=(const Float5 &other) {
        if (this != &other) {
            value = other.value;
            value1 = other.value1;
            value2 = other.value2;
            value3 = other.value3;
            value4 = other.value4;
        }
        return *this;
    }
    
    explicit operator bool() const {
        return (bool)(value) || (bool)value1 || (bool)value2 || (bool)value3 || (bool)value4;
    }

    friend std::ostream& operator<<(std::ostream &os, const Float5 &si) {
        os << "[" << si.get(0);
        for(int i=1;i<si.size();i++)
            os << "," << si.get(i);
        os << "]";
        return os;
    }

    const Float5& print(const std::string& text="") const {
        std::cout << text << *this << "\n";
        return *this;
    }

    const int size() const {
        return sizeof(VECTOR)*8;
    }

    const bool isZeroAt(int i) {
        VECTOR a = value | value1 | value2 | value3 | value4;
        return (a >> i) & 1;
    }

    const double get(int i) {
        return ((value >> i) & 1)/16.0 + ((value1 >> i) & 1)/8.0 + ((value2 >> i) & 1)/4.0 + ((value3 >> i) & 1)/2.0 + ((value4 >> i) & 1);
    }

    const double get(int i) const {
        return ((value >> i) & 1)/16.0 + ((value1 >> i) & 1)/8.0 + ((value2 >> i) & 1)/4.0 + ((value3 >> i) & 1)/2.0 + ((value4 >> i) & 1);
    }

    const Float5& set(int i, double val) {
        if(size()<=i || i<0)
            throw std::logic_error("out of of range");
        if(val<0 || val>7)
            throw std::logic_error("can only set values in range [0,2]");
        if(val>=1) {
            value4 |= ONEHOT(i);
            val -= 1;
        }
        else
            value4 &= ~ONEHOT(i);
        if(val>=0.5) {
            value3 |= ONEHOT(i);
            val -= 0.5;
        }
        else
            value3 &= ~ONEHOT(i);
        if(val>=0.25){
            value2 |= ONEHOT(i);
            val -= 0.25;
        }
        else
            value2 &= ~ONEHOT(i);
        if(val>=0.125){
            value1 |= ONEHOT(i);
            val -= 0.125;
        }
        else
            value1 &= ~ONEHOT(i);
        if(val>=0.0625/2)
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
    
    Float5& operator[](std::pair<int, double> p) {
        set(p.first, p.second);
        return *this;
    }

    int countNonZeros() { // WARNING: this is not parallelized
        VECTOR n = value | value1 | value2 | value3 | value4;
        int count = 0;
        while (n) {
            if(n & 1)
                ++count;
            n >>= 1;
        }
        return count;
    }
    
    void debug() const {
        std::cout<<"\n"<<(std::bitset<8>)value;
        std::cout<<"\n"<<(std::bitset<8>)value1;
        std::cout<<"\n"<<(std::bitset<8>)value2;
        std::cout<<"\n"<<(std::bitset<8>)value3;
        std::cout<<"\n"<<(std::bitset<8>)value4;
        std::cout<<"\n";
    }

    Float5 operator~() const {
        return Float5(0, 0, 0, 0, ~(value | value1 | value2 | value3 | value4));
    }

    Float5 operator!=(const Float5 &other) const {
        return Float5(0, 0, 0, 0, (other.value^value) | (other.value1^value1) | (other.value2^value2) | (other.value3^value3) | (other.value4^value4));
    }

    Float5 operator*(const Float5 &other) const {
        Float5 ret = Float5();
        ret += Float5(value4&other.value, value4&other.value1, value4&other.value2, value4&other.value3, value4&other.value4);
        ret += Float5(value3&other.value1, value3&other.value2, value3&other.value3, value3&other.value4, 0);
        ret += Float5(value2&other.value2, value2&other.value3, value2&other.value4, 0, 0);
        ret += Float5(value1&other.value3, value1&other.value4, 0, 0, 0);
        ret += Float5(value&other.value4, 0, 0, 0, 0);
        return ret;
    }

    Float5 addWithCarry(const Float5 &other, VECTOR &lastcarry) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        VECTOR carry3 = (value3 & other.value3) | (carry2 & (value3 ^ other.value3));
        lastcarry = (value4 & other.value4) | (carry3 & (value4 ^ other.value4));
        return Float5(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1,
                    other.value3^value3^carry2,
                    other.value4^value4^carry3
                    );
    }

    Float5 twosComplement(VECTOR mask) const {
        return Float5((mask&~value) | (~mask&value), 
                      (mask&~value1) | (~mask&value1),
                      (mask&~value2) | (~mask&value2),
                      (mask&~value3) | (~mask&value3),
                      (mask&~value4) | (~mask&value4)
        ) + Float5(mask,0,0,0,0);
    }

    Float5 twosComplement() const {
        return Float5(~value, value1, value2, value3, value4) + Float5(value|~value,0,0,0,0);
    }

    Float5 operator+(const Float5 &other) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        VECTOR carry3 = (value3 & other.value3) | (carry2 & (value3 ^ other.value3));
        return Float5(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1,
                    other.value3^value3^carry2,
                    other.value4^value4^carry3
                    );
    }
    const Float5& operator+=(const Float5 &other) {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        VECTOR carry2 = (value2 & other.value2) | (carry1 & (value2 ^ other.value2));
        VECTOR carry3 = (value3 & other.value3) | (carry2 & (value3 ^ other.value3));
        value = other.value^value;
        value1 = other.value1^value1^carry;
        value2 = other.value2^value2^carry1;
        value3 = other.value3^value3^carry2;
        value4 = other.value4^value4^carry3;
        return *this;
    }
    const Float5& operator*=(const Float5 &other) {
        Float5 ret = Float5();
        ret += Float5(value4&other.value, value4&other.value1, value4&other.value2, value4&other.value3, value4&other.value4);
        ret += Float5(value3&other.value1, value3&other.value2, value3&other.value3, value3&other.value4, 0);
        ret += Float5(value2&other.value2, value2&other.value3, value2&other.value4, 0, 0);
        ret += Float5(value1&other.value3, value1&other.value4, 0, 0, 0);
        ret += Float5(value&other.value4, 0, 0, 0, 0);
        value = ret.value;
        value1 = ret.value1;
        value2 = ret.value2;
        value3 = ret.value3;
        value4 = ret.value4;
        return *this;
    }

    const double sup() {
        return 1.9375;
    }

    const double inf() {
        return 0;
    }
};
}

#endif // FLOAT5_H
