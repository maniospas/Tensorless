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

#ifndef Int3_H
#define Int3_H

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
    static Int3 random() {return Int3(lrand(), lrand(), lrand());}
    static Int3 broadcast(double value) {
        int val = (int)value;
        return Int3(val&1?~(VECTOR)0:0, val&2?~(VECTOR)0:0, val&4?~(VECTOR)0:0);
    }

    Int3(const std::vector<int>& vec) : value(0), value1(0), value2(0) {
        for (int i = 0; i < vec.size(); ++i) 
            if (vec[i]) 
                set(i, vec[i]);
    }
    
    Int3(const Int3 &other) : value(other.value), value1(other.value1), value2(other.value2) {}
    
    Int3() : value(0) {}

    const int size() const {
        return sizeof(VECTOR)*8;
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
        return (bool)value || (bool)value1 || (bool)value2;
    }

    friend std::ostream& operator<<(std::ostream &os, const Int3 &si) {
        os << "[" << si.get(0);
        for(int i=1;i<si.size();i++)
            os << "," << si.get(i);
        os << "]";
        return os;
    }

    const Int3& print(const std::string& text="") const {
        std::cout << text << *this << "\n";
        return *this;
    }

    const bool isZeroAt(int i) {
        VECTOR a = value | value1 | value2;
        return (a >> i) & 1;
    }

    const int sum() {
        return bitcount(value) + bitcount(value1)*2 + bitcount(value2)*4;
    }

    const int sum(VECTOR mask) {
        return bitcount(value&mask) + bitcount(value1&mask)*2 + bitcount(value2&mask)*4;
    }

    const int get(int i) {
        return ((value >> i) & 1) + ((value1 >> i) & 1)*2 + ((value2 >> i) & 1)*4;
    }

    const int get(int i) const {
        return ((value >> i) & 1) + ((value1 >> i) & 1)*2 + ((value2 >> i) & 1)*4;
    }

    const Int3& set(int i, int val) {
        if(size()<=i || i<0)
            throw std::logic_error("out of of range");
        if(val<0 || val>7)
            throw std::logic_error("can only set values in range [0,7]");
        if(val&1) {
            #pragma omp atomic
            value |= ONEHOT(i);
        }
        else {
            #pragma omp atomic
            value &= ~ONEHOT(i);
        }
        if(val&2) {
            #pragma omp atomic
            value1 |= ONEHOT(i);
        }
        else {
            #pragma omp atomic
            value1 &= ~ONEHOT(i);
        }
        if(val&4) {
            #pragma omp atomic
            value2 |= ONEHOT(i);
        }
        else {
            #pragma omp atomic
            value2 &= ~ONEHOT(i);
        }
        return *this;
    }

    int operator[](int i) {
        return get(i);
    }

    int operator[](int i) const {
        return get(i);
    }
    
    Int3& operator[](std::pair<int, int> p) {
        set(p.first, p.second);
        return *this;
    }

    int countNonZeros() { 
        VECTOR n = value | value1 | value2;
        return bitcount(n);
    }
    
    Int3 operator~() const {
        return Int3(~(value | value1 | value2), 0, 0);
    }
    
    Int3 operator!=(const Int3 &other) const {
        return Int3((other.value^value) | (other.value1^value1) | (other.value2^value2), 0, 0);
    }

    Int3 operator*(const Int3 &other) const {
        return Int3(other.value&value, 
                    (other.value1&value) | (other.value&value1),
                    (other.value2&value) | (other.value&value2) | (other.value1&value1));
    }
 
    Int3 twosComplement(VECTOR mask) const {
        VECTOR notmask = ~mask;
        return Int3(mask&~value | (notmask&value), 
                      mask&~value1 | (notmask&value1),
                      mask&~value2 | (notmask&value2)
        ).addWithoutCarry(Int3(mask,0,0));
    }
    
    Int3 twosComplement() const {
        return Int3(~value, ~value1, ~value2).addWithoutCarry(Int3(~(VECTOR)0,0,0));
    }

    Int3 addWithoutCarry(const Int3 &other) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        return Int3(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1
                    );
    }

    Int3 addWithCarry(const Int3 &other, VECTOR &lastcarry) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        lastcarry = (value2 & other.value2) | (carry & (value2 ^ other.value2));
        return Int3(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1
                    );
    }

    Int3 operator+(const Int3 &other) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        return Int3(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1
                    );
    }

    const Int3& operator+=(const Int3 &other) {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        value = other.value^value;
        value1 = other.value1^value1^carry;
        value2 = other.value2^value2^carry1;
        return *this;
    }

    const Int3& operator*=(const Int3 &other) {
        value2 = (other.value2&value) | (other.value&value2) | (other.value1&value1);
        value1 = (other.value1&value) | (other.value&value1);
        value = other.value&value; 
        return *this;
    }

    static double sup() {
        return 7;
    }

    static double inf() {
        return 0;
    }
};


}

#endif // Int3_H
