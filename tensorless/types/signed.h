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

#ifndef SIGNED_H
#define SIGNED_H

#include <iostream>
#include <vector>
#include <bitset>
#include <cstdlib>
#include <random>
#include "vecutils.h"

namespace tensorless {

template <typename Number>
class Signed {
private:
    VECTOR isNegative;
    Number value;
    Signed(const Number &value, const VECTOR &isNeg) : value(value), isNegative(isNeg) {}
public:
    static Signed<Number> random() {
        VECTOR isNegative = lrand();
        return Signed(Number::random(), 0).twosComplement(isNegative);
    }

    static Signed<Number> broadcast(double value) {
        if(value<0)
            return Signed(Number::broadcast(-value), 0).twosComplement();
        return Signed(Number::broadcast(value), 0);
    }

    static int num_params() {
        return 1+Number::num_params();
    }

    static int num_bits() {
        return Number::num_bits() + VECTOR_SIZE;
    }

    Signed<Number> times2() const {
        Number abs = value.twosComplement(isNegative);
        return Signed(abs.times2().twosComplement(isNegative), isNegative);
    }

    Signed<Number> half() const {
        Number abs = value.twosComplement(isNegative);
        return Signed(abs.half().twosComplement(isNegative), isNegative);
    }

    Signed<Number> times2(const VECTOR &mask) const {
        Number abs = value.twosComplement(isNegative);
        return Signed(abs.times2(mask).twosComplement(isNegative), isNegative);
    }

    Signed<Number> half(const VECTOR &mask) const {
        Number abs = value.twosComplement(isNegative);
        return Signed(abs.half(mask).twosComplement(isNegative), isNegative);
    }

    Signed<Number> zerolike() const {
        return Signed();
    }

    Signed<Number> zerolike(const VECTOR &mask) const {
        return Signed(value.zerolike(mask), isNegative&~mask);
    }

    Signed<Number>& operator=(const Signed<Number>& other) {
        if (this != &other) {
            this->value = other.value;
            this->isNegative = other.isNegative;
        }
        return *this;
    }

    Signed(): isNegative(0) {}

    Signed(const std::vector<double>& vec) {
        value = Number();
        for (int i = 0; i < vec.size(); ++i) 
            if (vec[i]) {
                if(vec[i]>0)
                    value.set(i, vec[i]);
                else {
                    value.set(i, -vec[i]);
                    isNegative |= ((VECTOR)1 << i);
                }
            }
        value = value.twosComplement(isNegative);
    }

    static const double sup() {
        return Number::sup();
    }

    static const double inf() {
        return -Number::sup();
    }

    const double sum() const {
        return value.sum(~isNegative) - value.twosComplement(isNegative).sum(isNegative);
    }

    const double sum(VECTOR mask) const {
        return value.sum(~isNegative&mask) - value.twosComplement(isNegative).sum(isNegative&mask);
    }

    const double absmax() const {
        return value.twosComplement(isNegative).absmax();
    }

    const double get(int i) const {
        if((isNegative >> i) & 1) {
            return -value.twosComplement(isNegative).get(i);
        }
        return value.get(i);
    }

    Signed<Number> set(int i, int val) {
        VECTOR applyNegative = ((VECTOR)1 << i);
        if(val<0) {
            value.set(i, -val);
            VECTOR carry;
            value = value.twosComplementWithCarry(applyNegative, carry);
            #pragma omp atomic
            isNegative |= applyNegative ^ carry;
        }
        else {
            value.set(i, val);
            #pragma omp atomic
            isNegative &= ~applyNegative;
        }
        return *this;
    }

    Signed<Number> set(int i, double val) {
        VECTOR applyNegative = ((VECTOR)1 << i);
        if(val<0) {
            value.set(i, -val);
            value = value.twosComplement(applyNegative);
            #pragma omp atomic
            isNegative |= applyNegative;
        }
        else {
            value.set(i, val);
            #pragma omp atomic
            isNegative &= ~applyNegative;
        }
        return *this;
    }

    double operator[](int i) {
        return get(i);
    }

    double operator[](int i) const {
        return get(i);
    }
    
    Signed<Number>& operator[](std::pair<int, double> p) {
        set(p.first, p.second);
        return *this;
    }

    int size() const {
        return value.size();
    }

    friend std::ostream& operator<<(std::ostream &os, const Signed<Number> &si) {
        os << "[" << si.get(0);
        for(int i=1;i<10;i++)
            os << "," << si.get(i);
        os << ", ... ]";
        return os;
    }

    Signed<Number> addWithUnderflow(const Signed<Number> &other, VECTOR& underflow) const {
        VECTOR carryOut;
        Number result = value.addWithCarry(other.value, carryOut);
        VECTOR finalSign = isNegative ^ other.isNegative ^ carryOut;
        underflow = isNegative & other.isNegative & ~finalSign;
        return Signed(result, finalSign);
    }

    Signed<Number> operator+(const Signed<Number> &other) const {
        VECTOR carryOut;
        Number result = value.addWithCarry(other.value, carryOut);
        VECTOR finalSign = isNegative ^ other.isNegative ^ carryOut;

        // Detect overflow
        #ifdef DEBUG_OVERFLOWS
        // Compute final sign after addition
        //VECTOR finalSign = (isNegative & other.isNegative) | (~carryOut & signChange);
        //VECTOR overflow = (isNegative & other.isNegative) | (carryOut & (isNegative ^ other.isNegative));

        if (finalSign ^ carryOut) {
            throw std::logic_error("arithmetic overflow");
        }
        #endif

        return Signed(result, finalSign);
    }

    Signed<Number> twosComplement() const {
        VECTOR negative;
        Number complement = value.twosComplement(negative);
        return Signed(complement, negative ^ ~isNegative);
    }

    Signed<Number> twosComplement(const VECTOR &mask) const {
        VECTOR negative;
        Number complement = value.twosComplementWithCarry(mask, negative);
        return Signed(complement, ((negative ^ ~isNegative)&mask) | (isNegative&~mask) );
    }
    
    Signed<Number> operator-(const Signed<Number> &other) const {
        return *this+other.twosComplement();
    }

    Signed<Number> operator*(const Signed<Number> &other) const {
        Number ret = value.twosComplement(isNegative)*other.value.twosComplement(other.isNegative);
        VECTOR neg = isNegative ^ other.isNegative;
        return Signed<Number>(ret.twosComplement(neg), neg);
    }
    
    template <typename RetNumber> RetNumber applyShifts(const RetNumber &number) const {
        return value.applyHalf(value.applyTimes2(number, ~isNegative), isNegative);
    }

    template <typename RetNumber> RetNumber applyShifts(const RetNumber &number, const VECTOR &mask) const {
        return value.applyHalf(value.applyTimes2(number, mask&~isNegative), mask&isNegative);
    }

    Number nonNegatives() const {
        return value.zerolike(~isNegative);
    }

    Number negatives() const {
        return value.zerolike(isNegative);
    }
};

}
#endif  // SIGNED_H
