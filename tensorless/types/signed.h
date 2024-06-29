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
    inline static Signed<Number> random() {
        VECTOR isNegative = lrand();
        return Signed(Number::random(), 0).twosComplement(isNegative);
    }

    inline static Signed<Number> broadcast(double value) {
        if(value<0)
            return Signed(Number::broadcast(-value), 0).twosComplement();
        return Signed(Number::broadcast(value), 0);
    }

    inline static Signed<Number> broadcastOnes(const VECTOR &mask) {
        return Signed(Number::broadcastOnes(mask), 0);
    }

    inline static int num_params() {
        return 1+Number::num_params();
    }

    inline static int num_bits() {
        return Number::num_bits() + VECTOR_SIZE;
    }

    inline Signed<Number> times2() const {
        Number abs = value.twosComplement(isNegative);
        return Signed(abs.times2().twosComplement(isNegative), isNegative);
    }

    inline Signed<Number> half() const {
        //Number abs = value.twosComplement(isNegative);
        //return Signed(abs.half().twosComplement(isNegative), isNegative);
        return Signed(value.half(), isNegative);
    }

    inline Signed<Number> quarter() const {
        //Number abs = value.twosComplement(isNegative);
        //return Signed(abs.quarter().twosComplement(isNegative), isNegative);
        return Signed(value.quarter(), isNegative);
    }

    inline Signed<Number> eighth() const {
        //Number abs = value.twosComplement(isNegative);
        //return Signed(abs.eighth().twosComplement(isNegative), isNegative);
        return Signed(value.eighth(), isNegative);
    }

    inline Signed<Number> times2(const VECTOR &mask) const {
        Number abs = value.twosComplement(isNegative);
        return Signed(abs.times2(mask).twosComplement(isNegative), isNegative);
    }

    inline Signed<Number> half(const VECTOR &mask) const {
        //Number abs = value.twosComplement(isNegative);
        //return Signed(abs.half(mask).twosComplement(isNegative), isNegative);
        return Signed(value.half(mask), isNegative);
    }

    inline Signed<Number> quarter(const VECTOR &mask) const {
        //Number abs = value.twosComplement(isNegative);
        //return Signed(abs.quarter(mask).twosComplement(isNegative), isNegative);
        return Signed(value.quarter(mask), isNegative);
    }

    inline Signed<Number> eighth(const VECTOR &mask) const {
        //Number abs = value.twosComplement(isNegative);
        //return Signed(abs.eighth(mask).twosComplement(isNegative), isNegative);
        return Signed(value.eighth(mask), isNegative);
    }

    inline Signed<Number> relu() const {
        return zerolike(isNegative);
    }

    inline Signed<Number> zerolike() const {
        return Signed();
    }

    inline Signed<Number> zerolike(const VECTOR &mask) const {
        return Signed(value.zerolike(mask), isNegative);
    }

    inline Signed<Number>& operator=(const Signed<Number>& other) {
        if (this != &other) {
            this->value = other.value;
            this->isNegative = other.isNegative;
        }
        return *this;
    }

    inline Signed(): isNegative(0), value() {}

    inline Signed(const std::vector<double>& vec) {
        value = Number();
        for (int i = 0; i < vec.size(); ++i) 
            if (vec[i]) {
                if(vec[i]>0)
                    value.set(i, vec[i]);
                else {
                    value.set(i, -vec[i]);
                    isNegative |= ONEHOT(i);
                }
            }
        value = value.twosComplement(isNegative);
    }

    inline static const double sup() {
        return Number::sup();
    }

    inline static const double inf() {
        return -Number::sup();
    }

    inline const double sum() const {
        return value.sum(~isNegative) - value.twosComplement(isNegative).sum(isNegative);
    }

    inline const double sum(VECTOR mask) const {
        return value.sum(~isNegative&mask) - value.twosComplement(isNegative).sum(isNegative&mask);
    }

    inline const double absmax() const {
        return value.twosComplement(isNegative).absmax();
    }

    inline const double get(int i) const {
        if(GETAT(isNegative, i)) {
            return value.get(i)-(Number::sup()+Number::eps());
        }
        return value.get(i);
    }

    inline Signed<Number> set(int i, int val) {
        VECTOR applyNegative = ONEHOT(i);
        if(val<0) {
            value.set(i, Number::sup()+Number::eps()+val);
            //VECTOR carry;
            //value = value.twosComplementWithCarry(applyNegative, carry);
            // #pragma omp atomic
            isNegative |= applyNegative;// ^ carry;
        }
        else {
            value.set(i, val);
            // #pragma omp atomic
            isNegative &= ~applyNegative;
        }
        return *this;
    }

    inline Signed<Number> set(int i, double val) {
        VECTOR applyNegative = ONEHOT(i);
        if(val<0) {
            value.set(i, Number::sup()+Number::eps()+val);
            // #pragma omp atomic
            isNegative |= applyNegative;
        }
        else {
            value.set(i, val);
            // #pragma omp atomic
            isNegative &= ~applyNegative;
        }
        return *this;
    }

    inline double operator[](int i) {
        return get(i);
    }

    inline double operator[](int i) const {
        return get(i);
    }
    
    inline Signed<Number>& operator[](std::pair<int, double> p) {
        set(p.first, p.second);
        return *this;
    }

    inline int size() const {
        return value.size();
    }

    inline friend std::ostream& operator<<(std::ostream &os, const Signed<Number> &si) {
        os << "[" << si.get(0);
        for(int i=1;i<10;i++)
            os << "," << si.get(i);
        os << ", ... ]";
        return os;
    }

    inline Signed<Number> addWithUnderflow(const Signed<Number> &other, VECTOR& underflow) const {
        VECTOR carryOut;
        Number result = value.addWithCarry(other.value, carryOut);
        VECTOR finalSign = isNegative ^ other.isNegative ^ carryOut;
        underflow = isNegative & other.isNegative & ~finalSign;
        #ifdef DEBUG_OVERFLOWS
            if(ANY(~isNegative & ~other.isNegative & finalSign))
                throw std::logic_error("arithmetic overflow");
        #endif
        return Signed(result, finalSign);
    }

    inline Signed<Number> minimum(const Signed<Number> &other) const {
        Signed<Number> diff = *this-other;
        VECTOR isNeg = diff.isNegative;
        return Signed(value.merge(other.value, isNeg), (isNeg&isNegative) | (other.isNegative&~isNeg));
    }

    inline Signed<Number> maximum(const Signed<Number> &other) const {
        Signed<Number> diff = *this-other;
        VECTOR isNeg = ~diff.isNegative;
        return Signed(value.merge(other.value, isNeg), (isNeg&isNegative) | (other.isNegative&~isNeg));
    }

    inline Signed<Number> operator+(const Signed<Number> &other) const {
        VECTOR carryOut;
        Number result = value.addWithCarry(other.value, carryOut);
        VECTOR finalSign = isNegative ^ other.isNegative ^ carryOut;

        // Detect overflow
        #ifdef DEBUG_OVERFLOWS
        VECTOR overflow = (finalSign ^ isNegative) & ~(isNegative ^ other.isNegative);
        if (ANY(overflow)) {
            throw std::logic_error("arithmetic overflow or underflow");
        }
        #endif

        return Signed(result, finalSign);
    }

    inline Signed<Number> twosComplement() const {
        VECTOR negative;
        Number complement = value.twosComplementWithCarry(negative);
        return Signed(complement, negative ^ ~isNegative);
    }

    inline Signed<Number> twosComplement(const VECTOR &mask) const {
        VECTOR negative;
        Number complement = value.twosComplementWithCarry(mask, negative);
        return Signed(complement, ((negative ^ ~isNegative)&mask) | (isNegative&~mask) );
    }
    
    inline Signed<Number> operator-(const Signed<Number> &other) const {
        return *this+other.twosComplement();
    }

    inline Signed<Number> operator*(const Signed<Number> &other) const {
        Number ret1 = value.twosComplement(isNegative);
        Number ret2 = other.value.twosComplement(other.isNegative);
        Number ret = ret1*ret2;
        VECTOR neg = isNegative ^ other.isNegative;
        return Signed<Number>(ret.twosComplement(neg), neg);
    }
    
    /*template <typename RetNumber> inline RetNumber applyShifts(const RetNumber &number) const {
        return value.applyHalf(value.applyTimes2(number, ~isNegative), isNegative);
    }

    template <typename RetNumber> inline RetNumber applyShifts(const RetNumber &number, const VECTOR &mask) const {
        return value.applyHalf(value.applyTimes2(number, mask&~isNegative), mask&isNegative);
    }*/
    
    template <typename RetNumber> inline RetNumber applyHalf(const RetNumber &number) const {
        return value.applyHalf(number, ~isNegative);
    }

    template <typename RetNumber> inline RetNumber applyHalf(const RetNumber &number, const VECTOR &mask) const {
        return value.applyHalf(number, mask&~isNegative);
    }

    inline Number nonNegatives() const {
        return value.zerolike(~isNegative);
    }

    inline Number negatives() const {
        return value.zerolike(isNegative);
    }
};

}
#endif  // SIGNED_H
