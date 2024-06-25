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

#ifndef DYNAMIC_H
#define DYNAMIC_H

#include <iostream>
#include <vector>
#include <bitset>
#include <cstdlib>
#include <random>
#include "vecutils.h"
#include <omp.h>

namespace tensorless {

template <typename Number>
class Dynamic {
private:
    double mantisa;
    Number value;
    Dynamic(const Number& value, double mantisa) : value(value), mantisa(mantisa) {}
public:
    static Dynamic<Number> random() {return Dynamic(Number::random(), 1);}  // 2.0/Number::sup()
    static Dynamic<Number> broadcast(double value) {
        //double scale = Number::sup()/2.0;
        if(value<0)
            return Dynamic(Number::broadcast(-1), -value);
        return Dynamic(Number::broadcast(1), value);
    }

    static int num_params() {
        return 1+Number::num_params();
    }

    static int num_bits() {
        return Number::num_bits() + sizeof(double)*8;
    }

    Dynamic<Number> times2() {
        return Dynamic(value, mantisa*2);
    }

    Dynamic<Number> zerolike() {
        return Dynamic();
    }

    Dynamic<Number>& operator=(const Dynamic<Number>& other) {
        if (this != &other) {
            this->value = other.value;
            this->mantisa = other.mantisa;
        }
        return *this;
    }

    Dynamic(): mantisa(1) {}

    Dynamic(const std::vector<double>& vec) {
        value = Number();
        double maxElement = 0;
        for (int i = 0; i < vec.size(); ++i) {
            double value = vec[i];
            if(value<0)
                value = -value;
            if(value>maxElement)
                maxElement = value;
        }
        if(maxElement==0)
            maxElement = 1;
        for (int i = 0; i < vec.size(); ++i) 
            if (vec[i]) 
                value.set(i, vec[i]/maxElement);
        mantisa = maxElement;
    }

    const double sum() const {
        return value.sum()*mantisa;
    }

    const double sum(VECTOR mask) const {
        return value.sum(mask)*mantisa;
    }

    const double absmax() const {
        return value.absmax();
    }

    const double get(int i) {
        return value.get(i)*mantisa;
    }

    const double get(int i) const {
        return value.get(i)*mantisa;
    }

    Dynamic<Number>& set(int i, double val) {
        double absval = val;
        if(absval<0)
            absval = -absval;
        
        if(absval>mantisa) {
            double mult = mantisa/absval;
            value = value*Number::broadcast(mult);
            mantisa = absval;
            value.set(i, val>0?1:-1);
        }
        else
            value.set(i, val/mantisa);
        return *this;
    }

    double operator[](int i) {
        return get(i);
    }

    double operator[](int i) const {
        return get(i);
    }
    
    Dynamic<Number>& operator[](std::pair<int, double> p) {
        set(p.first, p.second);
        return *this;
    }

    int size() const {
        return value.size();
    }

    friend std::ostream& operator<<(std::ostream &os, const Dynamic<Number> &si) {
        os << "[" << si.get(0);
        for(int i=1;i<10;i++)
            os << "," << si.get(i);
        os << ", ... ]";
        return os;
    }

    Dynamic<Number> operator+(const Dynamic<Number> &other) const {
        if(other.mantisa<mantisa)
            return Dynamic(value+other.value*Number::broadcast(other.mantisa/mantisa), mantisa);
        else
            return Dynamic(value*Number::broadcast(mantisa/other.mantisa)+other.value, other.mantisa);
    }

    Dynamic<Number> operator-(const Dynamic<Number> &other) const {
        if(other.mantisa<mantisa)
            return Dynamic(value-other.value*Number::broadcast(other.mantisa/mantisa), mantisa);
        else if(other.mantisa==mantisa)
            return Dynamic(value-other.value, other.mantisa);
        else
            return Dynamic(value*Number::broadcast(mantisa/other.mantisa)-other.value, other.mantisa);
    }

    Dynamic<Number> operator*(const Dynamic<Number> &other) const {
        return Dynamic<Number>(value*other.value, mantisa*other.mantisa);
    }

    Dynamic<Number> operator*(const double &other) const {
        return Dynamic<Number>(value, mantisa*other);
    }

};

}
#endif  // DYNAMIC_H
