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

#ifndef FLOATING_H
#define FLOATING_H

#include <iostream>
#include <vector>
#include <bitset>
#include <cstdlib>
#include <random>
#include "vecutils.h"
#include <omp.h>

namespace tensorless {

template <typename Number, typename Mantisa>
class Floating {
private:
    Mantisa mantisa;
    Number value;
    Floating(const Number& value, const Mantisa& mantisa) : value(value), mantisa(mantisa) {}
public:
    // standard declarations
    Floating() : value(), mantisa() {}
    static Floating<Number, Mantisa> random() {return Floating(Number::random(), Mantisa::broadcast(0));}
    static int num_params() {return Number::num_params() + Mantisa::num_params();} 
    static int num_bits() {return Number::num_bits() + Mantisa::num_bits();}
    Floating<Number, Mantisa> times2() const {return Floating(value, mantisa+Mantisa::broadcast(1));}
    Floating<Number, Mantisa> zerolike() const {return Floating();}
    Mantisa getMantisa() const {return mantisa;}
    Number getBody() const {return value;}
    const double get(int i) const {
        int mant=(int)mantisa.get(i); 
        if(mant<0)
            return value.get(i)/(1<<-mant);
        return value.get(i)*(1<<mant);
    }
    const bool isZeroAt(int i) const {return value.isZeroAt(i);}
    double operator[](int i) {return get(i);}
    double operator[](int i) const {return get(i);}
    Floating<Number, Mantisa>& operator[](std::pair<int, double> p) {set(p.first, p.second);return *this;}
    int size() const {return value.size();}

    // print
    friend std::ostream& operator<<(std::ostream &os, const Floating<Number, Mantisa> &si) {
        os << "[" << si.get(0);
        for(int i=1;i<10;i++)
            os << "," << si.get(i);
        os << ", ... ]";
        return os;
    }

    // setters
    static Floating<Number, Mantisa> broadcast(double val) {
        int mantisa = 0;
        if(val) {
            while(val>1 || val<-1) {
                val /= 2;
                mantisa++;
            }
            while(val<0.5 && val>-0.5) {
                val *= 2;
                mantisa--;
            }
        }
        return Floating(Number::broadcast(val), Mantisa::broadcast(mantisa));
    }

    Floating<Number, Mantisa>& set(int i, double val) {
        int mant = 0;
        if(val) {
            int mantsup = Mantisa::sup();
            int mantinf = -mantsup;
            double sup = Number::sup()*0.5; // this is enabled here to avoid performing a convertion within all operations
            while((val>sup || val<-sup) && mant < mantsup) {
                val /= 2;
                mant++;
            }
            sup *= 0.5;
            while(val<sup && val>-sup && mant > mantinf) {
                val *= 2;
                mant--;
            }
        }
        value.set(i, val);
        mantisa.set(i, mant);
        return *this;
    }

    // operations
    const double sum() const {
        return value.sum();
    }
    

    /*const double sum(VECTOR mask) const {
        return value.sum(mask)*mantisa;
    }

    const double absmax() const {
        return value.absmax();
    }

    Floating<Number> operator-(const Floating<Number> &other) const {
        if(other.mantisa<mantisa)
            return Floating(value-other.value*Number::broadcast(other.mantisa/mantisa), mantisa);
        else if(other.mantisa==mantisa)
            return Floating(value-other.value, other.mantisa);
        else
            return Floating(value*Number::broadcast(mantisa/other.mantisa)-other.value, other.mantisa);
    }
    */

    Floating<Number, Mantisa> operator+(const Floating<Number, Mantisa> &other) const {
        Mantisa diff = mantisa-other.mantisa;
        Mantisa selfDiff = diff.relu();
        Mantisa otherDiff = diff.twosComplement().relu();
        Number selfValue = otherDiff.applyHalf(value);
        Number otherValue = selfDiff.applyHalf(other.value);

        return Floating<Number, Mantisa>((selfValue+otherValue).half(), 
                                         mantisa.maximum(other.mantisa)+Mantisa::broadcastOnes(~(VECTOR)0));
    }

    Floating<Number, Mantisa> operator-(const Floating<Number, Mantisa> &other) const {
        Mantisa diff = mantisa-other.mantisa;
        Mantisa selfDiff = diff.relu();
        Mantisa otherDiff = diff.twosComplement().relu();
        Number selfValue = otherDiff.applyHalf(value);
        Number otherValue = selfDiff.applyHalf(other.value);

        return Floating<Number, Mantisa>(selfValue-otherValue, mantisa.maximum(other.mantisa));
    }

    Floating<Number, Mantisa> operator*(const Floating<Number, Mantisa> &other) const {
        VECTOR underflow;
        Mantisa newMantisa = mantisa.addWithUnderflow(other.mantisa, underflow);
        return Floating<Number, Mantisa>(value.zerolike(underflow)*other.value, newMantisa);
    }

    
    Floating<Number, Mantisa> operator*(const double other) const {
        return *this+broadcast(other);
    }


    Floating<Number, Mantisa>& operator=(const Floating<Number, Mantisa>& other) {
        if (this != &other) {
            this->value = other.value;
            this->mantisa = other.mantisa;
        }
        return *this;
    }
};

}
#endif  // FLOATING_H
