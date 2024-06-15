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


#ifndef VECTORLESS_DENSE_H
#define VECTORLESS_DENSE_H

#include <iostream>
#include <vector>
#include <random>
#include <bitset>
#include <climits>
#include <cmath>
#include "types/all.h"

namespace tensorless {

template <typename Tensor>
class Dense: public Neural<Tensor> {
private:
    int ins;
    int outs;
    std::vector<Tensor> weights;

protected:
    Tensor in; // keeps track of last input
    Tensor out; // keeps track of last output

public:
    Dense(int ins, int outs, int broadcast=1) : ins(ins), outs(outs) {
        for (int i=0; i<outs;++i) 
            weights.push_back(Tensor::random());
    }

    friend std::ostream& operator<<(std::ostream &os, const Dense *si) {
        os << "Bernouli layer";
        os << "\n  Inputs  " << si->ins;
        os << "\n  Outputs " << si->outs;
        os << "\n  Weights";
        for (int i=0;i<si->outs;++i) 
            os << "\n   " << si->weights[i];
        os << "\n";
        return os;
    }

    Tensor forward(Tensor input) {
        in = input; 
        //for(int i=1;i<broadcast;++i) 
        //    in += input << (ins*i);

        out = Tensor();
        for (int i=0;i<outs;++i) {
            Tensor weightedIns = in*weights[i]; // *Tensor:random();
            if(weightedIns)  // if any bit different than zero
                out += Tensor(1 << i);
        }
        return out;
    }

    Tensor backward(Tensor error) {
        Tensor back = Tensor();
        Tensor mask = Tensor::random();
        Tensor notmask = ~mask; // is boolean
        mask = ~mask; // now is boolean
        for (int i=0;i<outs;++i) 
            if(!error.isZeroAt(i)) {
                back += weights[i];
                weights[i] = (notin*mask) + (weights[i]*notmask);
            }
        return DATA(back);
    }
};

}
#endif  // VECTORLESS_DENSE_H