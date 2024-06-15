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


#ifndef TENSORLESS_DENSE_H
#define TENSORLESS_DENSE_H

#include <iostream>
#include <vector>
#include <random>
#include <bitset>
#include <climits>
#include <cmath>
#include "neural.h"
#include "../types/all.h"

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
    std::vector<Tensor> activations;

public:
    Dense(int ins, int outs) : ins(ins), outs(outs) {
        for (int i=0; i<outs;++i) 
            weights.push_back(Tensor::random());
    }

    friend std::ostream& operator<<(std::ostream &os, const Dense *si) {
        os << "Layer";
        os << "\n  Inputs  " << si->ins;
        os << "\n  Outputs " << si->outs;
        /*os << "\n  Weights";
        for (int i=0;i<si->outs;++i) 
            os << "\n   " << si->weights[i];*/
        os << "\n";
        return os;
    }

    virtual Tensor forward(const Tensor& input) {
        in = input; 
        out = Tensor();
        #pragma omp parallel for
        for (int i=0;i<outs;++i) {
            Tensor weightedIns = input*weights[i]; 
            double sum = weightedIns.sum()/ins*8;
            if(sum>1)
                sum = 1;
            if(sum>0) {// relu
                out.set(i, sum);
            }
        }
        return out;
    }

    virtual Tensor backward(const Tensor &error) {
        return in;
    }
};

}
#endif  // TENSORLESS_DENSE_H