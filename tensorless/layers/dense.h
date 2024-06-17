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
    std::vector<double> biases;

protected:
    Tensor in; // keeps track of last input
    Tensor out; // keeps track of last output
    std::vector<Tensor> grads; // keeps track of gradients

public:
    Dense(int ins, int outs) : ins(ins), outs(outs) {
        weights.reserve(outs);
        for (int i=0; i<outs;++i) 
            weights.emplace(weights.end(), Tensor::random());
    }

    virtual std::string describe() const {
        std::string description;
        int paramSpace = Tensor::num_bits()*outs/8+outs*sizeof(double)/8;
        description += "Dense";
        description += "\n  Inputs   " + std::to_string(ins);
        description += "\n  Outputs  " + std::to_string(outs);
        description += "\n  Params   " + std::to_string(Tensor::num_params()*outs+outs)
                            +" ("+std::to_string(paramSpace)+" bytes, "+std::to_string(paramSpace/sizeof(float)*100/ins/outs)+"% of float32)";
        description += "\n";
        return description;
    }

    virtual Tensor forward(const Tensor& input) {
        in = input; 
        out = Tensor();
        #pragma omp parallel for
        for (int i=0;i<outs;++i) {
            Tensor weightedIns = input*weights[i]; 
            double sum = weightedIns.sum()/ins;
            if(sum>1)
                sum = 1;
            if(sum>0) {// relu
                #pragma omp critical 
                {
                    out.set(i, sum);
                }
            }
        }
        return out;
    }

    virtual Tensor backward(const Tensor &error) {
        return in;
    }

    virtual void zerograd() {
        grads.clear();
        grads.reserve(outs);
        for(int i=0;i<outs;++i)
            grads.emplace(grads.end());
    }
};

}
#endif  // TENSORLESS_DENSE_H