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


#ifndef TENSORLESS_CONV_H
#define TENSORLESS_CONV_H

#include <iostream>
#include <vector>
#include "neural.h"
#include "../types/all.h"
#include <cmath>

namespace tensorless {

template <typename Tensor, int ins, int outs>
class Conv: public Neural<Tensor> {
private:
    Tensor weights[outs];
    double biases[outs];
    bool activations[outs];

public:
    Conv() {
        for (int i=0; i<outs;++i) 
            weights[i] = Tensor::random();
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
        Tensor in = input; 
        Tensor out = Tensor();
        double scale = 2*std::sqrt(12.0/outs);
        //#pragma omp parallel for
        for (int i=0;i<outs;++i) {
            Tensor weightedIns = input*weights[i]; 
            double sum = weightedIns.sum()*scale+biases[i];
            activations[i] = sum>0;
            if(activations[i]) { // relu
                #pragma omp critical 
                {
                    out.set(i, sum);
                }
            }
        }
        return out;
    }

    virtual Tensor backward(const Tensor &error, Optimizer<Tensor> &optimizer) {
        return error;
    }

    virtual void zerograd() {
    }
};

}
#endif  // TENSORLESS_CONV_H