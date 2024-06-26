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
#include "neural.h"
#include "../types/all.h"
#include <cmath>

namespace tensorless {

template <typename Tensor, int ins, int outs>
class Dense: public Neural<Tensor> {
private:
    Tensor weights[outs];
    double biases[outs];
    bool activations[outs];

public:
    Dense() {
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
                            +" ("+std::to_string(paramSpace)+" bytes, "+std::to_string(paramSpace/sizeof(float)*100/ins/outs)+"% of float)";
        description += "\n";
        return description;
    }

    virtual Tensor forward(const Tensor& input) {
        Tensor in = input; 
        Tensor out = Tensor();
        bool activation;
        double sum;
        Tensor weightedIns;
        //#pragma omp parallel for shared(input, weights, out) private(weightedIns, sum, activation)
        for (int i=0;i<outs;++i) {
            weightedIns = input*weights[i]; 
            sum = weightedIns.sum();
            activation = sum>0;
            if(activation) { // relu
                //#pragma omp critical 
                {
                    out.set(i, sum);
                }
            }
        }
        return out;
    }

    virtual Tensor backward(const Tensor &error, Optimizer<Tensor> &optimizer) {
        /*Tensor err;
        double scale = 2*std::sqrt(12.0/outs);
        double invscale = 1./scale;
        double errorsum = error.sum();
        //#pragma omp parallel for
        for (int i=0;i<outs;++i) {
            if(!activations[i])
                continue;
            Tensor erri = weights[i]*error*scale;
            optimizer.update(weights[i], error, invscale);
            //optimizer.update(biases[i], errorsum, invscale);
            #pragma omp critical
            {
                err = err + erri;
            }
        }
        return err;*/
        return error;
    }

    virtual void zerograd() {
    }
};

}
#endif  // TENSORLESS_DENSE_H