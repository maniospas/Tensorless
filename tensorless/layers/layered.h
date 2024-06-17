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


#ifndef TENSORLESS_LAYERED_H
#define TENSORLESS_LAYERED_H

#include <iostream>
#include <vector>
#include <memory>
#include "neural.h"
#include "../types/all.h"

namespace tensorless {

template <typename Tensor>
class Layered: public Neural<Tensor> {
protected:
    std::vector<std::shared_ptr<Neural<Tensor>>> layers;

public:
    Layered() {
    }
    
    virtual std::string describe() const {
        std::string description;
        description += "-----------------------------------------------\n";
        for(const auto& layer : layers) 
            description += layer->describe();
        description += "-----------------------------------------------\n";
        return description;
    }

    Layered& add(const std::shared_ptr<Neural<Tensor>> &layer) {
        layers.push_back(layer);
        return *this;
    }

    virtual Tensor forward(const Tensor &input) {
        Tensor in = input;
        for(const auto& layer : layers) 
            in = layer->forward(in);
        return in;
    }

    virtual Tensor backward(const Tensor &error, Optimizer<Tensor> &optimizer) {
        Tensor err = err;
        for(int i=layers.size()-1;i>=0;--i)
            err = layers[i]->backward(err, optimizer);
        return err;
    }

    virtual void zerograd() {
        for(const auto& layer : layers) 
            layer->zerograd();
    }
};

}
#endif  // TENSORLESS_LAYERED_H