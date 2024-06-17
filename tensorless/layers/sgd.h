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


#ifndef TENSORLESS_SGD_H
#define TENSORLESS_SGD_H


#include "neural.h"
#include "../types/all.h"


#include <iostream>
#include <vector>
#include "neural.h"
#include "../types/all.h"
#include <cmath>

namespace tensorless {

template <typename Tensor>
class SGD: public Optimizer<Tensor> {
    virtual Tensor update(int identifier, Tensor grads, double lr_mult=1) {
        return grads*Tensor::broadcast(lr_mult*0.01);
    }
};


}
#endif  // TENSORLESS_SGD_H
