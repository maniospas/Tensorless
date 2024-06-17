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

#ifndef NEURAL_H
#define NEURAL_H

namespace tensorless {

// forward declaration to be used by optimizer
template <typename Tensor> 
class Neural;

// optimizer class
template <typename Tensor>
class Optimizer {
public:
    virtual void update(Tensor& param, const Tensor &grads, double lr_mult=1) = 0;
    virtual void update(double& param, double grads, double lr_mult=1) = 0;
};

// neural class
template <typename Tensor>
class Neural {
public:
    virtual Tensor forward(const Tensor &input) = 0;
    virtual Tensor backward(const Tensor &error, Optimizer<Tensor> &optimizer) = 0;
    virtual void zerograd() = 0;
    virtual std::string describe() const = 0;

    friend std::ostream& operator<<(std::ostream &os, const Neural &si) {
        os << si.describe();
        return os;
    }
};

}
#endif  // NEURAL_H
