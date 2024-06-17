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

template <typename Tensor>
class Neural {
public:
    virtual Tensor forward(const Tensor &input) = 0;
    virtual Tensor backward(const Tensor &error) = 0;
    virtual std::string describe() const = 0;

    friend std::ostream& operator<<(std::ostream &os, const Neural &si) {
        os << si.describe();
        return os;
    }
};

}
#endif  // NEURAL_H
