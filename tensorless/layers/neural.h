#ifndef NEURAL_H
#define NEURAL_H

namespace tensorless {

template <typename Tensor>
class Neural {
    virtual Tensor forward(Tensor input) = 0;
    virtual Tensor backward(Tensor error) = 0;
};

}
#endif  // NEURAL_H
