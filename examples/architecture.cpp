#include "../tensorless/types/all.h"
#include "../tensorless/layers/all.h"
#include <memory>

using namespace tensorless;
#define TYPE float8

int main() {
    auto in = TYPE::random();
    auto optimizer = SGD<TYPE>();
    auto arch = Layered<TYPE>()
                .add(std::make_shared<Dense<TYPE, 64, 64>>());
    std::cout << arch << "\n";

    for(int epoch=0;epoch<10;++epoch) {
        auto out = arch.forward(in);
        double error = ((out-in)*(out-in)).sum();
        arch.backward(in-out, optimizer);
        std::cout << "Epoch "<<epoch+1<<" squaresum "<<error<<"\n";
    }

}
