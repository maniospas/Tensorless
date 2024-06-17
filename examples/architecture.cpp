#include "../tensorless/types/all.h"
#include "../tensorless/layers/all.h"
#include <memory>

using namespace tensorless;
#define TYPE SFloat8

int main() {
    auto in = TYPE::random();
    auto arch = Layered<TYPE>()
                .add(std::make_shared<Dense<TYPE>>(64, 64))
                .add(std::make_shared<Dense<TYPE>>(64, 64));
    std::cout << arch << "\n";


    auto out = arch.forward(in);
    auto error = out-in;
    arch.backward(out-in);

    std::cout << in << "\n";
    std::cout << out << "\n";
}
