#include "../tensorless/types/all.h"
#include "../tensorless/layers/dense.h"

#define TYPE tensorless::Signed<tensorless::Float5>

int main() {
    auto input = TYPE().set(0, 0.5);
    auto layer = new tensorless::Dense<TYPE>(32, 32);
    auto out = layer->forward(input);
    std::cout << out << "\n";
}
