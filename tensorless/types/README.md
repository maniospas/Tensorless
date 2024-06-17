# Tensorless/Types

The [tensorless/types](tensorless/types/README.md) package vectorizes quantized numbers.
It has no external dependencies and can be used without the rest of the `tensorless` framework.

**Author:** Emmanouil (Manios) Krasanakis<br>
**License:** Apache 2.0

:warning: Mandatory compilation parameters: `-O2 -fopenmp` and, if `Float8` or its derivative types are used, `-finline-limit=400` or larger.

## Quickstart

```cpp
#include "../tensorless/types/all.h"
#include "../tensorless/layers/all.h"
#include <memory>

using namespace tensorless;

int main() {
    auto in = TYPE::random(); // in range [1,1]
    auto optimizer = SGD<float8>(0.001);
    auto arch = Layered<float8>()
                .add(std::make_shared<Dense<float8, 64, 64>>())
                .add(std::make_shared<Dense<float8, 64, 64>>());
    std::cout << arch << "\n";  // print the architecture

    for(int epoch=0;epoch<30;++epoch) {
        auto out = arch.forward(in);
        double error = ((out-in)*(out-in)).sum();
        arch.backward(in-out, optimizer);
        std::cout << "Epoch "<<epoch+1<<" squaresum "<<error<<"\n";
    }

}
```
