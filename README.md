# Tensorless

*Make the CPU behave like a GPU for quantized data.*

**Author:** Emmanouil (Manios) Krasanakis<br>
**Language:** C++<br>
**License:** Apache 2.0

:warning: Mandatory compilation parameters: `-O2 -fopenmp` and, if `Float8` or its derivative types are used, `-finline-limit=400` or larger.


## :fire: CPU vectorization

- [x] Get GPU-like parallelization and speed in the CPU for quantized numbers.
- [x] Copy-paste header installation.
- [ ] Ready-to-use neural components.

## :rocket: Quickstart

```cpp
#include "tensorless/types/all.h"

using namespace tensorless;

int main() {
    int pos0 = 0;
    int pos1 = 1;
    auto data1 = Signed<Float3>().set(pos0, 1).set(pos1, 0.5);
    auto data2 = Signed<Float3>().set(pos0, 1.5).set(pos1, 0.45);
    auto sum = data1-data2;
    std::cout<<"Data size: "<<sum.size()<<"\n";
    std::cout << sum<<"\n";
}
```


## :brain: About

No tensors under-the-hood: 128x 8bit numbers are packed into 8x 128bit numbers.
The latter are few in number and thus can be processed all at once in CPU
registers. This is much faster than memory transfers (normal SIMD optimizations
still apply to more efficiently run operations).
Matrix multiplication of neural components are further parallelized with openmp.







