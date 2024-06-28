# Tensorless

*Make the CPU behave like a GPU for quantized data.*

**Author:** Emmanouil (Manios) Krasanakis<br>
**Language:** C++<br>
**License:** Apache 2.0

:warning: Mandatory compilation parameters: `-O2 -fopenmp -finline-limit=1000 -fearly-inlining`. All these are necessary for performant inlining.


## :fire: CPU vectorization

- [x] Get GPU-like parallelization and speed in the CPU for quantized numbers.
- [x] Copy-paste header installation.
- [ ] Floating point formats.
- [ ] Ready-to-use neural components.
- [ ] Python interface.

## :rocket: Quickstart

Here's how to use the `float8` data type: 
this packs 128 numbers discertized into 8 bits of numerical precision, 1 sign bit,
and a shared double mantissa.


```cpp
#include "tensorless/types/all.h"

using namespace tensorless;

int main() {
    int pos0 = 0;
    int pos1 = 1;
    auto data1 = float8().set(pos0, 1).set(pos1, 0.5);
    auto data2 = float8().set(pos0, 1.5).set(pos1, 0.45);
    auto sum = data1-data2;
    std::cout << "Stored numbers: " << sum.size() << "\n";  // 128
    std::cout << "Used bytes: " << sum.num_bits()/8 << "\n";  // 152
    std::cout << sum << "\n";  // -0.503906,0.046875,0,0,0,0,0,0,0,0, ... ]
}
```


## :brain: About

No tensors under-the-hood: 128x 8bit numbers are packed into 8x 128bit numbers.
The latter are few in number and thus can be processed all at once in CPU
registers. This is much faster than memory transfers (normal SIMD optimizations
still apply to more efficiently run operations).
Matrix multiplication of neural components are further parallelized with openmp.







