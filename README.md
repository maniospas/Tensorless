# Tensorless

*Make the CPU behave like a GPU.*

**Author:** Emmanouil (Manios) Krasanakis<br>
**License:** Apache 2.0


## :fire: CPU vectorization

- No tensors under-the-hood: 6x 128bit numbers store 128x 6bit numbers.
- No SIMD-specific programming: get parallelization everwhere.
- Ready-to-use neural components.
- Copy-paste header installation.

## :rocket: Quickstart

Compile with the O2 or O3 flag for fast execution.

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

Let's say that we have a numerical vector of (up to 128) elements. 
First, quantize each of the elemnts into N bits. Sumbolically, let's write as A[k][n] the n-th bit of the k-th element. Using these, create integers I[n] for n=0..N-1 that are 128 bits wide. Each of those integers I[n] holds on its k-th bit I[n][k]=A[k][n] for k=0..128.

We are now in the position of running operations on A[k] in parallel by converting them (compiling them by hand) into B bitwise operations. These operations run for all number, which means that we sped up clock 
operations at least to B/128 of the original ones. This works in addition to other optimizations, like SIMD.

For example, let's say that you had 8-bit quantized numbers. Using normal SIMD SIMD parallelization we would in theory able to submit
128/8=16 additions within one clock cycle.
However, practical implementations
are rarely as kind; even with cumbersomly
optimized code, many times the number 
of cycles are needed to read and write
data from RAM and organize them into
SIMD instuctions.

To the contrary, `cpunn`'s addition requires
only 16 bitwise operations, which in theory
yields a worse rate of
128 additions within 16 clock cycles (which
averages to half the rate of)
128/16=8 additions per cycle, but *incures no
overhreads*. Data reside in processor
registers and low-level cache, wehich means
that any memory access is performed directly
on the metal.







