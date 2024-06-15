# Tensorless/Types

The `tensorless/types` package has no external dependencies
and can be used without the rest of the `tensorless` framework.
Use it to create vectorized and highly compressed datatypes.

**Author:** Emmanouil (Manios) Krasanakis<br>
**License:** Apache 2.0

## Quickstart

Compile with the O2 or O3 flag for fast execution.

```cpp
#include "tensorless/types/all.h"

int main() {
    int pos0 = 0;
    int pos1 = 1;
    auto data1 = tensorless::Signed<tensorless::Float5>().set(pos0, 1).set(pos1, 0.5);
    auto data2 = tensorless::Signed<tensorless::Float5>().set(pos0, 1.5).set(pos1, 0.45);
    auto sum = data1-data2;
    std::cout<<"Data size: "<<sum.size()<<"\n";  // this will be fixed once you compile
    std::cout << sum<<"\n";
}
```
