#include "../tensorless/types/all.h"

using namespace tensorless;

int main() {
    int pos0 = 0;
    int pos1 = 1;
    auto data1 = float8().set(pos0, 1).set(pos1, 0.5);
    auto data2 = float8().set(pos0, 1.5).set(pos1, 0.45);
    auto sum = data1-data2;
    std::cout << "Stored numbers: " << sum.size() << "\n";
    std::cout << "Used bytes: " << sum.num_bits()/8 << "\n";
    std::cout << sum<<"\n";
}
