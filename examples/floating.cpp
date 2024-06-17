#include "../tensorless/types/all.h"

using namespace tensorless;

int main() {
    auto data1 = Dynamic<SFloat8>().set(0, 0.52).set(1, 0.02).set(2, 0.02);
    std::cout << "Stored numbers: " << data1.size() << "\n";
    std::cout << "Used bytes: " << data1.num_bits()/8 << "\n";
    
    auto data2 = Dynamic<SFloat8>().set(0, 0.5);
    std::cout << data1.sum() <<"\n";
}
