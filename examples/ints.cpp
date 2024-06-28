#include "../tensorless/types/all.h"


int main() {
    // this test runs only on architectures that support int128
    auto data1 = tensorless::int2().set(0, -1);
    auto data2 = tensorless::int2().set(0, 2);
    std::cout << "data1         " << data1 <<"\n";
    std::cout << "data2         " << data2 <<"\n";
    std::cout << "-data2        " << data2.twosComplement() <<"\n";
    std::cout << "data1+data2         " << data2+data1 <<"\n";
    std::cout << "max         " << data1.maximum(data2) <<"\n";
    std::cout << "min         " << data1.minimum(data2) <<"\n";
    //std::cout << "data1*data2         " << data2*data1 <<"\n";
    //std::cout << "data2-data1*data2         " << data2-data2*data1 <<"\n";
    //std::cout << "data1*data2-data2         " << data2*data1-data2 <<"\n";
}
