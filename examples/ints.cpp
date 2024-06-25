#include "../tensorless/types/all.h"


int main() {
    // this test runs only on architectures that support int128
    auto data1 = tensorless::int3().set(0, -4);
    auto data2 = tensorless::int3().set(0, -4);
    std::cout << "data1         " << data1 <<"\n";
    std::cout << "data2         " << data2 <<"\n";
    std::cout << "-data2         " << data2.twosComplement() <<"\n";
    std::cout << "data1+data2         " << data2+data1 <<"\n";
    std::cout << "data1-data2         " << data1-data2 <<"\n";
    //std::cout << "data1*data2         " << data2*data1 <<"\n";
    //std::cout << "data2-data1*data2         " << data2-data2*data1 <<"\n";
    //std::cout << "data1*data2-data2         " << data2*data1-data2 <<"\n";
}
