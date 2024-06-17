#include "../tensorless/types/all.h"


int main() {
    // this test runs only on architectures that support int128
    auto data1 = tensorless::SFloat8().set(0, 0.4).set(1, -0.4).set(2, 0.3).set(3, -0.664062).set(4, -0.46875);
    auto data2 = tensorless::SFloat8().set(0, -0.6).set(1, -0.5).set(2, 0.5).set(3, 0).set(4, 0);
    std::cout << "data1-data2         " << data2-data1 <<"\n";
    std::cout << "data1-data2         " << data1-data2 <<"\n";
    std::cout << "data1*data2         " << data2*data1 <<"\n";
    std::cout << "data2-data1*data2         " << data2-data2*data1 <<"\n";
    std::cout << "data1*data2-data2         " << data2*data1-data2 <<"\n";
}
