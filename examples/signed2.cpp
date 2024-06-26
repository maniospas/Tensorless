#include "../tensorless/types/all.h"


int main() {
    auto data1 = tensorless::float14().set(0, 0.4).set(1, -0.4).set(2, 0.3).set(3, -0.664062).set(4, -0.46875);
    auto data2 = tensorless::float14().set(0, -0.6).set(1, -0.5).set(2, 0.5).set(3, 0).set(4, 0);
    std::cout << "data1         " << data1 <<"\n";
    std::cout << "data2         " << data2 <<"\n";
    std::cout << "data2-data1         " << data2-data1 <<"\n";
    std::cout << "data1-data2         " << data1-data2 <<"\n";
    std::cout << "data1*data2         " << data2*data1 <<"\n";
    std::cout << "data2-data1*data2         " << data2-data2*data1 <<"\n";
    std::cout << "data1*data2-data2         " << data2*data1-data2 <<"\n";
}
