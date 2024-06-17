#include "../tensorless/types/all.h"

int main() {
    // this test runs only on architectures that support int128
    auto data1 = tensorless::float8().set(0, 2).set(1, 20).set(2, 35);
    auto data2 = tensorless::float8().set(0, 0.5).set(1, -5).set(2, 20);
    auto add = data1-data2;
    std::cout << add<<"\n";

    
    auto data3 = tensorless::float8().set(0, 2);
    auto data4 = tensorless::float8().set(0, 0.5);
    auto mul = data3*data4;
    std::cout << data3<<"\n";
    std::cout << data4<<"\n";
    std::cout << mul<<"\n";
}
