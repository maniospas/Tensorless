#include "../tensorless/types/all.h"


int main() {
    auto data1 = tensorless::float10().set(0, 2).set(1, 10).set(2, 15);
    auto data2 = tensorless::float10().set(0, 0.5).set(1, -5).set(2, 20);
    auto add = data1-data2;
    std::cout << add<<"\n";

    
    auto data3 = tensorless::float8().set(0, 2);
    auto data4 = tensorless::float8().set(0, 0.5);
    auto mul = data3*data4;
    std::cout << data3<<"\n";
    std::cout << data4<<"\n";
    std::cout << mul<<"\n";
}
