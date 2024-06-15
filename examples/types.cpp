#include "../tensorless/types/all.h"

int main() {
    int pos0 = 0;
    int pos1 = 1;
    auto data1 = tensorless::Signed<tensorless::Float5>().set(pos0, 1).set(pos1, 0.5);
    auto data2 = tensorless::Signed<tensorless::Float5>().set(pos0, 1.5).set(pos1, 0.45);
    auto sum = data1-data2;
    std::cout<<"Data size: "<<sum.size()<<"\n";
    std::cout << sum<<"\n";
}