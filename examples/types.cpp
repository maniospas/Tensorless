#include "../tensorless/types/all.h"

int main() {
    int pos0 = 125;
    int pos1 = 126;
    int pos2 = 127;
    auto data1 = tensorless::Signed<tensorless::Float5>().set(pos0, -1).set(pos1, 0.125).set(pos2, 0.25);
    auto data2 = tensorless::Signed<tensorless::Float5>().set(pos0, -0.5).set(pos1, -1.5).set(pos2, 1);
    std::cout<<"Number of dimensions  "<<data1.size()<<"\n\n";

    std::cout << "data1[125]          " << data1[125] <<"\n";
    std::cout << "data1.sum()         " << data1.sum() <<"\n";
    std::cout << "data2.sum()         " << data2.sum() <<"\n\n";
    
    auto add = data1+data2;
    std::cout << "data1+data2         " << add <<"\n";
    std::cout << "(data1+data2).sum() " << add.sum() <<"\n\n";

    auto diff = data1-data2;
    std::cout << "data1-data2         " << diff <<"\n";
    std::cout << "(data1-data2).sum() " << diff.sum() <<"\n\n";

    std::cout << "data1*data2         " << data1*data2 <<"\n";
}
