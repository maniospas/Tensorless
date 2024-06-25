#include "../tensorless/types/all.h"

using namespace tensorless;

int main() {
    auto data1 = float14().set(0, -25).set(1, 0.2).set(2, -0.05); 
    auto data2 = float14().set(0, 2.5).set(1, 0.2).set(2, 0.05); 
    
    std::cout << "Stored numbers: " << data1.size() << "\n";
    std::cout << "Used bytes: " << data1.num_bits()/8 << "\n";
    std::cout << "Data1 " << data1 << "\n";
    std::cout << "Data2 " << data2 << "\n";
    std::cout << "Mult " << data1*data2 << "\n";
    std::cout << "Sum " << data1+data2 << "\n";
}
