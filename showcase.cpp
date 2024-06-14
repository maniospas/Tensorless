#include "cpunn/types/float5.cpp"
#include "cpunn/types/float4.cpp"
#include "cpunn/types/float3.cpp"
#include "cpunn/signed.cpp"
#include <cmath>
#include <chrono>



int main() {
    int pos0 = 0;
    int pos1 = 1;
    auto data1 = Signed<Float3>().set(pos0, 1).set(pos1, 0.5);
    auto data2 = Signed<Float3>().set(pos0, 1.5).set(pos1, 0.45);
    auto sum = data1-data2;
    std::cout<<"Data size: "<<sum.size()<<"\n";
    std::cout << sum<<"\n";
}
