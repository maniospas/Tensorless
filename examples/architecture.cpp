#include "../tensorless/types/all.h"
#include "../tensorless/layers/all.h"
#include "../tensorless/types/fixed.h"
#include <memory>
#include <chrono>

using namespace tensorless;
//typedef Fixed<double, 128> TYPE;
typedef float8 TYPE;

int main() {
    auto in = TYPE::random();
    auto optimizer = SGD<TYPE>(0.1);
    auto arch = Layered<TYPE>()
                .add(std::make_shared<Dense<TYPE, 128, 128>>());
    std::cout << arch << "\n";

    auto start = std::chrono::high_resolution_clock::now();
    double s = 0;
    for(long epoch=0;epoch<50000;++epoch) {
        //arch.zerograd();
        auto out = arch.forward(in);
        //auto error = (out-in);
        //arch.backward(out-in, optimizer);
        //std::cout << "Epoch "<<epoch+1<<" squaresum "<<error<<"\n";
        //std::cout << "Epoch "<<epoch+1<<" "<<error<<"\n";
        s += out.sum();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << ((std::chrono::duration<double>)(end - start)).count() << " secs" << std::endl;

}
