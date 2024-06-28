#include "../tensorless/types/all.h"
#include "../tensorless/layers/all.h"
#include <memory>
#include <chrono>

using namespace tensorless;
#define TYPE dfloat8

int main() {
    auto in = TYPE::random();
    auto optimizer = SGD<TYPE>(0.1);
    auto arch = Layered<TYPE>()
                .add(std::make_shared<Dense<TYPE, 128, 128>>());
    std::cout << arch << "\n";
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << omp_get_num_threads() << " threads\n";
    }


    auto start = std::chrono::high_resolution_clock::now();
    for(int epoch=0;epoch<50;++epoch) {
        arch.zerograd();
        auto out = arch.forward(in);
        auto error = (out-in);
        arch.backward(out-in, optimizer);
        //std::cout << "Epoch "<<epoch+1<<" squaresum "<<error<<"\n";
        std::cout << "Epoch "<<epoch+1<<" "<<error<<"\n";
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << ((std::chrono::duration<double>)(end - start)).count() << " secs" << std::endl;

}
