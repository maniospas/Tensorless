#include "../tensorless/types/all.h"
#include "../tensorless/layers/all.h"
#include <memory>
#include <chrono>

using namespace tensorless;
#define TYPE float8

int main() {
    auto in = TYPE::random();
    auto optimizer = SGD<TYPE>(0.01);
    auto arch = Layered<TYPE>()
                .add(std::make_shared<Dense<TYPE, 128, 128>>());
    std::cout << arch << "\n";
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << omp_get_num_threads() << " threads\n";
    }


    auto start = std::chrono::high_resolution_clock::now();
    for(int epoch=0;epoch<3000;++epoch) {
        auto out = arch.forward(in);
        //double error = ((out-in)*(out-in)).sum();
        //arch.backward(in-out, optimizer);
        //std::cout << "Epoch "<<epoch+1<<" squaresum "<<error<<"\n";
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << ((std::chrono::duration<double>)(end - start)).count() << " secs" << std::endl;

}
