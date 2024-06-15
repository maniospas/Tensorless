#include "../tensorless/types/all.h"
#include <cmath>
#include <chrono>

using namespace tensorless;

int main() {
    // TO SHOWCASE WORST-CASE SPEEDUPS (OPTIMIZATION IS TOO EFFICIENT FOR THE TENSORLESS BENCHMARK)
    // THIS FILE MAY BE TESTED WITHOUT ANY OPTIMIZATION FLAGS

    long N = 1000000;
    
    auto data1 = Signed<Float5>();
    auto data2 = Signed<Float5>();
    int size = data1.size();
    std::cout<<"Data size "<<size<<"\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    for(long i=0;i<N;++i) {
        data1 = data1+data2;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "--- Print the contents of data to prevent O2 from optimizing the variable out "<<data1<<"\n";
    std::cout << "Time taken for "<<N<<" cpu-parallel vector muls: " << elapsed.count() << " seconds\n";

    double* d1 = new double[size];
    double* d2 = new double[size];    
    start = std::chrono::high_resolution_clock::now();
    for(long i=0;i<N;++i)  {
        for(int j=0;j<64;++j)
            d1[j] = d1[j] + d2[j];
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time taken for "<<N<<" slow vector muls: " << elapsed.count() << " seconds\n";

}
