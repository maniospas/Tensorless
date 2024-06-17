#include "../tensorless/types/all.h"
#include <cmath>
#include <chrono>

using namespace tensorless;

int main() {
    // WITHOUT O2, TENSORLESS IS SLOWER, BUT OTHERWISE IS x70 times faster for Signed Float5 (1+5=6 bits)
    // Signed Float8 also needs greater inlining limits (see readme) for inline optimization.

    long N = 1000000;
    
    auto data1 =float8::random();
    auto data2 = float8::random();
    int size = data1.size();
    std::cout<<"Data size "<<size<<"\n";
    
    double res0 = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for(long i=0;i<N;++i) {
        res0 += (data1*data2).sum();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken for "<<N<<" cpu-parallel vector muls: " << elapsed.count() << " seconds\n";

    double* d1 = new double[size];
    double* d2 = new double[size]; 
    start = std::chrono::high_resolution_clock::now();
    double res = 0;
    for(long i=0;i<N;++i)  {
        for(int j=0;j<64;++j)
            res += d1[j] * d2[j];
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time taken for "<<N<<" slow vector muls: " << elapsed.count() << " seconds\n";

    
    std::cout << res0<<" "<<res<<"\n";

}
