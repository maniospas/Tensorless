#include "../tensorless/types/all.h"
#include <cmath>
#include <chrono>

using namespace tensorless;
typedef dfloat10 floatX; // change this to benchmark different datatypes

int main() {
    long N = 1000000;

    auto data1 = floatX::random();
    auto data2 = floatX::random();
    int size = data1.size();
    std::cout<<"Data size "<<size<<"\n";
    
    double res0 = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for(long i=0;i<N;++i) {
        double res = (data1*data2).sum();
        res += (data1*data2).sum();
        res += (data1*data2).sum();
        res += (data1*data2).sum();
        res0 += res;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken for "<<N<<" cpu-parallel vector muls: " << elapsed.count() << " seconds\n";

    double* d1 = new double[size];
    double* d2 = new double[size]; 
    for(int j=0;j<size;++j) {
        d1[j] = data1.get(j);
        d2[j] = data2.get(j);
    }
    start = std::chrono::high_resolution_clock::now();
    double res1 = 0;
    for(long i=0;i<N;++i)  {
        for(int j=0;j<size;++j) {
            double res = d1[j] * d2[j];
            res += d1[j] * d2[j];
            res += d1[j] * d2[j];
            res += d1[j] * d2[j];
            res1 += res;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time taken for "<<N<<" slow vector muls: " << elapsed.count() << " seconds\n";

    
    std::cout << res0/N/size<<" "<<(double)res1/N/size<<"\n";

}
