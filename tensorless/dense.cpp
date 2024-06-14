/*
Copyright 2024 Emmanouil Krasanakis

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <iostream>
#include <vector>
#include <random>
#include <bitset>
#include <climits>
#include <cmath>
#include "tensorless/types/types.h"


int inline getBit(int a, int i) {
    return (a >> i) & 1;
}

int inline getBit(DATA a, int i) {
    return ((VECTOR)a >> i) & 1;
}

int countBits(VECTOR n) {
    VECTOR count = 0;
    while (n) {
        count += n & 1; // Add the least significant bit
        n >>= 1;        // Shift the number right by 1 bit
    }
    return count;
}

int inline countBits(DATA n) {
    return countBits((VECTOR)n);
}



class Dense {
private:
    int outs;
    int ins;
    int broadcast;
    std::vector<DATA> weights;
    std::vector<DATA> momentum;

protected:
    DATA in; // keeps track of last input
    DATA out; // keeps track of last output

public:
    Dense(int ins, int outs, int broadcast=1) : ins(ins), outs(outs), broadcast(broadcast) {
        for (int i=0; i<outs;++i) {
            weights.push_back(rand((int)std::log2(1+outs)));
            momentum.push_back(DATA(0));
        }
    }

    friend std::ostream& operator<<(std::ostream &os, const Dense *si) {
        os << "Bernouli layer";
        os << "\n  Inputs  " << si->ins;
        os << "\n  Outputs " << si->outs;
        os << "\n  Weights";
        for (int i=0;i<si->outs;++i) 
            os << "\n   >" << si->weights[i];
        os << "\n";
        return os;
    }

    DATA forward(DATA input) { // Running time: O(broadcast+outs)
        in = input; 
        for(int i=1;i<broadcast;++i) 
            in += input << (ins*i);

        out = DATA(0);
        for (int i=0;i<outs;++i) {
            DATA weightedIns = in*weights[i];
            if(weightedIns)  // if any bit different than zero
                out += DATA(1 << i);
        }
        return out;
    }

    DATA backward(DATA error, int llr=1) {
        DATA back = DATA(0);
        DATA notin = ~in; 
        DATA mask = rand(llr);
        DATA notmask = ~mask;
        for (int i=0;i<outs;++i) 
            if(error.isZeroAt(i)) {
                back += weights[i];
                weights[i] = (notin*mask) + (weights[i]*notmask);
            }
        return DATA(back);
    }
};