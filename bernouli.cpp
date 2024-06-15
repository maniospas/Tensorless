#include "tensorless/dense.h"
#include <cmath>
#include <chrono>



DATA predict(const std::vector<Dense*>& layers, DATA prediction) {
    for(int i=0;i<layers.size();++i)
        prediction = layers[i]->forward(prediction);
    return prediction;
}


void train(const std::vector<Dense*>& layers, 
           const std::vector<DATA>& inputs, 
           const std::vector<DATA>& outputs,
           int epochs=500,
           int llr=1  // learning rate = 0.5^llr
           ) {
    for(int epoch=0;epoch<epochs;++epoch) {
        for(int batch=0;batch<inputs.size();++batch) {
            int s = std::rand() % inputs.size();
            DATA err = predict(layers, inputs[s]) != outputs[s];
            err  *= rand(3);
            // backpropagate
            for(int i=layers.size()-1;i>=0;--i)
                err = layers[i]->backward(err, 3);
        }

        double total_err = 0;
        for(int sample=0;sample<inputs.size();++sample) {
            DATA prediction = predict(layers, inputs[sample]);
            DATA labels = outputs[sample];
            DATA err = prediction!=labels;
            total_err += err.countNonZeros();
        }
        total_err /= (double)inputs.size();
        std::cout<<"Epoch "<<epoch<<"/"<<epochs<<" errors: "<<total_err<<"\n";
        if(total_err<0.2)
            break;
    }
}


#define INS 8
#define HIDDEN 8
#define OUTS 8


int main() {
    int pos = 56;
    Double data1 = Double().set(pos, 0.5);
    Double data2 = Double().set(pos, 0.5);
    std::cout <<data1.get(pos)<<"\n";
    std::cout <<data2.get(pos)<<"\n";
    std::cout << "+="<<(data1+data2).get(pos)<<"\n";
    std::cout << "*="<<(data1*data2).get(pos)<<"\n";


    /*std::vector<Dense*> layers;
    layers.push_back(new Dense(INS, HIDDEN));
    layers.push_back(new Dense(HIDDEN, OUTS));

    std::vector<DATA> inputs;
    std::vector<DATA> outputs;
    for(int i=2;i<5;++i) {
        inputs.push_back(DATA(i));
        outputs.push_back(DATA(i/2-1));
    }

    for(auto layer : layers)
        std::cout<<layer;

    train(layers, inputs, outputs, 300);
    DATA out = predict(layers, DATA(4));
    std::cout << out.lower() << "\n";

    return 0;*/
}
