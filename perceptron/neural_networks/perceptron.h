#ifndef PERCEPTRON_H_ACXUHTU1
#define PERCEPTRON_H_ACXUHTU1


#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <cmath>

#include "representation.h"
#include "neuron.h"

using namespace std;
using namespace cv;
using namespace Neural;

#define OUTPUT_LAYER 1

typedef Mat Image;
typedef vector<Image> Images;
typedef vector<float> Weights;
/* typedef vector<Layer> Perceptron; */

class Perceptron {
    typedef vector<Neuron> Layer;
    typedef vector<Layer> Layers;
    Layers layers;
public:

    //
    // w(t + 1) = w(t) + η⋅δ⋅(df(e)/de)*y
    // df(e)/de = f(e)⋅(1 - f(e))
    //
    // Err:
    //   Output layer(z - expected, y - received):
    //     δ_out = z - y
    //   hidden layer:
    //     δh = ∑ δ_out⋅f'⋅whi

    Perceptron(int inSize, int outSize, vector<int>&& hidden = {})
    {
        if(!hidden.empty()) {
            int nInputs = inSize;
            layers.resize(hidden.size() + OUTPUT_LAYER);
            for(int i = 0; i < hidden.size(); ++i) {
                layers[i].resize(hidden[i]);
                for(int neuron = 0; neuron < hidden[i]; ++neuron) {
                    layers[i][neuron].build(nInputs, hidden[i]);
                }
               nInputs = layers[i].size();
            }
        } else {
            layers.resize(1);
        }

        Layer& outLayer = layers[layers.size() - 1];
        int nInputs = layers.empty() ? inSize : (layers[layers.size() - 2]).size();
        outLayer.resize(outSize);
        for(int neuron = 0; neuron < outSize; ++neuron) {
            outLayer[neuron].build(nInputs, outSize);
        }
    }
    float train(Representation& input, NeuroIO&& expected)
    {
        directPass(input);
        calcDeltas(expected);
        updateWeights(input);
        return errRate(expected);
    }

    float errRate(NeuroIO& expected)
    {
        Layer& l = layers[layers.size()];
        float sum = 0.;
        for(int neuron = 0; neuron < l.size(); ++neuron) {
            sum += pow(expected[neuron] - l[neuron].getOut(), 2);
        }
        return 0.5 * sum;
    }

private:
    void updateWeights(Representation& input)
    {
        for(int l = 0; l < layers.size(); ++l) {
            for(int neuron = 0; neuron < layers[l].size(); ++neuron) {
                layers[l][neuron].updateWeight(input);
            }
        }
    }

    void directPass(Representation& r)
    {
        NeuroIO layerOutput;
        NeuroIO layerInput(r.size());

        for(int i = 0; i < r.size(); ++i)
            layerInput[i] = r[i];

        for(int l = 0; l < layers.size(); ++l) {
            layerOutput.resize(layers[l].size());
            for(int neuron = 0; neuron < layers[l].size(); ++neuron) {
                layerOutput[neuron] = layers[l][neuron].induce(layerInput);
            }
            layerInput = layerOutput;
        }
    }

    void calcDeltas(NeuroIO& expected)
    {
        calcOutDelta(*layers.end(), expected);
        calcHiddenDelta(*(layers.end() - 1), *layers.end());
    }

    //     δh = ∑ δ_out⋅f'⋅whi
    void calcHiddenDelta(Layer& layer, Layer& prevLayer)
    {
        for(int i = 0; i < layer.size(); ++i) {
            float error = 0;
            for(int prevN = 0; prevN < prevLayer.size(); ++prevN) {
                error += prevLayer[prevN].getDelta() * prevLayer[prevN].atWeight(i);
            }
            float out = layer[i].getOut();
            layer[i].setDelta(out * (1 - out) * error);
        }
    }

    /* layer.each_with_index do |neuron, neuron_index| */
    /*   error = 0 */
    /*   @network.last.each do |output_neuron| */
    /*     error += output_neuron.delta * output_neuron.weights[neuron_index] */
    /*   end */
    /*   output = neuron.last_output */
    /*   neuron.delta = output * (1 - output) * error */
    /* end */
        //     δ_out = z - y
    void calcOutDelta(Layer& layer, NeuroIO& expected)
    {
        for(int neuron = 0; neuron < layer.size(); ++neuron) {
            float out = layer[neuron].getOut();
            layer[neuron].setDelta(expected[neuron] - out);
        }
    }
    /* static void shiftForward(Perceptron& p, Representation& r, NeuroIO& outputs) */
    /* { */
    /*     NeuroIO layerOutput; */

    /*     NeuroIO layerInput(r.size()); */
    /*     for(int i = 0; i < r.size(); ++i) */
    /*         layerInput[i] = r[i]; */


    /*     for(int l = 0; l < p.size(); ++l) { */
    /*         layerOutput.resize(p[l].size()); */
    /*         for(int n = 0; n < p[l].size(); ++n) { */
    /*             layerOutput[n] = out(); */
    /*         } */
    /*     } */
    /* } */
};

/* void build(Perceptron& p, vector<int> sizes) */
/* { */
/*     /1* int nLayers = sizes.size(); *1/ */
/*     /1* p.resize(nLayers); *1/ */
/*     /1* for(int layer = 0; layer < nLayers; ++layer) { *1/ */
/*     /1*     p[layer].resize(sizes[layer]); *1/ */
/*     /1* } *1/ */
/* } */

/* float train(Perceptron& p, Representation& img, int outSize) */
/* { */
/*     /1* NeuroIO outs(outSize); *1/ */
/*     /1* shiftForward(p, outs); *1/ */
/*     return 0.; */
/* } */


/* /1* float out() *1/ */

/* void calcDeltas(Perceptron& p) */
/* { */
/* } */


#endif /* end of include guard: PERCEPTRON_H_ACXUHTU1 */
