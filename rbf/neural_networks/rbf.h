#ifndef PERCEPTRON_H_ACXUHTU1
#define PERCEPTRON_H_ACXUHTU1

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

#include "representation.h"
#include "../util.h"
#include "neuron.h"

using namespace std;
using namespace cv;
using namespace Neural;

#define OUTPUT_LAYER 1

typedef Mat Image;
typedef vector<Image> Images;
typedef vector<float> Weights;

class RBF {
    typedef vector<Neuron> Layer;
    typedef vector<Layer> Layers;
    Layer outLayer;
public:

    RBF(int inSize, int outSize, vector<int>&& hidden = {})
    {
        outLayer.resize(outSize);
        int nLayerInputs = inSize;
        for(int neuron = 0; neuron < outSize; ++neuron) {

            outLayer[neuron].build(nLayerInputs, outSize);
        }
    }

    /* NeuroIO classify(Representation& input) */
    /* { */
    /*     /1* directPass(input); *1/ */
    /*     /1* Layer& outLayer = outLayer[outLayer.size()-1]; *1/ */
    /*     /1* NeuroIO klass(outLayer.size()); *1/ */
    /*     /1* for(int i = 0; i < klass.size(); ++i) { *1/ */
    /*     /1*     klass[i] = outLayer[i].getOut(); *1/ */
    /*     /1* } *1/ */

    /*     /1* return klass; *1/ */
    /* } */

    float train(Representation& inp, NeuroIO expected)
    {
        directPass(inp);
        calcDeltas(expected);
        updateWeights(inp);
        return errRate(expected);
    }

    float errRate(NeuroIO& expected)
    {
        Layer& l = outLayer;
        float sum = 0.;
        for(int neuron = 0; neuron < l.size(); ++neuron) {
            sum += pow(expected[neuron] - l[neuron].getOut(), 2);
        }
        return 0.5 * sum;
    }

private:

    void directPass(Representation& r)
    {
        NeuroIO layerOutput;
        NeuroIO layerInput(r.size());

        for(int i = 0; i < r.size(); ++i) {
            layerInput[i] = r[i];
        }

        layerOutput.resize(outLayer.size());
        for(int neuron = 0; neuron < outLayer.size(); ++neuron) {
            layerOutput[neuron] = outLayer[neuron].induce(layerInput);
        }
        layerInput = layerOutput;
    }

    void updateWeights(Representation& input)
    {
        Representation layerInput = input;
        for(int neuron = 0; neuron < outLayer.size() ; ++neuron) {
            outLayer[neuron].updateWeight();
        }
    }

    void calcDeltas(NeuroIO& expected)
    {
        calcOutDelta(outLayer, expected);
        /* if (outLayer.size() > 1) { */
        /*     calcHiddenDelta((outLayer[outLayer.size()-2]), outLayer.back()); */
        /* } */
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

    //     δ_out = z - y
    void calcOutDelta(Layer& layer, NeuroIO& expected)
    {
        for(int neuron = 0; neuron < layer.size(); ++neuron) {
            float out = layer[neuron].getOut();
            layer[neuron].setDelta(out * (1 - out) * (expected[neuron] - out));
            layer[neuron].setDelta(expected[neuron] - out);
        }
    }
public:
    /* enum class WHAT : unsigned char { DELTAS, OUTPUTS }; */
    /* std::string inspectLayers(WHAT what = WHAT::OUTPUTS) */
    /* { */
    /*     using namespace std; */
    /*     string dumpStr; */
    /*     for(int i = 0; i < outLayer.size(); ++i) { */
    /*         dumpStr += string("Layer ") + to_string(i) + ": "; */
    /*         dumpStr += inspectLayer(i, what) + "\n"; */
    /*     } */
    /*     return dumpStr; */
    /* } */

    /* std::string inspectOut(WHAT what = WHAT::OUTPUTS) { return  inspectLayer(outLayer.size()-1, what); } */
    /* std::string inspectLayer(int number, WHAT what = WHAT::OUTPUTS) */
    /* { */
    /*     using namespace std; */
    /*     string dumpStr; */
    /*     for(int j = 0; j < outLayer[number].size(); ++j) { */
    /*         if(what == WHAT::OUTPUTS) { */
    /*             dumpStr += to_string(outLayer[number][j].getOut()) + " "; */
    /*         } else { */
    /*             dumpStr += to_string(outLayer[number][j].getDelta()) + " "; */
    /*         } */
    /*     } */
    /*     return dumpStr; */
    /* } */
    NeuroIO out()
    {
        Layer& last = outLayer;
        NeuroIO output(last.size());
        for(int i = 0; i < last.size(); ++i) {
            output[i] = last[i].getOut();
        }
        return output;
    }
};


#endif /* end of include guard: PERCEPTRON_H_ACXUHTU1 */
