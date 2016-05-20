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
        int nLayerInputs = inSize;
        layers.resize(hidden.size() + OUTPUT_LAYER);

        for(int i = 0; i < hidden.size(); ++i) {
            layers[i].resize(hidden[i]);
            for(int neuron = 0; neuron < hidden[i]; ++neuron) {
                layers[i][neuron].build(nLayerInputs, hidden[i]);
            }
           nLayerInputs = layers[i].size();
        }

        nLayerInputs = layers.empty() ? inSize : (layers[layers.size() - 2]).size();
        layers.back().resize(outSize);
        for(int neuron = 0; neuron < outSize; ++neuron) {
            layers.back()[neuron].build(nLayerInputs, outSize);
        }
    }

    NeuroIO classify(Representation& input)
    {
        directPass(input);
        Layer& outLayer = layers[layers.size()-1];
        NeuroIO klass(outLayer.size());
        for(int i = 0; i < klass.size(); ++i) {
            klass[i] = outLayer[i].getOut();
        }

        return klass;
    }

    /* printf("%f %f %f\n", inp[0], inp[1], inp[2]); */
    /* printf("%f %f %f\n", layers[0][0].getOut(), layers[0][1].getOut(), layers[0][2].getOut()); */
    float train(Representation& inp, NeuroIO expected)
    {
        directPass(inp);
        calcDeltas(expected);
        updateWeights(inp);
        return errRate(expected);
    }

    float errRate(NeuroIO& expected)
    {
        Layer& l = layers[layers.size()-1];
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

        for(int l = 0; l < layers.size(); ++l) {
            layerOutput.resize(layers[l].size());
            for(int neuron = 0; neuron < layers[l].size(); ++neuron) {
                layerOutput[neuron] = layers[l][neuron].induce(layerInput);
            }
            layerInput = layerOutput;
        }
    }

    void updateWeights(Representation& input)
    {
        Representation layerInput = input;
        for(int l = layers.size() - 1; l > 0; --l) {
            for(int neuron = 0; neuron < layers[l].size() ; ++neuron) {
                layers[l][neuron].updateWeight();
            }
        }
    }

    void calcDeltas(NeuroIO& expected)
    {
        calcOutDelta(layers.back(), expected);
        if (layers.size() > 1) {
            calcHiddenDelta((layers[layers.size()-2]), layers.back());
        }
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
    enum class WHAT : unsigned char { DELTAS, OUTPUTS };
    std::string inspectLayers(WHAT what = WHAT::OUTPUTS)
    {
        using namespace std;
        string dumpStr;
        for(int i = 0; i < layers.size(); ++i) {
            dumpStr += string("Layer ") + to_string(i) + ": ";
            dumpStr += inspectLayer(i, what) + "\n";
        }
        return dumpStr;
    }

    std::string inspectOut(WHAT what = WHAT::OUTPUTS) { return  inspectLayer(layers.size()-1, what); }
    std::string inspectLayer(int number, WHAT what = WHAT::OUTPUTS)
    {
        using namespace std;
        string dumpStr;
        for(int j = 0; j < layers[number].size(); ++j) {
            if(what == WHAT::OUTPUTS) {
                dumpStr += to_string(layers[number][j].getOut()) + " ";
            } else {
                dumpStr += to_string(layers[number][j].getDelta()) + " ";
            }
        }
        return dumpStr;
    }
    NeuroIO out()
    {
        Layer& last = layers.back();
        NeuroIO output(last.size());
        for(int i = 0; i < last.size(); ++i) {
            output[i] = last[i].getOut();
        }
        return output;
    }
};


#endif /* end of include guard: PERCEPTRON_H_ACXUHTU1 */
