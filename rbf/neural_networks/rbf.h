#ifndef PERCEPTRON_H_ACXUHTU1
#define PERCEPTRON_H_ACXUHTU1

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <list>
#include <cmath>
#include <string>
#include <map>
#include <algorithm>
#include <iterator>

#include "representation.h"
#include "../util.h"
#include "neuron.h"
#include "example.h"
#include "cell.h"

using namespace std;
using namespace cv;
using namespace Neural;

#define OUTPUT_LAYER 1

typedef Mat Image;
typedef vector<Image> Images;
typedef vector<float> Weights;

class RBF
{
    typedef vector<Cell> RBFLayer;
    typedef vector<Neuron> Layer;
    typedef vector<vector<vector<float>>> RBFOutp;
    Layer outLayer;
    RBFLayer rbfLayer;
    RBFOutp  rbfOutp;

    int inSize;

  public:
    RBF(int inSize, int outSize, int hiddenSize)
    {
        this->inSize = inSize;
        outLayer.resize(outSize);
        int nLayerInputs = hiddenSize;
        for (int neuron = 0; neuron < outSize; ++neuron) {
            outLayer[neuron].build(nLayerInputs, outSize);
        }
    }

    NeuroIO classify(Representation& input)
    {
        directPass(input);
        NeuroIO klass(outLayer.size());
        for(int i = 0; i < klass.size(); ++i) {
            klass[i] = outLayer[i].getOut();
        }

        return klass;
    }

    void configureRBF(Examples &examples, vector<int> &classes)
    {
        rbfLayer.resize(examples.size());
        for(int i = 0; i < examples.size(); ++i) {
            rbfLayer[i].expectations = examples[i].in();
        }

    }

    float train(Representation &inp, NeuroIO expected)
    {
        directPass(inp);
        calcDeltas(expected);
        updateWeights(inp);
        return errRate(expected);
    }

    float errRate(NeuroIO &expected)
    {
        Layer &l = outLayer;
        float sum = 0.;
        for (int neuron = 0; neuron < l.size(); ++neuron) {
            sum += pow(expected[neuron] - l[neuron].getOut(), 2);
        }
        return 0.5 * sum;
    }

  private:

    void directPass(Representation &r)
    {
        NeuroIO rbfInp = r.getImpl();
        NeuroIO rbfOutp(rbfLayer.size());
        for(int neuron = 0; neuron < rbfLayer.size(); ++neuron) {
            rbfOutp[neuron] = rbfLayer[neuron].induce(rbfInp);
        }

        NeuroIO out(outLayer.size());
        for(int neuron = 0; neuron < outLayer.size(); ++neuron) {
            outLayer[neuron].induce(rbfOutp);
        }
    }

    void updateWeights(Representation &input)
    {
        for (int neuron = 0; neuron < outLayer.size(); ++neuron) {
            outLayer[neuron].updateWeight();
        }
    }

    //     δ_out = z - y
    void calcDeltas(NeuroIO &expected)
    {
        for (int neuron = 0; neuron < outLayer.size(); ++neuron) {
            float out = outLayer[neuron].getOut();
            /* outLayer[neuron].setDelta(expected[neuron] - out); */
            outLayer[neuron].setDelta(out * (1 - out) * (expected[neuron] - out));
        }
    }


  public:
    enum class WHAT : unsigned char { DELTAS, OUTPUTS, WEIGHTS, RBF };
    std::string inspectLayers(WHAT what = WHAT::OUTPUTS)
    {
        using namespace std;
        string dumpStr;
        /* for(int i = 0; i < outLayer.size(); ++i) { */
        /*     dumpStr += string("Layer ") + to_string(i) + ": "; */
        /*     dumpStr += inspectLayer(i, what) + "\n"; */
        /* } */
        return dumpStr;
    }

    std::string inspectOut(WHAT what = WHAT::OUTPUTS) { return inspectLayer(outLayer.size()-1, what); }
    std::string inspectLayer(int number, WHAT what = WHAT::OUTPUTS)
    {
        using namespace std;
        string dumpStr;
        for(int j = 0; j < outLayer.size(); ++j) {
            if(what == WHAT::OUTPUTS) {
                dumpStr += to_string(outLayer[j].getOut()) + " ";
            } else if(what == WHAT::DELTAS) {
                 dumpStr += to_string(outLayer[j].getDelta()) + " ";
            } else {
                vector<float> weights = outLayer[j].getWeights();
                for(int i = 0; i < weights.size(); ++i) {
                    dumpStr += to_string(weights[i]) + " ";
                    
                }
            }
        }
        return dumpStr;
    }
    NeuroIO out()
    {
        NeuroIO output(outLayer.size());
        for (int i = 0; i < outLayer.size(); ++i) {
            output[i] = outLayer[i].getOut();
        }
        return output;
    }
};

#endif /* end of include guard: PERCEPTRON_H_ACXUHTU1 */
