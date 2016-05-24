#ifndef COHEN_H_90MDQFWO
#define COHEN_H_90MDQFWO

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

using namespace std;
using namespace cv;
using namespace Neural;

#define OUTPUT_LAYER 1

typedef Mat Image;
typedef vector<Image> Images;
typedef vector<float> Weights;

class Cohen {
    typedef vector<Neuron> Layer;
    Layer outLayer;

    int inSize;
    vector<float> outp;

  public:
    Cohen(int inSize, int outSize)
    {
        this->inSize = inSize;
        outLayer.resize(outSize);

        for (int neuron = 0; neuron < outSize; ++neuron) {
            outLayer[neuron].build(inSize, outSize);
        }
    }

    NeuroIO classify(Representation& input)
    {
        vector<float> inp = input.getImpl();
        vector<float> tmp(inp.size());

        tmp = inp;
        transform(begin(tmp), end(tmp), begin(tmp), begin(tmp), multiplies<float>());
        float inpModule = sqrt(accumulate(begin(tmp), end(tmp), 0.0));

        for(int neurIter = 0; neurIter < outLayer.size(); ++neurIter) {
            outLayer[neurIter].weights;
            transform(begin(inp), end(inp), begin(outLayer[neurIter].weights), begin(tmp), multiplies<float>());
            float numerator = accumulate(begin(tmp), end(tmp), 0.0);

            tmp = outLayer[neurIter].weights;
            transform(begin(tmp), end(tmp), begin(tmp), begin(tmp), multiplies<float>());
            float weightsModule = sqrt(accumulate(begin(tmp), end(tmp), 0.0));

            outp[neurIter] = numerator / (inpModule * weightsModule);
            if(outp[neurIter] > 1.0) {
                cout << outp[neurIter];
                exit(1);
            }
        }

        return outp;
    }

    float train(Representation &r, NeuroIO expected)
    {
        NeuroIO inp = r.getImpl();
        transform(begin(inp), end(inp), begin(inp), bind1st(divides<float>(), norm(inp)));

        outp.resize(outLayer.size());
        vector<float> delta(inp.size());

        for(int neurIter = 0; neurIter < outLayer.size(); ++neurIter) {
            Neuron& neuron = outLayer[neurIter];
            // delta = (X - W)
            transform(begin(inp), end(inp), begin(neuron.weights), begin(delta), minus<float>());
            // delta *= delta
            transform(begin(delta), end(delta), begin(delta), begin(delta), multiplies<float>());
            // len = |delta|
            float module = sqrt(accumulate(begin(delta), end(delta), 0.0));

            /* cout << module; */
            outp[neurIter] = module * neuron.winCount;
            /* outp[neurIter] = module; */
            /* if(outp[neurIter] > 1.0) { */
            /*     cout << outp[neurIter]; */
            /*     exit(1); */
            /* } */
        }
        int winnerInd = min_index(outp);
        outLayer[winnerInd].updateWeights(inp);
        classify(r);

        return 0;
    }

  private:

    //     δ_out = z - y
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
                vector<float> weights = outLayer[j].weights;
                for(int i = 0; i < weights.size(); ++i) {
                    dumpStr += to_string(weights[i]) + " ";
                }
            }
        }
        return dumpStr;
    }
    NeuroIO& out()
    {
        /* NeuroIO output(outLayer.size()); */
        /* for (int i = 0; i < outLayer.size(); ++i) { */
        /*     output[i] = outLayer[i].getOut(); */
        /* } */
        return outp;
    }
};

#endif /* end of include guard: COHEN_H_90MDQFWO */
