#ifndef NEURON_H_B8PUMZTW
#define NEURON_H_B8PUMZTW

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <functional>

#include "representation.h"
#include "../util.h"

using namespace std;
using namespace cv;
using namespace Neural;

#ifndef LEARNING_RATE
#define LEARNING_RATE 0.15
#endif

typedef vector<float> NeuroIO;

class Neuron {

public:
    float outp;
    NeuroIO inp;
    vector<float> weights;
    int winCount = 1;
    Neuron() {}

    void build(int inputsCount, int neuronsInLayer)
    {
        weights.resize(inputsCount + 1);
        for(int i = 0; i < weights.size(); ++i) {
            /* weights[i] = randInRange(-1.0/(2.0*neuronsInLayer), 1.0/(2.0*neuronsInLayer)); */
            weights[i] = randInRange(0., 1);
            /* weights[i] = randInRange(.0, 1.0); */
        }
    }

    void updateWeights(NeuroIO& neuronInp)
    {
        winCount++;
        vector<float> inps = neuronInp;

        vector<float> tmp(inps.size());
        transform(begin(inps), end(inps), begin(weights), begin(tmp), [](float i, float w) { return i-w; });
        transform(begin(inps), end(inps), begin(tmp), begin(tmp), [](float i, float t) { return i + LEARNING_RATE*t; });
        float div = euklidNorm(tmp);
        /* div = 1; */

        for(int i = 0; i < weights.size(); ++i) {
            weights[i] = (weights[i] + LEARNING_RATE*(inps[i] - weights[i])) / div;
        }
    }

    float sigmoid(float x)
    {
        return 1.0 / (1 + exp(-x));
    }

    float normalizedDelta(NeuroIO& neuronInp)
    {
        return delta(neuronInp) * winCount;
    }

    float delta(NeuroIO& neuronInp)
    {
        inp = neuronInp;
        float sum = 0;
        for (int i = 0; i < inp.size(); ++i) {
            sum += pow(inp[i] - weights[i], 2);
        }
        return sqrt(sum);
    }

    float induce(NeuroIO& neuronInp)
    {
        inp = neuronInp;
        float sum = 0.0;
        for(int i = 0; i < inp.size(); ++i) {
            sum += inp[i] * weights[i];
        }
        outp = sqrt(sum);

        return outp;
    }

    float getOut()
    {
        return outp;
    }

};

#endif /* end of include guard: NEURON_H_B8PUMZTW */
