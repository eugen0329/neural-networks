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

#ifndef LEARNING_FACTOR
#define LEARNING_FACTOR 0.15
#define THRESHOLD 0.5
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
            weights[i] = randInRange(0, 1.0/(2.0*neuronsInLayer));
        }
    }

    void updateWeights(NeuroIO& neuronInp)
    {
        vector<float> inps = neuronInp;
        vector<float> prevWeights = weights;

        for(int i = 0; i < weights.size(); ++i) {
            weights[i] = prevWeights[i] + LEARNING_FACTOR*(inps[i] - prevWeights[i]);
        }
        winCount += 1;
    }

    float sigmoid(float x)
    {
        return 1.0 / (1 + exp(-x));
    }

    float normalizedDelta(NeuroIO& neuronInp)
    {
        inp = neuronInp;
        float sum = 0;
        for (int i = 0; i < inp.size(); ++i) {
            sum += pow(inp[i] - weights[i], 2);
        }
        return sqrt(sum) * winCount;
    }

    float induce(NeuroIO& neuronInp)
    {
        inp = neuronInp;
        float sum = 0.0;
        for(int i = 0; i < inp.size(); ++i) {
            sum += inp[i] * weights[i];
        }
        outp = sum;

        return outp;
    }

    float getOut()
    {
        return outp;
    }

};

#endif /* end of include guard: NEURON_H_B8PUMZTW */
