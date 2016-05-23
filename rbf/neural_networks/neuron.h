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
    float outp;
    NeuroIO inp;
    float delta;

    vector<float> weights;

public:
    Neuron() {}

    void build(int inputsCount, int neuronsInLayer)
    {
        weights.resize(inputsCount + 1);
        for(int i = 0; i < weights.size(); ++i) {
            weights[i] = randInRange(-1.0/(2.0*neuronsInLayer), 1.0/(2.0*neuronsInLayer));
        }
    }

    void updateWeight()
    {
        Representation inputs = inp;
        /* inputs.push_back(-1); */
        for(int i = 0; i < weights.size(); ++i) {
            weights[i] += LEARNING_FACTOR *  delta * inputs[i];
        }
    }

    float sigmoid(float x)
    {
        return 1.0 / (1 + exp(-x));
    }

    float induce(NeuroIO& neuronInp)
    {
        inp = neuronInp;
        inp.push_back(-1);
        float sum = 0.0;
        for(int i = 0; i < inp.size(); ++i) {
            sum += inp[i] * weights[i];
        }
        outp = sigmoid(sum);
        return outp;
    }

    vector<float>& getWeights()
    {
        return weights;
    }

    float atWeight(int i)
    {
        return weights[i];
    }

    float getOut()
    {
        return outp;
    }

    float getIn()
    {
        return outp;
    }

    float setDelta(float newDelta)
    {
        delta = newDelta;
        return delta;
    }

    float getDelta()
    {
        return delta;
    }

};

#endif /* end of include guard: NEURON_H_B8PUMZTW */
