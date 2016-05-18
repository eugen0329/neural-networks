#ifndef NEURON_H_B8PUMZTW
#define NEURON_H_B8PUMZTW

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <functional>

#include "representation.h"
#include "activation_functs.h"
#include "../util.h"

using namespace std;
using namespace cv;
using namespace Neural;

#ifndef LEARNING_FACTOR
#define LEARNING_FACTOR 0.5
#endif

typedef vector<float> NeuroIO;

class Neuron {
    float out;
    float delta;
    typedef function<float(float)> ActivationFunc;

    ActivationFunc activFunc;
    vector<float> inWeights;

public:
    Neuron() {}

    void build(int inputsCount, int neuronsInLayer, ActivationFunc activFunc = ActivationFuncs::sigmoid)
    {
        this->activFunc = activFunc;
        inWeights.resize(inputsCount);

        for(int i = 0; i < inWeights.size(); ++i)
            inWeights[i] = randInRange(-1/(2*neuronsInLayer), 1/(2*neuronsInLayer));
    }

    void updateWeight(Representation& input)
    {
        for(int i = 0; i < inWeights.size(); ++i) {
            inWeights[i] = LEARNING_FACTOR * delta * out * (1 - out) * input[i];
        }
    }

    float induce(NeuroIO& inputs)
    {
        float sum = 0;
        for(int i = 0; i < inputs.size(); ++i) {
            sum += inputs[i] * inWeights[i];
        }
        out = activFunc(sum);

        return out;
    }

    float atWeight(int i)
    {
        return inWeights[i];
    }

    float getOut()
    {
        return out;
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
