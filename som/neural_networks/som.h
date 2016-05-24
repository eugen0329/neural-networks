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

class SOM {
    typedef vector<Neuron> Layer;
    Layer outLayer;

    int inSize;
    vector<float> outp;

  public:
    SOM(int inSize, int clustersCount)
    {
        this->inSize = inSize;
        outLayer.resize(clustersCount);
        outp.resize(clustersCount);
        for (int neuron = 0; neuron < clustersCount; ++neuron) {
            outLayer[neuron].build(inSize, clustersCount);
        }

    }

    NeuroIO classify(Representation& input)
    {
        vector<float> inp = input.getImpl();
        vector<float> tmp(inp.size());

        return outp;
    }

    float train(Representation &r)
    {
        NeuroIO inp = r.getImpl();
        /* transform(begin(inp), end(inp), begin(inp), bind1st(divides<float>(), norm(inp))); */

        vector<float> deltas(outLayer.size());

        for(int i = 0; i < outLayer.size(); ++i) {
            Neuron& neuron = outLayer[i];
            deltas[i] = neuron.normalizedDelta(inp);
        }

        int winnerIndex = min_index(deltas);
        /* cout << winnerIndex << endl; */

        /* copy(network.out().begin(), network.out().end(), ostream_iterator<float>(cout, " ")); */
        /* cout << endl; */
        outLayer[winnerIndex].updateWeights(inp);

        for(int i = 0; i < outLayer.size(); ++i) {
            Neuron& neuron = outLayer[i];
            outp[i] = neuron.induce(inp);
        }
        return 0;
    }

    NeuroIO& out() { return outp; }
};

#endif /* end of include guard: COHEN_H_90MDQFWO */
