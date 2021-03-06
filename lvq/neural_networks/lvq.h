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

class LVQNetwork {
    typedef vector<Neuron> Layer;
    Layer outLayer;

    int inSize;
    vector<float> outp;

  public:
    LVQNetwork(int inSize, int clustersCount)
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
        float norm = euklidNorm(inp);
        transform(begin(inp), end(inp), begin(inp), [&norm](const float& a) { return a / norm ; });

        for(int i = 0; i < outp.size(); ++i) {
            outp[i] = outLayer[i].induce(inp);
        }

        /* copy(outp.begin(), outp.end(), ostream_iterator<float>(cout, " ")); cout << endl; */
        return outp;
    }



    float train(Representation &r)
    {
        NeuroIO inp = r.getImpl();
        float norm = euklidNorm(inp);
        transform(begin(inp), end(inp), begin(inp), [&norm](const float& a) { return a / norm; });

        Neuron& winner = outLayer[winnerIndex(inp)];
        winner.updateWeights(inp);

        /* copy(winner.weights.begin(), winner.weights.end(), ostream_iterator<float>(cout, " ")); */
        /* cout << endl; */

        return winner.delta(inp);
    }

    int winnerIndex(NeuroIO& inp)
    {
        vector<float> deltas(outLayer.size());
        for(int i = 0; i < outLayer.size(); ++i) {
            deltas[i] = outLayer[i].normalizedDelta(inp);
        }

        return min_index(deltas);
    }



    NeuroIO& out() { return outp; }
};

#endif /* end of include guard: COHEN_H_90MDQFWO */
