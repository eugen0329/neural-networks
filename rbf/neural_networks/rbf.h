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
    RBF(int inSize, int outSize)
    {
        this->inSize = inSize;
        outLayer.resize(outSize);
        int nLayerInputs = inSize;
        for (int neuron = 0; neuron < outSize; ++neuron) {

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
    void configureRBF(Examples &examples, vector<int> &classes)
    {
        Examples clustImgs;
        rbfLayer.resize(classes.size());
        int inputsSize = examples.front().in().size();
        int clustCount = classes.size();

        for (int clustIter = 0; clustIter < clustCount; ++clustIter) {
            int klass = classes[clustIter];

            // combine examples which is included in {klass} cluster BEGIN
            clustImgs.resize(examples.size());
            Examples::iterator beg = examples.begin(), end = examples.end(),
                               dst = clustImgs.begin();
            auto block = [&klass](Example &e) { return klass == e.getTag(); };
            Examples::iterator it = copy_if(beg, end, dst, block);
            clustImgs.resize(distance(clustImgs.begin(), it));
            // END

            for(int inpIter = 0; inpIter < inputsSize; ++inpIter) {
                for(int imgIter = 0; imgIter < clustImgs.size(); ++imgIter) {
                    rbfLayer[clustIter].expectations[inpIter] = clustImgs[imgIter][inpIter];
                }
                rbfLayer[clustIter].expectations[inpIter] /= clustImgs.size();
            }
        }

        // One way to calculate the spread parameter (deviation) is:
        //   deviation = Dmax / sqrt(N)
        // Where:
        //   N: number of centers, Dmax: distance between them
        list<float> centerDistances;
        combinations<Cell>(rbfLayer, 2, [&](std::list<Cell> comb) {
            /* vector<float> sum(inputsSize); */
            float sum = 0;
            Representation& c1 = comb.back().expectations;
            Representation& c2 = comb.front().expectations;
            /* Representation& c1 = rbfLayer[comb.back()].expectations; */
            /* Representation& c2 = rbfLayer[comb.front()].expectations; */
            for(int i = 0; i < inputsSize; ++i) {
                sum += pow(c1[i] - c2[i], 2);
            }
            centerDistances.push_back(sqrt(sum));
        });
        auto floatCmp = [](float a, float b) { return (a < b) ? -1 : (a > b); };
        float maxDistance = *max_element(begin(centerDistances), end(centerDistances), floatCmp);
        float squareDeviation = pow(maxDistance, 2) / clustCount;
        float deviation = maxDistance / sqrt((float) clustCount);
        for(int i = 0; i < rbfLayer.size(); ++i) {
            rbfLayer[i].squareDeviation = squareDeviation;
            rbfLayer[i].deviation       = deviation;
        }
    }

    /* void configureRBF(Examples &examples, vector<int> &classes) */
    /* { */
    /*     Examples classImages; */
    /*     // train */
    /*     int rbfLayerSize = examples.size(); */
    /*     rbfLayer.resize(rbfLayerSize); */
    /*     for (int i = 0; i < examples.size(); ++i) { */
    /*         rbfLayer[i].expectations = examples[i].in(); */
    /*     } */
    /*     vector<list<float>> deviations(rbfLayer.size()); */
    /*     for(int i = 0; i < rbfLayerSize; ++i) { */
    /*         for(int j = 0; j < rbfLayerSize; ++j) { */
    /*             float deviation = 0; */
    /*             for (int k = 0; k < inSize; ++k) { */
    /*                 deviation += pow(rbfLayer[i].expectations[k] - rbfLayer[j].expectations[k], 2); */
    /*             } */
    /*             deviation = sqrt(deviation); */
    /*             deviations[i].push_back(deviation); */
    /*             deviations[j].push_back(deviation); */
    /*         } */
    /*     } */
    /*     for (int i = 0; i < rbfLayer.size(); ++i) { */
    /*         rbfLayer[i].deviation = *min_element(begin(deviations[i]), end(deviations[i])); */
    /*     } */

    /*     // computing rbf layer outputs */
    /*     for (int classIter = 0; classIter < classes.size(); ++classIter) { */
    /*         int klass = classes[classIter]; */

    /*         // combine examples by class, mov class referred examples vector to classImages */
    /*         classImages.resize(examples.size()); */
    /*         Examples::iterator beg = examples.begin(), end = examples.end(), */
    /*                            to = classImages.begin(); */
    /*         auto block = [&klass](Example &e) { return klass == e.getTag(); }; */
    /*         Examples::iterator it = copy_if(beg, end, to, block); */
    /*         classImages.resize(distance(classImages.begin(), it)); */


    /*         for(int imgIter = 0; imgIter < classImages.size(); ++imgIter) { */
    /*             Representation& image = classImages[imgIter].in(); */
    /*             for(int neurIter = 0; neurIter < rbfLayer.size(); ++neurIter) { */
    /*                 float sum = 0; */
    /*                 for(int inpIter = 0; inpIter < image.size(); ++inpIter) { */
    /*                     sum += pow(image[inpIter] - rbfLayer[neurIter].expectations[inpIter], 2); */
    /*                 } */
    /*                 rbfOutp[classIter][imgIter][neurIter] = exp(-sum / pow(rbfLayer[neurIter].deviation, 2)); */
    /*             } */
    /*         } */
    /*     } */
    /* } */
    /* vector<vector<float>> expectations(tags.size()); */
    /* vector<float> deviations(tags.size()); */
    /* Examples classImages; */

    /* for (int i = 0; i < tags.size(); ++i) { */
    /*     int tag = tags[i]; */
    /*     classImages.resize(examples.size()); */
    /*     Examples::iterator beg = examples.begin(), end = examples.end(), */
    /*                        to = classImages.begin(); */
    /*     auto block = [&tag](Example &e) { return tag == e.getTag(); }; */
    /*     Examples::iterator it = copy_if(beg, end, to, block); */
    /*     classImages.resize(std::distance(classImages.begin(), it)); */


    /*     for(int j = 0; j < classImages.size(); ++j) { */
    /*         vector<float>& vec = classImages[j].in().getImpl(); */
    /*         expectations[tag].resize(vec.size()); */
    /*         for(int i = 0; i < vec.size(); ++i) { */
    /*             expectations[tag][j] += vec[j] / classImages.size(); */
    /*         } */
    /*     } */
    /* } */

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
        NeuroIO layerOutput;
        NeuroIO layerInput(r.size());

        for (int i = 0; i < r.size(); ++i) {
            layerInput[i] = r[i];
        }

        layerOutput.resize(outLayer.size());
        for (int neuron = 0; neuron < outLayer.size(); ++neuron) {
            layerOutput[neuron] = outLayer[neuron].induce(layerInput);
        }
        layerInput = layerOutput;
    }

    void updateWeights(Representation &input)
    {
        Representation layerInput = input;
        for (int neuron = 0; neuron < outLayer.size(); ++neuron) {
            outLayer[neuron].updateWeight();
        }
    }

    void calcDeltas(NeuroIO &expected)
    {
        calcOutDelta(outLayer, expected);
        /* if (outLayer.size() > 1) { */
        /*     calcHiddenDelta((outLayer[outLayer.size()-2]), outLayer.back());
         */
        /* } */
    }

    //     δh = ∑ δ_out⋅f'⋅whi
    void calcHiddenDelta(Layer &layer, Layer &prevLayer)
    {
        for (int i = 0; i < layer.size(); ++i) {
            float error = 0;
            for (int prevN = 0; prevN < prevLayer.size(); ++prevN) {
                error +=
                    prevLayer[prevN].getDelta() * prevLayer[prevN].atWeight(i);
            }
            float out = layer[i].getOut();
            layer[i].setDelta(out * (1 - out) * error);
        }
    }

    //     δ_out = z - y
    void calcOutDelta(Layer &layer, NeuroIO &expected)
    {
        for (int neuron = 0; neuron < layer.size(); ++neuron) {
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

    /* std::string inspectOut(WHAT what = WHAT::OUTPUTS) { return
     * inspectLayer(outLayer.size()-1, what); } */
    /* std::string inspectLayer(int number, WHAT what = WHAT::OUTPUTS) */
    /* { */
    /*     using namespace std; */
    /*     string dumpStr; */
    /*     for(int j = 0; j < outLayer[number].size(); ++j) { */
    /*         if(what == WHAT::OUTPUTS) { */
    /*             dumpStr += to_string(outLayer[number][j].getOut()) + " "; */
    /*         } else { */
    /*             dumpStr += to_string(outLayer[number][j].getDelta()) + " ";
     */
    /*         } */
    /*     } */
    /*     return dumpStr; */
    /* } */
    NeuroIO out()
    {
        Layer &last = outLayer;
        NeuroIO output(last.size());
        for (int i = 0; i < last.size(); ++i) {
            output[i] = last[i].getOut();
        }
        return output;
    }
};

#endif /* end of include guard: PERCEPTRON_H_ACXUHTU1 */
