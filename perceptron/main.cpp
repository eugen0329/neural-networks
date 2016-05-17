#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>

#include <clocale>
#include <dirent.h>

#include <cmath>

#include "../hopfield/neural_networks.h"

#include "activation_functs.h"
#include "../hopfield/util.h"


using namespace std;
using namespace cv;
using namespace Neural;

typedef Mat Image;
typedef vector<Image> Images;
typedef vector<float> Weights;
typedef vector<Weights> Layer;
typedef vector<Layer> Perceptron;

void build(Perceptron& p, vector<int> layersSizes)
{
    int nLayers = layersSizes.size();
    p.resize(nLayers);
    for(int layer = 0; layer < nLayers; ++layer) {
        p[layer].resize(layersSizes[layer]);
    }
}

int hiddenLayerSize(int inSize, int outSize, int examplesCount)
{
    // http://stackoverflow.com/questions/10565868/multi-layer-perceptron-mlp-architecture-criteria-for-choosing-number-of-hidde
    //return std::ceil((inSize + outSize) * 2.0 / 3);
    // OR
    // h = sqrt(p/n)
    // n: inputs, m: outputs, h: hidden, p:examples count
    return std::ceil(std::sqrt(TO_F(examplesCount) / inSize));
}

int main(int argc, char *argv[])
{
    Weights weights;
    Representation r;

    Images images = {
        imread("../hopfield/examples/a.jpg", CV_LOAD_IMAGE_GRAYSCALE),
        imread("../hopfield/examples/i.jpg", CV_LOAD_IMAGE_GRAYSCALE),
        imread("../hopfield/examples/p.jpg", CV_LOAD_IMAGE_GRAYSCALE)
    };
    Neural::Representations representations(images.size());
    for(int i = 0; i < images.size(); ++i)
        img2representation(images[i], representations[i]);

    Perceptron p;
    int inSize = representations[0].size(); // input vector sizes
    #define CLASSES_COUNT 3
    int outSize = CLASSES_COUNT;
    int hiddenSize = hiddenLayerSize(inSize, outSize, images.size());
    build(p, {inSize, hiddenSize, outSize});

    return 0;
}

