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
#include "util.h"


using namespace std;
using namespace cv;
using namespace Neural;

typedef vector<vector<int>> Image;
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

int main(int argc, char *argv[])
{
    Weights weights;
    Representation r;
    Neural::Representations representations(images.size());
    Images images = {
        imread("examples/2.1.png", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/2.2.png", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/3.1.png", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/3.2.png", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/4.1.png", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/4.2.png", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/5.1.png", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/5.2.png", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/7.1.png", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/7.2.png", CV_LOAD_IMAGE_GRAYSCALE),
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

