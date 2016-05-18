#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>

#include <clocale>
#include <dirent.h>

#include <cmath>

#include "neural_networks/representation.h"
#include "neural_networks/perceptron.h"

#include "neural_networks/activation_functs.h"
#include "util.h"


using namespace std;
using namespace cv;
using namespace Neural;


int main(int argc, char *argv[])
{
    Weights weights;
    Representation r;
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

    int inSize = representations[0].size(); // input vector sizes
    #define CLASSES_COUNT 5
    int outSize = CLASSES_COUNT;
    int hiddenSize = hiddenLayerSize(inSize, outSize, images.size());
    Perceptron perceptron(inSize, outSize, {hiddenSize});
    float err = perceptron.train(representations[0], {1, 0, 0, 0, 0});
    printf("%f\n", err);

    return 0;
}

