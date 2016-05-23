#ifndef CELL_H_QXX5HNJA
#define CELL_H_QXX5HNJA

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
#define LEARNING_FACTOR 0.35
#define THRESHOLD 0.5
#endif

typedef vector<float> NeuroIO;

class Cell {
    float outp;
    NeuroIO inp;
    float delta;

    float gaussFunc(float x) { return exp( -x / squareDeviation ); }
public:
    Cell() {}

    Representation expectations;
    float deviation = 10;
    float squareDeviation = 10;

    float induce(NeuroIO& neuronInp)
    {
        inp = neuronInp;
        float sum = 0.0;
        for(int i = 0; i < inp.size(); ++i) {
            sum += pow(inp[i] - expectations[i], 2);
        }

        return (outp = gaussFunc(sum));
    }
};


#endif /* end of include guard: CELL_H_QXX5HNJA */
