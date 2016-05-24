#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <map>

#include <algorithm>
#include <clocale>
#include <dirent.h>

#include <cmath>

#include "neural_networks/representation.h"
#include "neural_networks/perceptron.h"

#include "neural_networks/activation_functs.h"
#include "neural_networks/example.h"
#include "csv_iterator.h"
#include "util.h"

using namespace std;
using namespace cv;
using namespace Neural;

typedef vector<Example> Examples;

#define AAA 3000

void iris();
void images();

int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");
    images();
    /* iris(); */
    return 0;
}

void images()
{
    Examples examples = {{Example(imread("examples/2.1.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "2", 0, 5)},
                         {Example(imread("examples/2.2.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "2", 0, 5)},
                         {Example(imread("examples/3.1.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "3", 1, 5)},
                         {Example(imread("examples/3.2.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "3", 1, 5)},
                         {Example(imread("examples/4.1.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "4", 2, 5)},
                         {Example(imread("examples/4.2.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "4", 2, 5)},
                         {Example(imread("examples/5.1.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "5", 3, 5)},
                         {Example(imread("examples/5.2.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "6", 3, 5)},
                         {Example(imread("examples/7.1.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "7", 4, 5)},
                         {Example(imread("examples/7.2.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "7", 4, 5)}};

    Perceptron network(examples[0].in().size(), examples[0].out().size(), {100});

    int errs = 0;
    do {
        errs = 0;
        for(Examples::iterator e = examples.begin(); e != examples.end(); ++e) {
            float err = network.train(e->in(), e->out());
            if(max_index(e->out()) != max_index(network.out())) errs++;
        }
        cout << errs * 1./ examples.size() << "\r";
    } while((errs * 1. / examples.size()) > 0.01);


    vector<float> noiseLevels = {0, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90};
    for(int i = 0; i < noiseLevels.size(); ++i) {
        errs = 0;
        cout << "Noise level: " << noiseLevels[i] << "%" << endl;
        for(Examples::iterator e = examples.begin(); e != examples.end(); ++e) {
            Representation in = e->in();
            NeuroIO result = network.classify(in.apply_noise(noiseLevels[i]));

            if(max_index(e->out()) == max_index(network.out())) {
                cout << "\033[32m✔\033[0m " << max_index(e->out()) << " == " << max_index(network.out()) << endl;
            } else {
                errs++;
                cout << "\033[31m✘\033[0m " << max_index(e->out()) << " != " << max_index(network.out()) << endl;

            }
        }
        cout << "Errors: " << errs * 1. / examples.size() << endl << endl;
    }
}

void iris()
{
    map<string, int> tagsMap = {{"Iris-setosa", 0}, {"Iris-versicolor", 1}, {"Iris-virginica", 2}};
    std::ifstream       file("iris.csv");
    Examples examples;
    for(CSVIterator r(file);r != CSVIterator();++r) {
        std::vector<float> features = {stof((*r)[0]), stof((*r)[1]), stof((*r)[2]), stof((*r)[3])};
        examples.push_back(Example(features, (*r)[4], tagsMap[(*r)[4]], tagsMap.size()));
    }

    /* random_shuffle(examples.begin(), examples.end()); */

    Perceptron perceptron(4, 3, {4});
    int errs = 0;
    do {
        errs = 0;
        for(Examples::iterator it = examples.begin(); it != examples.end(); ++it) {
            float err = perceptron.train(it->in(), it->out());
            if(max_index(it->out()) != max_index(perceptron.out())) errs++;
        }
        cout << errs / 100.0 << "\r";
    } while((errs / 100.0) > 0.05);
    errs = 0;
    for(Examples::iterator it = examples.begin(); it != examples.end(); ++it) {
        NeuroIO result = perceptron.classify(it->in());
        cout << max_index(it->out()) << max_index(result) << endl;
        if(max_index(it->out()) != max_index(perceptron.out())) errs++;
    }
    cout << errs / 100.0;
    cout << endl;
}
