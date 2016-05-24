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

#include "neural_networks/example.h"
#include "csv_iterator.h"
#include "util.h"
#include "neural_networks/som.h"

using namespace std;
using namespace cv;
using namespace Neural;

void iris();
void images();
void readIrisDB(Examples& examples, string path);

int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");
    images();
    /* iris(); */
    return 0;
}

map<string, int> tagsMap = {{"Iris-setosa", 0}, {"Iris-versicolor", 1}, {"Iris-virginica", 2}};
void iris()
{
    vector<int> tags;
    values(tagsMap, tags);

    Examples examples;
    readIrisDB(examples, "iris.csv");
    /* random_shuffle(examples.begin(), examples.end()); */

    /* transform(begin(examples), end(examles), begin(examples), [&examples]()) */

    int errs = 0;
    float avgDelta = 0.;
    float maxDelta = 0.05;
    int clustersCount = tagsMap.size();
    int trainIterations = 0;
    SOM network(4, clustersCount);
    do {
        avgDelta = 0.;
        for(Examples::iterator e = examples.begin(); e != examples.end(); ++e) {
            float delta = network.train(e->in());
            avgDelta += delta;
            trainIterations++;
        }
        avgDelta /= examples.size();
        cout << "\r" << avgDelta;
    } while(avgDelta > 0.14);

    cout << endl << "Trained in " << trainIterations << " iterations" << endl;

    cout << "Expected | Got" << endl;
    for(Examples::iterator e = examples.begin(); e != examples.end(); ++e) {
        NeuroIO result = network.classify(e->in());

        int expected = max_index(e->out()), got = max_index(network.out());
        if(expected == got) {
            cout << "\033[32m✔\033[0m " << expected << " == " << got << endl;
        } else {
            errs++;
            cout << "\033[31m✘\033[0m " << expected << " != " << got << endl;
        }
    }
    cout << "Errors: " << errs * 1. / examples.size() << endl << endl;
    cout << endl;
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
    int errs = 0;
    float avgDelta = 0.;
    float maxDelta = 0.05;
    int clustersCount = tagsMap.size();
    int trainIterations = 0;
    SOM network(4, clustersCount);
    do {
        avgDelta = 0.;
        for(Examples::iterator e = examples.begin(); e != examples.end(); ++e) {
            float delta = network.train(e->in());
            avgDelta += delta;
            trainIterations++;
        }
        avgDelta /= examples.size();
        cout << "\r" << avgDelta;
    } while(avgDelta > 0.74);

    cout << endl << "Trained in " << trainIterations << " iterations" << endl;

    vector<float> noiseLevels = {0, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90};
    for(int i = 0; i < noiseLevels.size(); ++i) {
        errs = 0;
        cout << "Noise level: " << noiseLevels[i] << "%" << endl;
        for(Examples::iterator e = examples.begin(); e != examples.end(); ++e) {
            Representation in = e->in();
            NeuroIO result = network.classify(in.apply_noise(noiseLevels[i]));

            int expected = max_index(e->out()), got = max_index(network.out());
            if(expected == got) {
                cout << "\033[32m✔\033[0m " << expected << " == " << got << endl;
            } else {
                errs++;
                cout << "\033[31m✘\033[0m " << expected << " != " << got << endl;
            }
        }
        cout << "Errors: " << errs * 1. / examples.size() << endl << endl;
    }
}

void readIrisDB(Examples& examples, string path)
{
   
    std::ifstream       file(path);
    for(CSVIterator r(file);r != CSVIterator();++r) {
        std::vector<float> features = {stof((*r)[0]), stof((*r)[1]), stof((*r)[2]), stof((*r)[3])};
        examples.push_back(Example(features, (*r)[4], tagsMap[(*r)[4]], tagsMap.size()));
    }
}
