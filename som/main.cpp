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
    /* images(); */
    iris();
    return 0;
}

map<string, int> tagsMap = {{"Iris-setosa", 0}, {"Iris-versicolor", 1}, {"Iris-virginica", 2}};
void iris()
{
    vector<int> tags;
    values(tagsMap, tags);

    Examples examples;
    readIrisDB(examples, "iris.csv");
    random_shuffle(examples.begin(), examples.end());


    int errs = 0;
    int clustersCount = tagsMap.size();
    SOM network(4, clustersCount);
    for(int i = 0; i < 10000; ++i) {
        errs = 0;
        for(Examples::iterator e = examples.begin(); e != examples.end(); ++e) {
            network.train(e->in());

            cout << max_index(e->out()) << max_index(network.out()) << '\n';
            if(max_index(e->out()) != max_index(network.out())) {
                errs++;
            }
            /* copy(network.out().begin(), network.out().end(), ostream_iterator<float>(cout, " ")); */
            /* cout << endl; */
        }
        cout << errs * 1. / examples.size()  << endl;
    }
    return;

    cout << "Expected | Got" << endl;
    for(Examples::iterator e = examples.begin(); e != examples.end(); ++e) {
        NeuroIO result = network.classify(e->in());

        if(max_index(e->out()) != max_index(network.out())) {
            /* errs++; */
            cout << "\033[31m✘\033[0m " << max_index(e->out()) << " != " << max_index(result) << endl;
        } else {
            cout << "  " << max_index(e->out()) << " == " << max_index(result) << endl;
        }
    }
    /* cout << "Err rate: " << errs * 100. / examples.size() << "%"; */
    cout << endl;
}

void images()
{
    /* Examples examples = {{Example(imread("examples/2.1.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "2", 0, 5)}, */
    /*                      {Example(imread("examples/2.2.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "2", 0, 5)}, */
    /*                      {Example(imread("examples/3.1.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "3", 1, 5)}, */
    /*                      {Example(imread("examples/3.2.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "3", 1, 5)}, */
    /*                      {Example(imread("examples/4.1.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "4", 2, 5)}, */
    /*                      {Example(imread("examples/4.2.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "4", 2, 5)}, */
    /*                      {Example(imread("examples/5.1.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "5", 3, 5)}, */
    /*                      {Example(imread("examples/5.2.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "6", 3, 5)}, */
    /*                      {Example(imread("examples/7.1.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "7", 4, 5)}, */
    /*                      {Example(imread("examples/7.2.png", CV_LOAD_IMAGE_GRAYSCALE), 180, "7", 4, 5)}}; */

    /* Perceptron perceptron(examples[0].in().size(), examples[0].out().size(), {100}); */

    /* int errs = 0; */
    /* do { */
    /*     errs = 0; */
    /*     for(Examples::iterator it = examples.begin(); it != examples.end(); ++it) { */
    /*         float err = perceptron.train((*it).in(), (*it).out()); */
    /*         if(max_index((*it).out()) != max_index(perceptron.out())) errs++; */
    /*     } */
    /*     cout << errs / 100.0 << "\r"; */
    /* } while((errs / 100.0) > 0.05); */

    /* vector<float> noiseLevels = {2, 3, 5}; */
    /* for(Examples::iterator it = examples.begin(); it != examples.end(); ++it) { */
    /*     for(int i = 0; i < noiseLevels.size(); ++i) { */
    /*         Representation in = (*it).in(); */
    /*         NeuroIO result = perceptron.classify(in.apply_noise(noiseLevels[i])); */
    /*         cout << max_index((*it).out()) << max_index(result) << endl; */
    /*         if(max_index((*it).out()) != max_index(perceptron.out())) errs++; */
    /*     } */
    /* } */
    /* cout << errs / 100.0; */
    /* cout << endl; */
}

void readIrisDB(Examples& examples, string path)
{
   
    std::ifstream       file(path);
    for(CSVIterator r(file);r != CSVIterator();++r) {
        std::vector<float> features = {stof((*r)[0]), stof((*r)[1]), stof((*r)[2]), stof((*r)[3])};
        examples.push_back(Example(features, (*r)[4], tagsMap[(*r)[4]], tagsMap.size()));
    }
}
