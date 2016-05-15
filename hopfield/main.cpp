#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>

#include <clocale>
#include <dirent.h>

#include "neural_networks/hopfield.h"

/* #include <io.h> */
/* #include <fcntl.h> */

#define WIN_NAME "L"

using namespace std;
using namespace cv;

// Kind of template typedef
/* typedef vector<int> Representation; */
typedef Mat Image;
typedef vector<Image> Images;


void img2representation(Mat& img, Neural::Representation& representation, int threshold = 180);
/* void inspect_matrix(matrix<int>& mt); */
int linearActivationFunction(int x);


int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");
    Images images = {
        imread("examples/a.jpg", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/i.jpg", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/p.jpg", CV_LOAD_IMAGE_GRAYSCALE)
    };


    Neural::Representations representations(images.size());
    Neural::Representations representations_after(images.size());

    Neural::Hopfield network(representations[0].size());

    for(int i = 0; i < images.size(); ++i)
        img2representation(images[i], representations[i]);

    network.teach(representations);

    Neural::Representation r = representations[1];
    cout << r.to_string(images[0].cols);

    r.apply_noise(90);
    cout << r.to_string(images[0].cols);
    puts("");

    Neural::Representation classified;
    network.classify(r, classified, linearActivationFunction);

    std::cout << classified.to_string(images[0].cols);


    waitKey(0);
    return 0;
}

int linearActivationFunction(int x)
{
    return x > 0 ? 1 : -1;
}

/* void print_images(Images& images) */
/* { */
/*     for(int i = 0; i < images.size(); ++i) { */
/*         for (int j = 0; j < images[i].rows; ++j) { */
/*             for (int k = 0; k < images[i].cols; ++k) { */
/*                 std::cout << (images[i].at<uchar>(j,k) > 180 ? " " : "█"); */
/*             } */
/*             puts(""); */
/*         } */
/*         puts(""); */
/*     } */
/* } */



/* void inspect_matrix(matrix<int>& mt) */
/* { */
/*     for(int i = 0; i < mt.size(); ++i) { */
/*         for (int j = 0; j < mt[i].size(); ++j) { */
/*              /1* printf("%3d\n", mt[i][j]); *1/ */
/*              printf("%d\n", mt[i][j]); */
/*         } */
/*         /1* puts(""); *1/ */
/*     } */
/* } */


void img2representation(Mat& img, Neural::Representation& representation, int threshold)
{
    representation.resize(img.rows * img.cols);
    int i = 0;
    for(int y = 0; y < img.rows; ++y) {
        /* representation[y].resize(img.cols); */
        for(int x = 0; x < img.cols; ++x) {
            representation[i] = img.at<uchar>(y, x) < 180 ? -1 : 1;
            i++;
        }
    }
}
