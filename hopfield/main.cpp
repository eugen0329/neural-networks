#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <functional>

#include <clocale>
#include <dirent.h>

#include "representation.h"

/* #include <io.h> */
/* #include <fcntl.h> */

#define WIN_NAME "L"

using namespace std;
using namespace cv;

// Kind of template typedef
template <typename T> class matrix : public std::vector<std::vector<T> > {};
/* typedef vector<int> Representation; */
typedef vector<Representation> Representations;
typedef Mat Image;
typedef vector<Image> Images;


void teach(Representations& representations, matrix<int>& weights);

int weight(Representations& representations, int y, int x);
void img2representation(Mat& img, Representation& representation, int threshold = 180);
/* void inspect_matrix(matrix<int>& mt); */
int linearActivationFunction(int x);



void classify(matrix<int>& weights, Representation& image, Representation& classified, function<int(int)> f)
{
    int retries = 1000;
    Representation post = image, pre = image;

    if(image.size() != classified.size())
        classified.resize(image.size());

    for(int retry = 0; retry < retries; ++retry) {
        for(int i = 0; i < weights.size(); ++i) {
            int sum = 0;
            for(int j = 0; j < weights[0].size(); ++j) {
                sum += weights[j][i] * pre[j];
            }
            post[i] = f(sum);
        }
        if(post == pre)
            break;
        pre = post;
    }
    classified = post;
}

int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");
    Images images = {
        imread("examples/a.jpg", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/i.jpg", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/p.jpg", CV_LOAD_IMAGE_GRAYSCALE)
    };

    Representations representations(images.size());
    Representations representations_after(images.size());
    for(int i = 0; i < images.size(); ++i)
        img2representation(images[i], representations[i]);

    matrix<int> weights;
    teach(representations, weights);

    Representation r = representations[1];
    cout << r.to_string(images[0].cols);

    r.apply_noise(90);
    cout << r.to_string(images[0].cols);
    puts("");

    Representation classified;
    classify(weights, r, classified, linearActivationFunction);

    /* inspect_representation(classified, images[0].cols); */




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

void teach(Representations& representations, matrix<int>& weights)
{
    int neurons_count = representations[0].size();
    weights.resize(neurons_count);
    for(int y = 0; y < neurons_count; ++y) {
        weights[y].resize(neurons_count);
        for(int x = 0; x < neurons_count; ++x) {
            weights[y][x] = weight(representations, y, x);
        }
    }

}

int weight(Representations& representations, int y, int x)
{
    if(x == y) {
        return 0;
    } else {
        int sum = 0;
        for(int i = 0; i < representations.size(); ++i) {
            sum += representations[i][x] * representations[i][y];
        }
        return sum;
    }
}

void img2representation(Mat& img, Representation& representation, int threshold)
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
