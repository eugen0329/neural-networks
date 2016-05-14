#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <dirent.h>
#include <clocale>

/* #include <io.h> */
/* #include <fcntl.h> */

#define WIN_NAME "L"

using namespace std;
using namespace cv;

// Kind of template typedef
template <typename T> class matrix : public std::vector<std::vector<T> > {};
typedef vector<int> Representation;
typedef vector<Representation> Representations;
typedef vector<Mat> Images;


void teach(Representations& , matrix<int>&);

int weight(Representations& representations, int y, int x);
void img2representation(Mat& img, Representation& representation, int threshold = 180);
void inspect_matrix(matrix<int>& mt);

int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");
    Images images = {
        imread("examples/a.jpg", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/i.jpg", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/p.jpg", CV_LOAD_IMAGE_GRAYSCALE)
    };

    Representations representations(images.size());


    for(int i = 0; i < images.size(); ++i)
        img2representation(images[i], representations[i]);


    matrix<int> weights;
    teach(representations, weights);

    matrix<int> weights2(weights);
    teach(representations, weights2);

    inspect_matrix(weights);
    puts("");
    inspect_matrix(weights2);

    /* img2representation(); */
    /* img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE); */

    /* namedWindow(WIN_NAME, CV_WINDOW_AUTOSIZE); */
    /* imshow(WIN_NAME, img); */

    waitKey(0);
    return 0;
}

void print_images(Images& images)
{
    for(int i = 0; i < images.size(); ++i) {
        for (int j = 0; j < images[i].rows; ++j) {
            for (int k = 0; k < images[i].cols; ++k) {
                std::cout << (images[i].at<uchar>(j,k) > 180 ? " " : "█");
            }
            puts("");
        }
        puts("");
    }
}


void inspect_matrix(matrix<int>& mt)
{
    for(int i = 0; i < mt.size(); ++i) {
        for (int j = 0; j < mt[i].size(); ++j) {
             printf("%3d", mt[i][j]);
        }
        puts("");
    }
}

void teach(Representations& representations, matrix<int>& weights)
{
    int neurons_count = 10;

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
        for (Representations::iterator r = representations.begin();  r != representations.end(); ++r) {
            sum += (*r)[x] + (*r)[y];
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
