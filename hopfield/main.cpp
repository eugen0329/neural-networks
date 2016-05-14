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
typedef matrix<int> Representation;
typedef vector<Representation> Representations;
typedef vector<Mat> Images;


void img2representation(Mat& img, Representation& representation, int threshold = 180)
{
    representation.resize(img.rows);
    for(int y = 0; y < img.rows; ++y) {
        representation[y].resize(img.cols);
        for(int x = 0; x < img.cols; ++x) {
            representation[y][x] = img.at<uchar>(y, x) < 180 ? -1 : 1;
        }
    }
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


    for(int i = 0; i < images.size(); ++i)
        img2representation(images[i], representations[i]);


    /* _setmode(_fileno(stdout), _O_U16TEXT); */
    for(int i = 0; i < representations.size(); ++i) {
        for (int j = 0; j < representations[i].size(); ++j) {
            for (int k = 0; k < representations[i][j].size(); ++k) {
                std::cout << (representations[i][j][k] == 1 ? " " : "█");
            }
            puts("");
        }
        puts("");
    }

    /* img2representation(); */
    /* img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE); */

    /* namedWindow(WIN_NAME, CV_WINDOW_AUTOSIZE); */
    /* imshow(WIN_NAME, img); */

    waitKey(0);
    return 0;
}

