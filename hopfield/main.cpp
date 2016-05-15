#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>

#include <clocale>
#include <dirent.h>

#include "neural_networks/hopfield.h"
#include "util.h"

/* #include <io.h> */
/* #include <fcntl.h> */

#define WIN_NAME "L"

using namespace std;
using namespace cv;
using namespace Neural;

typedef Mat Image;
typedef vector<Image> Images;


void img2representation(Mat& img, Neural::Representation& representation, int threshold = 180);
int linearActivationFunction(int x);


int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");
    Images images = {
        imread("examples/a.jpg", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/i.jpg", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/p.jpg", CV_LOAD_IMAGE_GRAYSCALE)
    };

    std::vector<int> noiseLevels = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

    Neural::Representations representations(images.size());
    for(int i = 0; i < images.size(); ++i)
        img2representation(images[i], representations[i]);

    Neural::Hopfield network(representations[0].size());
    network.teach(representations);


    Neural::Representations::iterator
        first = representations.begin(),
        last = representations.end();

    Neural::Representation image, classified;
    for(Neural::Representations::iterator r = first; r != last; ++r) {
        for(std::vector<int>::iterator l = noiseLevels.begin(); l != noiseLevels.end(); ++l) {
            image = *r;
            image.apply_noise(*l);
            network.classify(image, classified, linearActivationFunction);

            cout << "Noise level: " << *l << endl;
            cout << image.to_string(images[0].cols);
            cout << classified.to_string(images[0].cols);
        }
    }

    waitKey(0);
    return 0;
}


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
