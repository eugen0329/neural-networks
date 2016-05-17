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
const char imageNames[][3] = { "a", "и", "p" };

int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");
    Images images = {
        imread("examples/a.jpg", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/i.jpg", CV_LOAD_IMAGE_GRAYSCALE),
        imread("examples/p.jpg", CV_LOAD_IMAGE_GRAYSCALE)
    };

    std::vector<int> noiseLevels = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

    Neural::Representations representations(images.size());
    for(int i = 0; i < images.size(); ++i)
        img2representation(images[i], representations[i]);

    Neural::Hopfield network;
    network.teach(representations);


    Neural::Representation image, classified;
    for(int r = 0; r < representations.size(); ++r) {
        for(int l = 0; l != noiseLevels.size(); ++l) {
            image = representations[r];
            image.apply_noise(noiseLevels[l]);
            network.classify(image, classified, linearActivationFunction);

            cout << "Image: \"\033[32m"
                << imageNames[r] << "\033[39m\". Noise level: \033[31m"
                << noiseLevels[l] << "\033[39m" << endl;

            cout << "\033[37m\033[47m\033[1m"
                << image.to_string(images[0].cols)
                << "\033[0m" << endl;
            cout << "\033[37m\033[47m\x1B[1m"
                << classified.to_string(images[0].cols)
                << "\033[0m" << endl;
        }
    }

    waitKey(0);
    return 0;
}

