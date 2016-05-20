#ifndef UTIL_H_Z26W1JSC
#define UTIL_H_Z26W1JSC

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>

#define TO_F(val)  ((float) val)

#define M_BLACK_FG 30
#define M_BLACK_BG 40
#define M_BLUE_FG 34
#define M_BLUE_BG 44
#define M_BOLD  1
#define M_RESET 0

int linearActivationFunction(int x)
{
    return x > 0 ? 1 : -1;
}

void img2representation(cv::Mat& img, Neural::Representation& representation, int threshold = 180)
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

std::ostream& anci_escape(std::ostream& os, std::vector<unsigned char>&& modes)
{
    for(int i = 0; i < modes.size(); ++i) {
        os << "\033[" << modes[i] << 'm';

    }
    return os;
}


void representation2img(cv::Mat& img, Neural::Representation& representation, int threshold = 180)
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

#endif /* end of include guard: UTIL_H_Z26W1JSC */
