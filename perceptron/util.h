#ifndef UTIL_H_APDHYTVM
#define UTIL_H_APDHYTVM

#define TO_F(val)  ((float) val)

int hiddenLayerSize(int inSize, int outSize, int examplesCount)
{
    return 3;
    // http://stackoverflow.com/questions/10565868/multi-layer-perceptron-mlp-architecture-criteria-for-choosing-number-of-hidde
    return std::ceil((inSize + outSize) * 2.0 / 3);
    // OR
    // h = sqrt(p/n)
    // n: inputs, m: outputs, h: hidden, p:examples count
    return std::ceil(std::sqrt(TO_F(examplesCount) / inSize));
}

float randInRange(float from, float to)
{
    return from + (rand() % (int)(to - from + 1));
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

#endif /* end of include guard: UTIL_H_APDHYTVM */
