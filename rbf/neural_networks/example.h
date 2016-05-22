#ifndef EXAMPLE_H_W85U4RF9
#define EXAMPLE_H_W85U4RF9

#include "opencv2/imgproc/imgproc.hpp"
#include  "representation.h"
#include <vector>

namespace Neural {
class Example {
    Representation r;
    std::vector<float> expected;
    std::string tagName;
    int tag;

    static void img2representation(cv::Mat& img, Neural::Representation& representation, int threshold = 180)
    {
        representation.resize(img.rows * img.cols);
        int i = 0;
        for(int y = 0; y < img.rows; ++y) {
            for(int x = 0; x < img.cols; ++x) {
                representation[i] = img.at<uchar>(y, x) < 180 ? 0 : 1;
                i++;
            }
        }
    }

public:
    Example() {}

    Example(cv::Mat image, int binarizationThreshold, std::string tagName, int tag, int tagsCount)
        : expected(tagsCount), tagName(tagName), tag(tag)
    {
        expected[tag] = 1;
        img2representation(image, r, binarizationThreshold);
    }

    Example(std::vector<float> r, std::string tagName, int tag, int tagsCount)
        : r(r), expected(tagsCount), tagName(tagName), tag(tag)
    {
        expected[tag] = 1;
    }


    float& operator[](int ind)
    {
        return r[ind];
    }

    Representation& in()
    {
        return r;
    }
    std::vector<float>& out()
    {
        return expected;
    }
    int getTag()
    {
        return tag;
    }
};
}

typedef vector<Example> Examples;

#endif /* end of include guard: EXAMPLE_H_W85U4RF9 */
