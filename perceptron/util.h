#ifndef UTIL_H_APDHYTVM
#define UTIL_H_APDHYTVM

int hiddenLayerSize(int inSize, int outSize, int examplesCount)
{
    // http://stackoverflow.com/questions/10565868/multi-layer-perceptron-mlp-architecture-criteria-for-choosing-number-of-hidde
    //return std::ceil((inSize + outSize) * 2.0 / 3);
    // OR
    // h = sqrt(p/n)
    // n: inputs, m: outputs, h: hidden, p:examples count
    return std::ceil(std::sqrt(TO_F(examplesCount) / inSize));
}

#endif /* end of include guard: UTIL_H_APDHYTVM */
