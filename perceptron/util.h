#ifndef UTIL_H_APDHYTVM
#define UTIL_H_APDHYTVM

#define TO_F(val)  ((float) val)
#include <typeinfo>

int hiddenLayerSize(int inSize, int outSize, int examplesCount)
{
    return 20;
    // http://stackoverflow.com/questions/10565868/multi-layer-perceptron-mlp-architecture-criteria-for-choosing-number-of-hidde
    return std::ceil((inSize + outSize) * 2.0 / 3);
    // OR
    // h = sqrt(p/n)
    // n: inputs, m: outputs, h: hidden, p:examples count
    return std::ceil(std::sqrt(TO_F(examplesCount) / inSize));
}

float randInRange(float from, float to)
{
    return from + static_cast <float> (rand() * 1.0) /( static_cast <float> (RAND_MAX/(to-from)));
}

template<class T=float>
std::string inspectVec(std::vector<T> v)
{
    std::string dumped;
    for (typename std::vector<T>::const_iterator i = v.begin(); i != v.end(); ++i)
        dumped += std::to_string(*i) + ' ';
    return dumped;
}

template<class T>
int max_index(T it)
{
    return std::distance(it.begin(), std::max_element(it.begin(), it.end()));
}

#endif /* end of include guard: UTIL_H_APDHYTVM */
