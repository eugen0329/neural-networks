#ifndef UTIL_H_APDHYTVM
#define UTIL_H_APDHYTVM

#define TO_F(val)  ((float) val)
#include <typeinfo>
#include <functional>
#include <list>
#include <vector>

using namespace std;

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

void values(map<string, int>& from, vector<int>& to)
{
    to.resize(from.size());
    for(map<string, int>::iterator p = from.begin(); p != from.end(); ++p) {
        p->second;
    }
}

template<class T=float>
std::string inspectVec(std::vector<T> v)
{
    std::string dumped;
    for (typename std::vector<T>::const_iterator i = v.begin(); i != v.end(); ++i)
        dumped += std::to_string(*i) + ' ';
    return dumped;
}

float euklidNorm(const vector<float>& vec)
{
    vector<float> tmp(vec.size());
    transform(begin(vec), end(vec), begin(tmp), [](const float& a) { return a*a; });
    return sqrt(accumulate(begin(tmp), end(tmp), 0.0));
}

template<class T>
int max_index(T it)
{
    return std::distance(it.begin(), std::max_element(it.begin(), it.end()));
}

template<class T>
int min_index(T it)
{
    return std::distance(it.begin(), std::min_element(it.begin(), it.end()));
}


template<class T>
void combinations(vector<T> vec, int len, std::function<void(std::list<T>&)> callback)
{
    int n = vec.size(), r = len;

    std::vector<bool> selector(n);
    std::fill(selector.begin() + n - r, selector.end(), true);
    std::list<T> selected;

    do {
        for (int i = 0; i < n; ++i) {
            if (selector[i]) {
                selected.push_back(vec[i]);
            }
        }
        callback(selected);
        selected.clear();
    } while (std::next_permutation(selector.begin(), selector.end()));

}

#endif /* end of include guard: UTIL_H_APDHYTVM */
