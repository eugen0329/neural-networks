#ifndef HOPFIELD_H_KYMN5SRH
#define HOPFIELD_H_KYMN5SRH

#include <functional>
#include "representation.h"


namespace Neural {

class Hopfield {
private:
    typedef std::vector<std::vector<int>> Weights;
    Weights weights;

    int weight(Representations& representations, int y, int x)
    {
        if(x == y) return 0;

        int sum = 0;
        for(int i = 0; i < representations.size(); ++i) {
            sum += representations[i][x] * representations[i][y];
        }
        return sum;
    }

public:
    Hopfield() {}

    void teach(Representations &representations)
    {
        int neurons_count = representations[0].size();
        weights.resize(neurons_count);
        for (int y = 0; y < neurons_count; ++y) {
            weights[y].resize(neurons_count);
            for (int x = 0; x < neurons_count; ++x) {
               weights[y][x] = weight(representations, y, x);
            }
        }
    }

    void classify(Representation& image, Representation& classified, std::function<int(int)> f)
    {
        int retries = 1000;
        Representation post = image, pre = image;

        if(image.size() != classified.size())
            classified.resize(image.size());

        for(int retry = 0; retry < retries; ++retry) {
            for(int i = 0; i < weights.size(); ++i) {
                int sum = 0;
                for(int j = 0; j < weights[0].size(); ++j) {
                    sum += weights[j][i] * pre[j];
                }
                post[i] = f(sum);
            }
            if(post == pre)
                break;
            pre = post;
        }
        classified = post;
    }
};
}

#endif /* end of include guard: HOPFIELD_H_KYMN5SRH */
