#ifndef ACTIVATION_FUNCTS_HPP_XZ8R13HG
#define ACTIVATION_FUNCTS_HPP_XZ8R13HG

#include <cmath>


namespace ActivationFuncs {
    float sigmoid(float x)
    {
        return 1.0 / (1 + pow(M_E, -x));
    }
}


#endif /* end of include guard: ACTIVATION_FUNCTS_HPP_XZ8R13HG */
