#ifndef REPRESENTATIN_H_XRF6HDOZ
#define REPRESENTATIN_H_XRF6HDOZ

#include <vector>
#include <string>

namespace Neural {


template<class T=int>
class GenericRepresentation {
private:
    typedef typename std::vector<T> impl_t;
    impl_t impl;

    virtual void bipolar_inverse(int& val)
    {
        val = (val == 1 ? 0 : 1);
    }
public:
    GenericRepresentation() {}
    GenericRepresentation(int size) : impl(size) {}
    GenericRepresentation(const GenericRepresentation& other) : impl(other.impl) {}

    void resize(int n, int val = 0)
    {
        impl.resize(n, val);
    }

    T& operator[](int i)
    {
        return impl[i];
    }

    GenericRepresentation& operator=(const GenericRepresentation& rhs)
    {
        if(this != &rhs) {
            impl = rhs.impl;
        }
        return *this;
    }
    bool operator==(const GenericRepresentation& rhs)
    {
        return impl == rhs.impl;
    }

    int size()
    {
        return impl.size();
    }

    std::string to_string(int rowSize)
    {
        std::string str = "";
        for(int i = 0; i < size() / rowSize; ++i) {
            for (int j = 0; j < rowSize; ++j) {
                str += ( impl[i*rowSize + j] == 0 ? " " : "█");
            }
            str += '\n';
        }
        return str;
    }

    void apply_noise(float percent)
    {
        impl_t indexes(impl.size());
        for(int i = 0; i < impl.size(); ++i)
            indexes[i] = i;
        random_shuffle(indexes.begin(), indexes.end());

        typename impl_t::iterator lim = indexes.begin();
        std::advance(lim, indexes.size() * (percent / 100.0));
        for(typename impl_t::iterator index = indexes.begin(); index != lim; ++index) {
            bipolar_inverse(impl[*index]);
        }
    }
};

typedef GenericRepresentation<> Representation;
typedef std::vector<Representation> Representations;

}

#endif /* end of include guard: REPRESENTATIN_H_XRF6HDOZ */
