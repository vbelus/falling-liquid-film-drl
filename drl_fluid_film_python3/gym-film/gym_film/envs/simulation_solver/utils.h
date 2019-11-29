#include <valarray>

#ifndef UTILS
#define UTILS

namespace utils {

    template <typename T>
    void print_details_valarray(const std::valarray<T> &vec, bool print_addresses=false);

} // namespace utils

// those are templated functions; need to be in the header to allow instanciation
#include "utils.cpp"

#endif
