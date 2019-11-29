#include <iostream>
#include <valarray>
#include "utils.h"

namespace utils {

    template <typename T>
        void print_details_valarray(const std::valarray<T> &vec, bool print_addresses){
            std::cout << "\nprint valarray\n\n";

            for (const T i : vec){
                std::cout << i << "\n";
            }

            if (print_addresses){

                std::cout << "\nprint valarray addresses\n";


                for (const T &i : vec){

                    std::cout << &i << "\n";

                }

            }

            std::cout << "\nDone print valarray\n\n";
        }

} // namespace utils
