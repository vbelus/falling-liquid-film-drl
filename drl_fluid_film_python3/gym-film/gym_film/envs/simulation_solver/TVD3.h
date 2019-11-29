#include <valarray>

#ifndef NTVD3
#define NTVD3

namespace nTVD3
{

    void TVD3(std::valarray<double> & v_in, 
              std::valarray<double> & v_out,
              int NUM) noexcept;

} // namespace TVD3

#endif
