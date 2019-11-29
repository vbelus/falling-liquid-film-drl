#include <iostream>
#include <cmath>
#include <valarray>
#include <cassert>
#include "TVD3.h"

namespace nTVD3 {

  void TVD3(std::valarray<double> & v_in,
            std::valarray<double> & v_out,
            int NUM) noexcept {
    /*
     * Applying the TVD3 scheme to the v_in,
     * and returning the result in v_out.
     *
     * v_in and v_out must have same size.
     *
     * The rewrite is done in such a way to avoid
     * going through the arrays many times and
     * to avoid memory overhead, so a bit different
     * in appearance compared with the matlab code.
     *
     * TODO: check if the compiler unrolls the first and
     * last loop iterations to avoid the tests in the main
     * body.
     *
     * TODO: discuss with Jonathan if some better way to
     * write the numerical scheme in an efficient way.
     *
     */

    assert(v_in.size() == v_out.size());

    constexpr double tol = 1e-20;

    double diff_i;
    double diff_ip1;

    diff_ip1 = 0;

    double previous_in_array = 0;
    double crrt_in_array = 0;

    double crrt_tvd;

    double next_v_in;
    double crrt_v_in;

    next_v_in = v_in[0];

    // NOTE: the manual unroll is to avoid the
    // if else if else statement in the for loop

    // NOTE: only v_in[i + 1] is used at each step,
    // for filling v_out[i]

    // manual unroll for i = 0
    unsigned int i = 0;

    diff_i = diff_ip1;
    crrt_v_in = next_v_in;
    next_v_in = v_in[i + 1];

    diff_ip1 = next_v_in - crrt_v_in;

    crrt_tvd = diff_i / (diff_ip1 + tol);
    crrt_tvd = (crrt_tvd + std::abs(crrt_tvd)) / (1.0 + crrt_tvd);

    previous_in_array = crrt_in_array;
    crrt_in_array = crrt_tvd * diff_ip1 / 2.0;

    crrt_tvd = crrt_in_array - previous_in_array;

    v_out[i] = diff_i;

    for (i = 1; i < NUM - 1; i++){
      diff_i = diff_ip1;
      crrt_v_in = next_v_in;
      next_v_in = v_in[i + 1];

      diff_ip1 = next_v_in - crrt_v_in;

      crrt_tvd = diff_i / (diff_ip1 + tol);
      crrt_tvd = (crrt_tvd + std::fabs(crrt_tvd)) / (1.0 + crrt_tvd);

      previous_in_array = crrt_in_array;
      crrt_in_array = crrt_tvd * diff_ip1 / 2.0;

      crrt_tvd = crrt_in_array - previous_in_array;

      v_out[i] = diff_i + crrt_tvd;
    }

    // manual unroll i = v_in.size() - 1
    i = v_in.size() - 1;

    diff_i = diff_ip1;
    crrt_v_in = next_v_in;
    next_v_in = v_in[i + 1];

    diff_ip1 = 0;

    crrt_tvd = diff_i / (diff_ip1 + tol);
    crrt_tvd = (crrt_tvd + std::fabs(crrt_tvd)) / (1.0 + crrt_tvd);

    previous_in_array = crrt_in_array;
    crrt_in_array = crrt_tvd * diff_ip1 / 2.0;

    crrt_tvd = crrt_in_array - previous_in_array;

    v_out[i] = diff_i;
  }
} // namespace nTVD3
