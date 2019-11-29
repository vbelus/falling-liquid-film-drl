#include <valarray>

#ifndef ADV_IN_TIME
#define ADV_IN_TIME

namespace ait
{

  void advect_in_time(std::valarray<double> & h,
                      std::valarray<double> & q,
                      std::valarray<double> & noise,
                      unsigned int n_step,
                      double dx,
                      double dt,
                      double delta,
                      double & t,
                      int half_width_jet,
                      std::valarray<int> middle_jet,
                      std::valarray<double> jet_strength_old,
                      std::valarray<double> jet_strength_new,
                      int NUM) noexcept;

} // namespace ait

#endif
