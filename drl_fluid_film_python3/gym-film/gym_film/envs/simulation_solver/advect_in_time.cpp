#include <valarray>
#include "TVD3.h"
#include "advect_in_time.h"
#include <cassert>
#include "utils.h"
#include <vector>

#define MACRO_H_XX(i) ((h[i-1] + h[i+1] - 2.0 * h[i]) / dx / dx)

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
    int NUM) noexcept {

      std::valarray<double> q_old  (NUM);
      std::valarray<double> h_old  (NUM);
      std::valarray<double> q2_h_x (NUM);
      std::valarray<double> dh     (NUM);

      assert(h.size() == q.size());
      assert(noise.size() == n_step);

      double delta_5 = 5.0 * delta;
      double q2_h_BC;
      double crrt_dq;
      double crrt_h_xxx;
      double previous_h_xxx = 0;
      unsigned int i_p_1;

      double a[3] = {0.0, 0.25, 2.0/3.0};
      double b[3] = {1.0, 0.25, 2.0/3.0};
      double c[3] = {1.0, 0.75, 1.0/3.0};

      double idx  = 1.0/dx;
      double normalisation_jet = 1.0 / static_cast<double>(half_width_jet) /
      static_cast<double>(half_width_jet);

      double jet_strength;
      int index = 0;
      double jet_value;

      for (unsigned int i_time=0; i_time<n_step; i_time++)
      {
        h_old = h;
        q_old = q;



        for (unsigned int i_RK=0; i_RK<3; i_RK++)
        {
          // even those TVD3 stuff could be computed on-the-fly to avoid the need to
          // store arrays for dh and q2_h_x
          // This should be quite OK with the way TVD3 is written, manly need to rename
          // local variables.
          // gains to expect: loop through 4 arrays instead of 6, reduce memory consumption
          // (no need for dh and q2_h_x).
          nTVD3::TVD3(q, dh, NUM);

          dh     *= -idx;
          dh[1]   = -(q[1] - q[0])*idx;
          q2_h_x  = q*q/h;
          q2_h_BC = (q2_h_x[1] - q2_h_x[0])*idx;

          nTVD3::TVD3(q2_h_x, q2_h_x, NUM);

          q2_h_x    *= idx;
          q2_h_x[1]  = q2_h_BC;
          crrt_h_xxx =  (4.0 * MACRO_H_XX(1+1) - 3.0 * MACRO_H_XX(1)
          - MACRO_H_XX(1+2))*0.5*idx;

          for (int i=1; i < NUM-2; i++){

            previous_h_xxx = crrt_h_xxx;
            i_p_1          = i + 1;

            if (i_p_1 == NUM-2){
              //
            }
            else if (i_p_1 == NUM-3){
              crrt_h_xxx =  (MACRO_H_XX(NUM-2) - MACRO_H_XX(NUM-3))*idx;
            }
            else{
              crrt_h_xxx =  (4.0 * MACRO_H_XX(i_p_1+1)
              - 3.0 * MACRO_H_XX(i_p_1) - MACRO_H_XX(i_p_1+2))*0.5*idx;
            }

            crrt_dq = (h[i] * (previous_h_xxx + 1.0)
            - q[i] / h[i] / h[i]) / delta_5 - q2_h_x[i] * 1.2;


            // Here we apply the control
            for (unsigned int j=0; j < middle_jet.size(); j++)
            {
              jet_strength = (jet_strength_old[j] + (jet_strength_new[j] - jet_strength_old[j])*static_cast<double>(i_time) / static_cast<double>(n_step));

              index = i - middle_jet[j];
              if ((index > -half_width_jet) && (index < half_width_jet))
              {
                jet_value = (half_width_jet + index)
                * (half_width_jet - index)
                * jet_strength
                * normalisation_jet;
                crrt_dq += jet_value;
              }

            }


            h[i] = a[i_RK]*h[i] + b[i_RK]*dh[i]*dt   + c[i_RK]*h_old[i];
            q[i] = a[i_RK]*q[i] + b[i_RK]*crrt_dq*dt + c[i_RK]*q_old[i];
          }

          // the BCs
          h[0] = 1 + noise[i_time];
          q[0] = 1;

          h[NUM - 2] = h[NUM - 3];
          h[NUM - 1] = h[NUM - 2];

          q[NUM - 2] = q[NUM - 3];
          q[NUM - 1] = q[NUM - 2];
        }

        t += dt;

      }
    }

  } // namespace ait
