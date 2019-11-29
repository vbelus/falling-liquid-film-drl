#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <valarray>
#include <cmath>
#include <fstream>
#include <random>
#include <iostream>
#include <chrono>
#include <string>

#include "advect_in_time.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

class Simulation_cpp
{
private:
    std::valarray<double> h;
    std::valarray<double> q;
    std::valarray<int> jet_pos;
    std::valarray<double> jet_power;
    std::valarray<double> jet_power_old;
    std::valarray<double> Noise;

    double L;
    int NUM;
    double dx;
    double dt;
    double delta;
    double noise_mag;
    unsigned int n_step;
    unsigned int n_step2;

    int half_width_jet;
    double t;

    bool flag;
public:
    // Constructor
    Simulation_cpp(np::ndarray new_jet_pos,
                                  double _L,
                                  int _NUM,
                                  double _dx,
                                  double _dt,
                                  double _delta,
                                  double _noise_mag,
                                  unsigned int _n_step,
                                  unsigned int _n_step2,
                                  int _half_width_jet)
    {
        jet_pos = to_valarray_int(new_jet_pos);
        L = _L;
        NUM = _NUM;
        dx = _dx;
        dt = _dt;
        delta = _delta;
        noise_mag = _noise_mag;
        n_step = _n_step;
        n_step2 = _n_step2;
        half_width_jet = _half_width_jet;

        std::valarray<double> Noise1(n_step);
        Noise = Noise1;
        double t = 0.0;
        flag = true;
    }

    // Member functions
    std::valarray<double> to_valarray(np::ndarray array_from_py);
    std::valarray<int> to_valarray_int(np::ndarray array_from_py);
    void set_h(np::ndarray array_from_py);
    void set_q(np::ndarray array_from_py);
    void set_jet_power(np::ndarray array_from_py);

    void next_step();

    np::ndarray to_ndarray(std::valarray<double> & valarray_to_numpy);
    np::ndarray get_h();
    np::ndarray get_q();
    double get_time();
};

// Takes every element of array to given exponent
void Simulation_cpp::next_step()
{
  for (unsigned int i = 0; i < Noise.size(); i++)
  {
    Noise[i] = static_cast <float> (rand())/static_cast <float> (RAND_MAX) * noise_mag;
  }
  ait::advect_in_time(h, q, Noise,
                n_step,
                dx,
                dt,
                delta, t,
                half_width_jet,
                jet_pos,
                jet_power_old,
                jet_power,
                NUM);
}

std::valarray<double> Simulation_cpp::to_valarray(np::ndarray array_from_py)
{
  int len_array_from_py = array_from_py.shape(0);
  std::valarray<double> new_valarray(len_array_from_py);
  for(int i=0; i<len_array_from_py; i++)
  {
    new_valarray[i] = p::extract<double>(array_from_py[i]);
  }
  return new_valarray;
}

std::valarray<int> Simulation_cpp::to_valarray_int(np::ndarray array_from_py)
{
  int len_array_from_py = array_from_py.shape(0);
  std::valarray<int> new_valarray(len_array_from_py);
  for(int i=0; i<len_array_from_py; i++)
  {
    new_valarray[i] = p::extract<int>(array_from_py[i]);
  }
  return new_valarray;
}

void Simulation_cpp::set_h(np::ndarray array_from_py)
{
  h = Simulation_cpp::to_valarray(array_from_py);
}
void Simulation_cpp::set_q(np::ndarray array_from_py)
{
  q = Simulation_cpp::to_valarray(array_from_py);
}
void Simulation_cpp::set_jet_power(np::ndarray array_from_py)
{
  jet_power_old = jet_power;
  jet_power = Simulation_cpp::to_valarray(array_from_py);
  if (flag)
  {
    jet_power_old = Simulation_cpp::to_valarray(array_from_py);
    flag = false;
  }
}

// Takes valarray, returns np::ndarray - it's a deep copy
np::ndarray Simulation_cpp::to_ndarray(std::valarray<double> & valarray_to_numpy)
{
    int array_size = valarray_to_numpy.size();
    p::tuple shape = p::make_tuple(array_size);
    p::tuple stride = p::make_tuple(sizeof(double));
    np::dtype dt = np::dtype::get_builtin<double>();
    np::ndarray ndarray = np::from_data(&valarray_to_numpy[0], dt, shape, stride, p::object());
    return ndarray.copy();
}

// Return array to Python
np::ndarray Simulation_cpp::get_h()
{
  return Simulation_cpp::to_ndarray(h);
}
np::ndarray Simulation_cpp::get_q()
{
  return Simulation_cpp::to_ndarray(q);
}
double Simulation_cpp::get_time()
{
  return t;
}


BOOST_PYTHON_MODULE(cpp)
{
    Py_Initialize();
    np::initialize();
    p::class_<Simulation_cpp>("Simulation_cpp", p::init<np::ndarray, double, int, double, double, double, double, unsigned int, unsigned int, int>())
        .def("set_h", &Simulation_cpp::set_h)
        .def("set_q", &Simulation_cpp::set_q)
        .def("set_jet_power", &Simulation_cpp::set_jet_power)
        .def("next_step", &Simulation_cpp::next_step)
        .def("get_h", &Simulation_cpp::get_h)
        .def("get_q", &Simulation_cpp::get_q)
        .def("get_time", &Simulation_cpp::get_time)
    ;
}
