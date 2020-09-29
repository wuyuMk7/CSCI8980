#ifndef __RL_H__
#define __RL_H__

#include "runrl.h"

#include "xtensor/xbuilder.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xcsv.hpp"

#include "tbb/parallel_for.h"

#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <ctime>

using namespace xt::placeholders;

xt::xarray<double> reluLayer(xt::xarray<double> &in, xt::xarray<double> &weights,
                             xt::xarray<double> &biases);
xt::xarray<double> doubleLayerModel(unsigned int in_size, unsigned int hidden_size, unsigned int out_size,
                                    xt::xarray<double> &in, xt::xarray<double> &total_weights);

class RL 
{
public:
  RL(RLRunnable *obj): _obj(obj) {};
  ~RL() {};

  void cem();
  double reward();
  //double reward(xt::xarray<double> &in_data, double (*score_func) (xt::xarray<double> &out_data));
  void train();
  xt::xarray<double> run(xt::xarray<double> &in_data);

  void setStartingPoint();
  void setGoal();

  void save(const std::string &outfile);
  void load(const std::string &infile);
  bool check(const std::string &infile);

  void initModel();
  void switchModel2();
  void switchModel3();

  unsigned int in_size = 8;
  unsigned int out_size = 2;

  unsigned int model_version = 2;
  unsigned int hidden_size = 100;
  unsigned int hidden_size1 = 10;
  unsigned int hidden_size2 = 20;
  unsigned int evaluation_samples = 10;

  unsigned int policy_size = 0;

  unsigned int cem_iterations = 100;
  unsigned int cem_batch_size = 100;
  unsigned int cem_print_rate = 5;

  double cem_elite_frac = 0.25;
  double cem_mean = 1.0;
  double cem_stddev = 0.5;
  double cem_noise_factor = 1.0;

  unsigned int rl_num_of_tasks = 1;

  xt::xarray<double> params;
  RLRunnable *_obj;
};

#endif

  // using namespace xt::placeholders;  // required for `_` to work
  //
  // class cem
  // {
  // private:
  //     /* data */
  //     int in_size = 5;
  //     int out_size = 2;
  //
  //     int hidden_size = 5;
  //     int evalation_samples = 1;
  //
  //     // training parameters
  //     int cem_iterations = 100;// How many total CEM iterations
  //     int cem_batch_size = 50;// How many guassian samples in each CEM
  //     iteration double cem_elite_frac = 0.5;// What percentage of cem samples
  //     are used to fit the guassian for next iteration double cem_init_stddev
  //     = 1.0;// Initial CEM guassian uncertainty double cem_noise_factor
  //     = 1.0;  // Scaling factor of how much extra noise to add each iteration
  //     (noise_factor/iteration_number noise is added to std.dev.) int
  //     cem_print_rate = 5;
  //
  //     // simulation parameters
  //     double dt = 0.1;
  //     double runtime = 8.0;
  //
  //     // target task
  //     xt::xarray<double> car_start = xt::xarray<double>({-50.0, 0.0, 0.751});
  //     xt::xarray<double> car_goal = xt::xarray<double>({50.0, 0.0});
  //
  //     // car dynamics parameters
  //     double v_max = 80;
  //     double omega_max = 3.14;
  //
  //     // car shape
  //     double car_w = 5;
  //     double car_l = 10;
  //
  // public:
  //     cem(/* args */);
  //     ~cem();
  //     xt::xarray<double> linear_model(xt::xarray<double> params,
  //     xt::xarray<double> in_data); xt::xarray<double>
  //     doule_layer_model(xt::xarray<double> params, xt::xarray<double>
  //     in_data);
  //     // cem update
  //     xt::xarray<double> update_state(xt::xarray<double> params,
  //     xt::xarray<double> cur_state, xt::xarray<double> goal);
  //     // run model
  //     // reward
  // };
  //
  // cem::cem(/* args */)
  // {
  // }
  //
  // cem::~cem()
  // {
  // }
  //

  /*
  xt::xarray<double> cem::doule_layer_model(xt::xarray<double> params,
  xt::xarray<double> in_data)
  {
      int two_layer_policy_size = (in_size+1)*hidden_size +
  (hidden_size+1)*out_size; auto in_vec = in_data.reshape({in_size, 1});

      // layer1 input->hidden
      int m1_end = hidden_size * in_size;
      // matrix1 = np.reshape(params[0:m1_end], (hidden_size,in_size))
      xt::xarray<double> view1 = xt::view(params, xt::range(_, m1_end), 1);
      xt::xarray<double> matrix1 = view1.reshape({hidden_size, in_size});
      xt::xarray<double> view2 = xt::view(params, xt::range(m1_end, m1_end +
  hidden_size), 1); xt::xarray<double> biases1 = view2.reshape({hidden_size,
  1}); xt::xarray<double> hidden_out = (matrix1 * in_vec) + biases1; hidden_out
  = hidden_out  * (hidden_out > 0) + 0.1 * hidden_out * (hidden_out < 0);
  //Leaky ReLU
  }
  */
