#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"

using namespace xt::placeholders;  // required for `_` to work

class cem
{
private:
    /* data */
    int in_size = 5;
    int out_size = 2;

    int hidden_size = 5;
    int evalation_samples = 1;

    // training parameters
    int cem_iterations = 100;// How many total CEM iterations 
    int cem_batch_size = 50;// How many guassian samples in each CEM iteration
    double cem_elite_frac = 0.5;// What percentage of cem samples are used to fit the guassian for next iteration
    double cem_init_stddev = 1.0;// Initial CEM guassian uncertainty
    double cem_noise_factor = 1.0;  // Scaling factor of how much extra noise to add each iteration (noise_factor/iteration_number noise is added to std.dev.)
    int cem_print_rate = 5;
    
    // simulation parameters
    double dt = 0.1;
    double runtime = 8.0;
    
    // target task
    xt::xarray<double> car_start = xt::xarray<double>({-50.0, 0.0, 0.751});
    xt::xarray<double> car_goal = xt::xarray<double>({50.0, 0.0});
    
    // car dynamics parameters
    double v_max = 80;
    double omega_max = 3.14;

    // car shape 
    double car_w = 5;
    double car_l = 10;

public:
    cem(/* args */);
    ~cem();
    xt::xarray<double> linear_model(xt::xarray<double> params, xt::xarray<double> in_data);
    xt::xarray<double> doule_layer_model(xt::xarray<double> params, xt::xarray<double> in_data);
    // cem update
    xt::xarray<double> update_state(xt::xarray<double> params, xt::xarray<double> cur_state, xt::xarray<double> goal);
    // run model
    // reward
};

cem::cem(/* args */)
{
}

cem::~cem()
{
}

xt::xarray<double> cem::doule_layer_model(xt::xarray<double> params, xt::xarray<double> in_data) 
{
    int two_layer_policy_size = (in_size+1)*hidden_size + (hidden_size+1)*out_size;
    auto in_vec = in_data.reshape({in_size, 1});

    // layer1 input->hidden
    int m1_end = hidden_size * in_size;
    // matrix1 = np.reshape(params[0:m1_end], (hidden_size,in_size))
    xt::xarray<double> shit = xt::view(params, xt::range(_, m1_end), 1);
    xt::xarray<double> matrix1 = shit.reshape({hidden_size, in_size});
    xt::xarray<double> shit2 = xt::view(params, xt::range(m1_end, m1_end + hidden_size), 1);
    xt::xarray<double> biases1 = shit2.reshape({hidden_size, 1});
    xt::xarray<double> hidden_out = (matrix1 * in_vec) + biases1;
    hidden_out = hidden_out  * (hidden_out > 0) + 0.1 * hidden_out * (hidden_out < 0); //Leaky ReLU
}
