#include "rl.h"

/*
 * in: in_size * 1
 * weight: out_size * in_size
 * bias: out_size * 1
 * out: out_size * 1 
 *
 * out_size - hidden_size (when it's a hidden layer)
 *
 */

xt::xarray<double> reluLayer(
    xt::xarray<double> &in,
    xt::xarray<double> &weights,
    xt::xarray<double> &biases
) {
  xt::xarray<double> out = xt::linalg::dot(weights, in) + biases;
  out = out * (out > 0) + 0.1 * out * (out < 0);

  return out;
}

xt::xarray<double>
doubleLayerModel(
    unsigned int in_size,
    unsigned int hidden_size,
    unsigned int out_size,
    xt::xarray<double> &in,
    xt::xarray<double> &total_weights
) {
  unsigned int l1_end = hidden_size * in_size,
    l2_start = l1_end + hidden_size, l2_end = l2_start + out_size * hidden_size;

  // First layer
  xt::xarray<double> l1_weights_view = xt::view(total_weights, xt::range(_, l1_end), xt::all()),
    l1_weights = l1_weights_view.reshape({ hidden_size, in_size }),
    l1_biases_view = xt::view(total_weights, xt::range(l1_end, l2_start), xt::all()),
    l1_biases = l1_biases_view.reshape({ hidden_size, 1 });
  xt::xarray<double> l1_out = reluLayer(in, l1_weights, l1_biases);

  // Second layer
  xt::xarray<double> l2_weights_view = xt::view(total_weights, xt::range(l2_start, l2_end), xt::all()),
    l2_weights = l2_weights_view.reshape({ out_size, hidden_size }),
    l2_biases_view = xt::view(total_weights, xt::range(l2_end, l2_end + out_size), xt::all()),
    l2_biases = l2_biases_view.reshape({ out_size, 1 });
  xt::xarray<double> l2_out = xt::linalg::dot(l2_weights, l1_out) + l2_biases;

  return l2_out;
}

xt::xarray<double>
tripleLayerModel(
    unsigned int in_size,
    unsigned int hidden_size1,
    unsigned int hidden_size2,
    unsigned int out_size,
    xt::xarray<double> &in,
    xt::xarray<double> &total_weights
) {
  unsigned int l1_end = hidden_size1 * in_size,
    l2_start = l1_end + hidden_size1, l2_end = l2_start + hidden_size2 * hidden_size1,
    l3_start = l2_end + hidden_size2, l3_end = l3_start + out_size * hidden_size2;

  // First layer
  xt::xarray<double> l1_weights_view = xt::view(total_weights, xt::range(_, l1_end), xt::all()),
    l1_weights = l1_weights_view.reshape({ hidden_size1, in_size }),
    l1_biases_view = xt::view(total_weights, xt::range(l1_end, l2_start), xt::all()),
    l1_biases = l1_biases_view.reshape({ hidden_size1, 1 });
  xt::xarray<double> l1_out = reluLayer(in, l1_weights, l1_biases);

  // Second layer
  xt::xarray<double> l2_weights_view = xt::view(total_weights, xt::range(l2_start, l2_end), xt::all()),
    l2_weights = l2_weights_view.reshape({ hidden_size2, hidden_size1 }),
    l2_biases_view = xt::view(total_weights, xt::range(l2_end, l2_end + hidden_size2), xt::all()),
    l2_biases = l2_biases_view.reshape({ hidden_size2, 1 });
  xt::xarray<double> l2_out = reluLayer(l1_out, l2_weights, l2_biases);

  // Third layer
  xt::xarray<double> l3_weights_view = xt::view(total_weights, xt::range(l3_start, l3_end), xt::all()),
    l3_weights = l3_weights_view.reshape({ out_size, hidden_size2 }),
    l3_biases_view = xt::view(total_weights, xt::range(l3_end, l3_end + out_size), xt::all()),
    l3_biases = l3_biases_view.reshape({ out_size, 1 });
  xt::xarray<double> l3_out = xt::linalg::dot(l3_weights, l2_out) + l3_biases;

  return l3_out;
}

void RL::cem()
{
  double dou_n_elite;
  modf(cem_batch_size * cem_elite_frac, &dou_n_elite);
  int n_elite = (int)dou_n_elite;

  xt::random::seed(time(NULL));
  xt::xarray<double> th_std = xt::ones_like(this->params) * this->cem_stddev;
  for (size_t i = 0;i < cem_iterations; ++i) {
    xt::xarray<double> noises = th_std * xt::random::randn<double>({ (int)cem_batch_size, (int)(this->params.size()) });
    xt::xarray<double> ths = params + noises;

    std::vector<double> rewards_v;
    for (size_t j = 0;j < ths.shape(0); ++j) {
      this->params = xt::row(ths, j);
      //std::cout << xt::mean(params) << " " << xt::stddev(params) << std::endl;
      rewards_v.push_back(this->reward());
    }

    xt::xarray<double> rewards = xt::argsort(xt::adapt(rewards_v), 1), rev_rewards(rewards.shape());
    std::copy(rewards.crbegin(), rewards.crend(), rev_rewards.begin());
    rev_rewards = xt::view(rev_rewards, xt::range(_, n_elite));

    xt::xarray<double> elite_ths = xt::view(ths, xt::keep(rev_rewards), xt::all());
    this->params = xt::mean(elite_ths, {0});
    //th_std = xt::stddev(elite_ths,{0}) + cem_noise_factor / (i+1);

    //if (i < 30)
    //th_std = xt::stddev(elite_ths,{0}) + cem_noise_factor;
    //else
    th_std = xt::stddev(elite_ths,{0}) + cem_noise_factor / (i+1);

    //std::cout << xt::adapt(params.shape()) << std::endl;
    //std::cout << xt::adapt(th_std.shape()) << std::endl;
    std::cout << "Reward mean in it#" << i << " " << xt::mean(xt::adapt(rewards_v)) << std::endl;
    std::cout << "STD mean in it#" << i << " " << xt::mean(th_std) << std::endl;
    //return;
  }
}

double RL::reward()
{
  double total_reward = 0;

  for (size_t i = 0;i < rl_num_of_tasks; ++i) {
    _obj->runRL();
    total_reward += _obj->scoreRL();
  }

  //std::cout << total_reward << std::endl;
  return total_reward / rl_num_of_tasks;
}

xt::xarray<double> RL::run(xt::xarray<double> &in_data)
{
  if (model_version == 2) {
    xt::xarray<double> out = doubleLayerModel(in_size, hidden_size, out_size, in_data, params);
    return out;
  } else {
    xt::xarray<double> out = tripleLayerModel(in_size, hidden_size1, hidden_size2, out_size, in_data, params);
    return out;
  }
}

void RL::train()
{
  this->initModel();
  this->cem();
}

void RL::initModel()
{
  if (model_version == 2) {
    policy_size = (in_size + 1) * hidden_size + (hidden_size + 1) * out_size;
  } else {
    policy_size = (in_size + 1) * hidden_size1 + (hidden_size1 + 1) * hidden_size2 +  (hidden_size2 + 1) * out_size;    
  }
  this->params = xt::zeros<double>({ policy_size });
}

void RL::switchModel2()
{
  model_version = 2;
}

void RL::switchModel3()
{
  model_version = 3;
}

bool RL::check(const std::string &infile)
{
  if (std::filesystem::exists(infile))
    return true;

  return false;
}

void RL::load(const std::string &infile)
{
  if (check(infile)) {
    std::ifstream ifs(infile);
    auto data = xt::load_csv<double>(ifs);
    this->initModel();
    data.reshape(this->params.shape());
    this->params = data;
  }
}

void RL::save(const std::string &outfile)
{
  std::ofstream ofs(outfile);
  xt::xarray<double> to_be_saved = this->params;
  std::cout << to_be_saved << std::endl;
  to_be_saved.reshape({2, -1});
  xt::dump_csv(ofs, to_be_saved);
}
