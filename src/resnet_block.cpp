
#include "../include/resnet_block.h"

#include "../include/custom_logger.h"
#ifdef USE_CUDA
#include "../include/resnet_block_cuda.cuh"
#endif

void add_shortcut_mean_var(const std::vector<float> &mu_s,
                           const std::vector<float> &var_s, int num_states,
                           std::vector<float> &mu_a, std::vector<float> &var_a)
/*
 */
{
    for (int i = 0; i < num_states; i++) {
        mu_a[i] += mu_s[i];
        var_a[i] += var_s[i];
    }
}

void add_shortcut_delta(const std::vector<float> &mu_s,
                        const std::vector<float> &var_s,
                        const std::vector<float> &jcb_s, int num_states,
                        std::vector<float> &mu_a, std::vector<float> &var_a)
/*
 */
{
    for (int i = 0; i < num_states; i++) {
        mu_a[i] += mu_s[i] * jcb_s[i];
        var_a[i] += var_s[i] * jcb_s[i] * jcb_s[i];
    }
}

ResNetBlock::~ResNetBlock() {}

std::string ResNetBlock::get_layer_info() const
/*
 */
{
    return "ResNetBlock(" + std::to_string(this->input_size) + "," +
           std::to_string(this->output_size) + ")";
}

std::string ResNetBlock::get_layer_name() const
/*
 */
{
    return "ResNetBlock";
}

LayerType ResNetBlock::get_layer_type() const
/*
 */
{
    return LayerType::ResNetBlock;
}

int ResNetBlock::get_max_num_states()
/**/
{
    auto max_main_block = this->main_block->get_max_num_states();
    int max_shortcut = 0;
    if (this->shortcut != nullptr) {
        max_shortcut = this->shortcut->get_max_num_states();
    }

    return std::max(max_main_block, max_shortcut);
}

std::string ResNetBlock::get_device()
/*
 */
{
    auto main_block_device = this->main_block->get_device();
    std::string resnet_block_device =
        this->device + ":" + std::to_string(this->device_idx);
    if (main_block_device != resnet_block_device) {
        LOG(LogLevel::ERROR, "Main block device [" + main_block_device +
                                 "] does not match block device [" +
                                 resnet_block_device + "]");
    }
    if (this->shortcut != nullptr) {
        auto shortcut_device = this->shortcut->get_device();
        if (shortcut_device != resnet_block_device) {
            LOG(LogLevel::ERROR, "Shortcut device [" + shortcut_device +
                                     "] does not match block device [" +
                                     resnet_block_device + "]");
        }
    }
    return resnet_block_device;
}

void ResNetBlock::compute_input_output_size(const InitArgs &args)
/*
 */
{
    this->in_channels = args.depth;
    this->in_height = args.height;
    this->in_width = args.width;

    this->main_block->compute_input_output_size(args);
    if (this->shortcut != nullptr) {
        this->shortcut->compute_input_output_size(args);
    }

    this->out_channels = this->main_block->out_channels;
    this->out_height = this->main_block->out_height;
    this->out_width = this->main_block->out_width;

    this->input_size = this->in_width * this->in_width * this->in_channels;
    this->output_size = this->out_width * this->out_height * this->out_channels;
}

void ResNetBlock::init_shortcut_state()
/*
 */
{
    int max_num_states = this->shortcut->get_max_num_states();
    int size = max_num_states * this->_batch_size;
    this->shortcut_output_z =
        std::make_shared<BaseHiddenStates>(size, this->_batch_size);
}

void ResNetBlock::init_shortcut_delta_state()
/*
 */
{
    int max_num_states = this->shortcut->get_max_num_states();
    int size = max_num_states * this->_batch_size;
    this->shortcut_output_delta_z =
        std::make_shared<BaseDeltaStates>(size, this->_batch_size);
}

void ResNetBlock::init_input_buffer()
/*
 */
{
    int max_num_states = this->input_size;
    if (this->shortcut != nullptr) {
        max_num_states = this->shortcut->get_max_num_states();
    }
    int size = max_num_states * this->_batch_size;
    this->input_z = std::make_shared<BaseHiddenStates>(size, this->_batch_size);
    this->input_delta_z =
        std::make_shared<BaseDeltaStates>(size, this->_batch_size);
}

void ResNetBlock::init_weight_bias()
/*
 */
{
    this->main_block->init_weight_bias();
    if (this->shortcut != nullptr) {
        this->shortcut->init_weight_bias();
    }
}

void ResNetBlock::set_threads(int num)
/*
 */
{
    this->main_block->set_threads(num);
    if (this->shortcut != nullptr) {
        this->shortcut->set_threads(num);
    }
}

void ResNetBlock::train()
/*
 */
{
    this->main_block->train();
    if (this->shortcut != nullptr) {
        this->shortcut->train();
    }
}

void ResNetBlock::eval()
/*
 */
{
    this->main_block->eval();
    if (this->shortcut != nullptr) {
        this->shortcut->eval();
    }
}

void ResNetBlock::forward(BaseHiddenStates &input_states,
                          BaseHiddenStates &output_states,
                          BaseTempStates &temp_states)
/**/

{
    int batch_size = input_states.block_size;
    // Main block
    if (batch_size != this->_batch_size) {
        this->_batch_size = batch_size;
        this->init_input_buffer();
        if (this->shortcut != nullptr) {
            this->init_shortcut_state();
            if (this->training) {
                this->init_shortcut_delta_state();
            }
        }
    }
    // Store jacobian matrix for backward pass
    if (this->training) {
        int act_size = input_states.actual_size * input_states.block_size;
        if (this->bwd_states->size != act_size) {
            this->allocate_bwd_vector(act_size);
        }
        this->fill_bwd_vector(input_states);
    }

    // Make a copy of input states for residual connection
    this->input_z->copy_from(input_states, this->input_size * batch_size);

    this->main_block->forward(input_states, output_states, temp_states);
    int num_states = output_states.block_size * this->output_size;

    // Shortcut
    if (this->shortcut != nullptr) {
        this->shortcut->forward(*this->input_z, *this->shortcut_output_z,
                                temp_states);

        add_shortcut_mean_var(this->shortcut_output_z->mu_a,
                              this->shortcut_output_z->var_a, num_states,
                              output_states.mu_a, output_states.var_a);

    } else {
        add_shortcut_mean_var(this->input_z->mu_a, this->input_z->var_a,
                              num_states, output_states.mu_a,
                              output_states.var_a);
    }

    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;

    // Fill jacobian matrix for output with ones
    if (this->training) {
        this->fill_output_states(output_states);
    }
}

void ResNetBlock::backward(BaseDeltaStates &input_delta_states,
                           BaseDeltaStates &output_delta_states,
                           BaseTempStates &temp_states, bool state_update)
/**/
{
    // Make a copy of delta input used later for residual connection
    this->input_delta_z->copy_from(
        input_delta_states, this->output_size * input_delta_states.block_size);

    this->main_block->backward(input_delta_states, output_delta_states,
                               temp_states, state_update);

    int num_states = output_delta_states.block_size * this->input_size;

    if (this->shortcut != nullptr) {
        this->shortcut->backward(*this->input_delta_z,
                                 *this->shortcut_output_delta_z, temp_states,
                                 state_update);
        add_shortcut_mean_var(this->shortcut_output_delta_z->delta_mu,
                              this->shortcut_output_delta_z->delta_var,
                              num_states, output_delta_states.delta_mu,
                              output_delta_states.delta_var);

    } else {
        add_shortcut_delta(
            this->input_delta_z->delta_mu, this->input_delta_z->delta_var,
            this->bwd_states->jcb, num_states, output_delta_states.delta_mu,
            output_delta_states.delta_var);
    }
}

void ResNetBlock::update_weights()
/*
 */
{
    this->main_block->update_weights();
    if (this->shortcut != nullptr) {
        this->shortcut->update_weights();
    }
}

void ResNetBlock::update_biases()
/*
 */
{
    this->main_block->update_biases();
    if (this->shortcut != nullptr) {
        this->shortcut->update_biases();
    }
}

void ResNetBlock::save(std::ofstream &file)
/*
 */
{
    this->main_block->save(file);
    if (this->shortcut != nullptr) {
        this->shortcut->save(file);
    }
}

void ResNetBlock::load(std::ifstream &file)
/*
 */
{
    this->main_block->load(file);
    if (this->shortcut != nullptr) {
        this->shortcut->load(file);
    }
}

ParameterMap ResNetBlock::get_parameters_as_map(std::string suffix) {
    std::string main_suffix = "main." + suffix;
    ParameterMap params = this->main_block->get_parameters_as_map(main_suffix);
    if (this->shortcut != nullptr) {
        std::string shortcut_suffix = "shortcut." + suffix;
        auto shortcut_params =
            this->shortcut->get_parameters_as_map(shortcut_suffix);
        params.insert(shortcut_params.begin(), shortcut_params.end());
    }
    return params;
}

void ResNetBlock::load_parameters_from_map(const ParameterMap &param_map,
                                           const std::string &suffix) {
    std::string main_suffix = "main." + suffix;
    this->main_block->load_parameters_from_map(param_map, main_suffix);
    if (this->shortcut != nullptr) {
        std::string shortcut_suffix = "shortcut." + suffix;
        this->shortcut->load_parameters_from_map(param_map, shortcut_suffix);
    }
}

std::vector<ParameterTuple> ResNetBlock::parameters() {
    std::vector<ParameterTuple> params = this->main_block->parameters();
    if (this->shortcut != nullptr) {
        auto shortcut_params = this->shortcut->parameters();
        params.insert(params.end(), shortcut_params.begin(),
                      shortcut_params.end());
    }
    return params;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> ResNetBlock::to_cuda(int device_idx) {
    this->device = "cuda";

    // TODO: pass device_idx through main_block and shortcut might not be
    // optimal. Using "to()" is much more elegant.
    this->device_idx = device_idx;
    this->main_block->to(device_idx);
    if (this->shortcut != nullptr) {
        this->shortcut->to(device_idx);
    }
    return std::make_unique<ResNetBlockCuda>(this->main_block, this->shortcut);
}
#endif

void ResNetBlock::preinit_layer() {
    this->main_block->preinit_layer();

    if (this->shortcut != nullptr) {
        this->shortcut->preinit_layer();
    }
}

// DEBUG
std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>,
           std::vector<std::vector<float>>, std::vector<std::vector<float>>>
ResNetBlock::get_norm_mean_var() {
    std::vector<std::vector<float>> mu_ras, var_ras, mu_norms, var_norms;
    std::tie(mu_ras, var_ras, mu_norms, var_norms) =
        this->main_block->get_norm_mean_var();

    if (this->shortcut != nullptr) {
        std::vector<std::vector<float>> mu_ra, var_ra, mu_norm, var_norm;
        std::tie(mu_ra, var_ra, mu_norm, var_norm) =
            this->shortcut->get_norm_mean_var();
        for (size_t i = 0; i < mu_ra.size(); i++) {
            mu_ras.push_back(mu_ra[i]);
            var_ras.push_back(var_ra[i]);
            mu_norms.push_back(mu_norm[i]);
            var_norms.push_back(var_norm[i]);
        }
    }

    return {mu_ras, var_ras, mu_norms, var_norms};
}
