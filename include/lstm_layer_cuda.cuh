#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "base_layer.h"
#include "base_layer_cuda.cuh"
#include "data_struct_cuda.cuh"

class LSTMCuda : public BaseLayerCuda {
   public:
    int seq_len = 1;
    int _batch_size = -1;
    float act_omega = 0.0000001f;
    float gain_w;
    float gain_b;
    std::string init_method;
    int w_pos_f, b_pos_f, w_pos_i, b_pos_i, w_pos_c, b_pos_c, w_pos_o, b_pos_o;

    LSTMStateCuda lstm_state;

    LSTMCuda(size_t input_size, size_t output_size, int seq_len,
             bool bias = true, float gain_w = 1.0f, float gain_b = 1.0f,
             std::string init_method = "Xavier", int device_idx = 0);

    ~LSTMCuda();

    // NOTE: Make the class movable not copyable
    // Delete copy constructor and copy assignment
    LSTMCuda(const LSTMCuda &) = delete;
    LSTMCuda &operator=(const LSTMCuda &) = delete;

    // Optionally implement move constructor and move assignment
    LSTMCuda(LSTMCuda &&) = default;
    LSTMCuda &operator=(LSTMCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    int get_input_size() override;

    int get_output_size() override;

    void get_number_param();

    void init_weight_bias() override;

    void prepare_input(BaseHiddenStates &input_state);

    void forget_gate(int batch_size);

    void input_gate(int batch_size);

    void cell_state_gate(int batch_size);

    void output_gate(int batch_size);

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states,
                  bool state_udapte = true) override;

    std::unique_ptr<BaseLayer> to_host() override;

    void preinit_layer() override;

    void d_get_LSTM_states(std::vector<float> &mu_h, std::vector<float> &var_h,
                           std::vector<float> &mu_c,
                           std::vector<float> &var_c) const;

    void d_set_LSTM_states(const std::vector<float> &mu_h,
                           const std::vector<float> &var_h,
                           const std::vector<float> &mu_c,
                           const std::vector<float> &var_c);
    void to(int device_idx) override;

   protected:
    using BaseLayerCuda::allocate_param_memory;
    using BaseLayerCuda::params_to_device;
};
