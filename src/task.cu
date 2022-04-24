///////////////////////////////////////////////////////////////////////////////
// File:         task.cu
// Description:  providing different tasks such as regression, classification
//               that uses TAGI approach.
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 23, 2022
// Updated:      April 18, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "../include/task.cuh"

///////////////////////////////////////////////////////////////////////
// MISC FUNCTIONS
///////////////////////////////////////////////////////////////////////
void compute_net_memory(Network &net, size_t &id_bytes, size_t &od_bytes,
                        size_t &ode_bytes, size_t &max_n_s_bytes)
/*TODO: Might be removed
 */
{
    id_bytes = net.batch_size * net.nodes[0] * sizeof(float);
    od_bytes = net.batch_size * net.nodes[net.nodes.size() - 1] * sizeof(float);
    ode_bytes = net.batch_size * net.nye * sizeof(int);
    max_n_s_bytes = net.n_max_state * sizeof(float);
}

void get_output_states(std::vector<float> &ma, std::vector<float> Sa,
                       std::vector<float> &ma_output,
                       std::vector<float> &Sa_output, int idx)
/*Get output's distrinution
Args:
    ma: Mean of activation units of the entire network
    ma: Variance of activation units of the entire network
    ma_output: mean of activation units of the output layer
    Sa_output: Variance of activation units of the output layer
    idx: Starting index of the output layer
*/
{
    for (int i = 0; i < ma_output.size(); i++) {
        ma_output[i] = ma[idx + i];
        Sa_output[i] = Sa[idx + i];
    }
}

template <typename T>
void update_vector(std::vector<T> &v, std::vector<T> &new_values, int idx,
                   int w)
/*Save new value to vector.
Args:
    v: Vector of interest
    new_values: Values to be stored in the vector v
    idx: Indices of new value in vector v
*/
{
    int N = new_values.size() / w;
    if (v.size() - idx < new_values.size()) {
        throw std::invalid_argument(
            "Vector capacity is insufficient - task.cu");
    }
    for (int i = 0; i < N; i++) {
        for (int c = 0; c < w; c++) {
            v[idx + i * w + c] = new_values[w * i + c];
        }
    }
}

void get_obs_variance(std::vector<float> Sa, float sv) {
    for (int i = 0; i < Sa.size(); i++) {
        Sa[i] += sv;
    }
}

float compute_average_error_rate(std::vector<int> &error_rate, int curr_idx,
                                 int n_past_data)
/*Compute running error rate.
  Args:
    error_rate: Vector of error rate
    curr_idx: Index of the current error rate
    n_past_data: Number of past data from the current index
*/
{
    int end_idx = curr_idx - n_past_data;
    if (end_idx < 0) {
        end_idx = 0;
        n_past_data = curr_idx;
    }

    float tmp = 0;
    for (int i = 0; i < n_past_data; i++) {
        tmp += error_rate[end_idx + i];
    }

    float avg_error = tmp / n_past_data;

    return avg_error;
}

void initialize_network_to_device(Network &net, IndexOut &idx, NetState &state,
                                  Param &theta, IndexGPU &idx_gpu,
                                  StateGPU &state_gpu, ParamGPU &theta_gpu,
                                  DeltaStateGPU &d_state_gpu,
                                  DeltaParamGPU &d_theta_gpu)
/*Send network's data to device
  Args:
    net: Network properties on CPU
    idx: Indices of network on CPU
    state: Hidden states of network on CPU
    theta: Parameters of network on CPU
    idx_gpu: Indices of network on GPU
    state_gpu: Hidden states of network on GPU
    theta_gpu: Parameters of network on GPU
    d_state_gpu: Updated quantities for hidden states on GPU
    d_theta_gpu: Updated quantites for parameters on GPU
*/
{
    // Data transfer for indices
    idx_gpu.set_values(idx);
    idx_gpu.allocate_cuda_memory();
    idx_gpu.copy_host_to_device(idx);

    // Data transfer for states
    state_gpu.set_values(state);
    state_gpu.allocate_cuda_memory();
    state_gpu.copy_host_to_device(state);

    // Data transfer for parameters
    theta_gpu.set_values(theta.mw.size(), theta.mb.size(), theta.mw_sc.size(),
                         theta.mb_sc.size());
    theta_gpu.allocate_cuda_memory();
    theta_gpu.copy_host_to_device(theta);

    // Data transfer for delta state
    d_state_gpu.set_values(net.n_state, state.msc.size(), state.mdsc.size(),
                           net.n_max_state);
    d_state_gpu.allocate_cuda_memory();
    d_state_gpu.copy_host_to_device();

    // Data transfer for delta parameters
    d_theta_gpu.set_values(theta.mw.size(), theta.mb.size(), theta.mw_sc.size(),
                           theta.mb_sc.size());
    d_theta_gpu.allocate_cuda_memory();
    d_theta_gpu.copy_host_to_device();
}

///////////////////////////////////////////////////////////////////////
// AUTOENCODER
///////////////////////////////////////////////////////////////////////
void autoencoder(Network &net_e, IndexOut &idx_e, NetState &state_e,
                 Param &theta_e, Network &net_d, IndexOut &idx_d,
                 NetState &state_d, Param &theta_d, ImageData &imdb,
                 ImageData &test_imdb, int n_epochs, int n_classes,
                 SavePath &path, bool train_mode, bool debug)
/* Autoencoder network for generating images
   Args:
    net_e: Network properties for encoder
    idx_e: Indices of network for encoder
    state_e: Hidden states of network for encoder
    theta_e: Parameters of network for encoder
    net_d: Network properties for decoder
    idx_d: Indices of network for decoder
    state_d: Hidden states of network for decoder
    theta_d: Parameters of network for decoder
    imdb: Image database
    n_iter: Number of iteration for each epoch
    n_epochs: Number of epochs
    n_classes: Number of classes of image data
    path: Directory stored the final results
    debug: Debugging mode allows saving inference data
 */
{
    // Batch size check
    if (net_e.batch_size != net_d.batch_size) {
        throw std::invalid_argument(
            " Batch size is not equal - Task - Autoencoder");
    }

    // Compute number of data
    int n_iter = 1;  // imdb.num_data / net_d.batch_size;
    int test_n_iter = 1;

    // Input and output layer
    std::vector<float> x_batch, Sx_batch, y_batch, V_batch;
    std::vector<int> data_idx = create_range(imdb.num_data);
    std::vector<int> test_data_idx = create_range(imdb.num_data);
    std::vector<int> batch_idx(net_d.batch_size);
    std::vector<int> idx_ud_batch(net_d.nye * net_d.batch_size, 0);
    std::vector<int> idx_ud_batch_e(net_e.nye * net_e.batch_size, 0);
    std::vector<int> label_batch(net_d.batch_size, 0);

    x_batch.resize(net_e.batch_size * net_e.nodes[0], 0);
    Sx_batch.resize(net_e.batch_size * net_e.nodes[0], 0);
    y_batch.resize(net_d.batch_size * net_d.nodes[net_d.nodes.size() - 1], 0);
    V_batch.resize(net_d.batch_size * net_d.nodes[net_d.nodes.size() - 1],
                   net_d.sigma_v);

    // Transfer data for states of encoder
    IndexGPU idx_e_gpu;
    StateGPU state_e_gpu;
    ParamGPU theta_e_gpu;
    DeltaStateGPU d_state_e_gpu;
    DeltaParamGPU d_theta_e_gpu;
    initialize_network_to_device(net_e, idx_e, state_e, theta_e, idx_e_gpu,
                                 state_e_gpu, theta_e_gpu, d_state_e_gpu,
                                 d_theta_e_gpu);

    // Transfer data for states of decoder
    IndexGPU idx_d_gpu;
    StateGPU state_d_gpu;
    ParamGPU theta_d_gpu;
    DeltaStateGPU d_state_d_gpu;
    DeltaParamGPU d_theta_d_gpu;
    initialize_network_to_device(net_d, idx_d, state_d, theta_d, idx_d_gpu,
                                 state_d_gpu, theta_d_gpu, d_state_d_gpu,
                                 d_theta_d_gpu);

    // Transfer data for input and output
    InputGPU ip_gpu(net_e.nodes[0], net_d.batch_size);
    ip_gpu.allocate_cuda_memory();

    ObsGPU op_e_gpu(net_e.nodes[net_e.nodes.size() - 1], net_e.nye,
                    net_e.batch_size);
    op_e_gpu.allocate_cuda_memory();

    ObsGPU op_d_gpu(net_d.nodes[net_d.nodes.size() - 1], net_d.nye,
                    net_d.batch_size);
    op_d_gpu.allocate_cuda_memory();

    // Loop initialization
    int THREADS = 16;
    unsigned int BLOCKS =
        (net_e.batch_size * net_e.nodes[0] + THREADS - 1) / THREADS;
    unsigned int BLOCKS_D =
        (net_d.batch_size * net_d.nodes[0] + THREADS - 1) / THREADS;

    if (train_mode) {
        for (int e = 0; e < n_epochs; e++) {
            std::cout << "################"
                      << "\n"
                      << std::endl;
            std::cout << "Epoch #" << e << "\n" << std::endl;
            //    auto start_time = std::chrono::steady_clock::now();

            for (int i = 0; i < n_iter; i++) {
                // TODO: Make a cleaner way to handle both cases
                if (i == 0) {
                    net_e.ra_mt = 0.0f;
                    net_d.ra_mt = 0.0f;
                } else {
                    net_e.ra_mt = 0.9f;
                    net_d.ra_mt = 0.9f;
                }

                // Load input data for encoder and output data for decoder
                get_batch_idx(data_idx, i, net_e.batch_size, batch_idx);
                get_batch_data(imdb.images, batch_idx, net_e.nodes[0], x_batch);
                get_batch_data(imdb.labels, batch_idx, 1, label_batch);
                ip_gpu.copy_host_to_device(x_batch, Sx_batch);
                op_d_gpu.copy_host_to_device(x_batch, idx_ud_batch, V_batch);

                // Initialize input of encoder
                initializeStates<<<BLOCKS, THREADS>>>(
                    ip_gpu.d_x_batch, ip_gpu.d_Sx_batch, state_e_gpu.d_mz,
                    state_e_gpu.d_Sz, state_e_gpu.d_ma, state_e_gpu.d_Sa,
                    state_e_gpu.d_J, net_e.batch_size * net_e.nodes[0]);

                // Feed forward for encoder
                feedForward(net_e, theta_e_gpu, idx_e_gpu, state_e_gpu);

                // Initialize the decoder's input. TODO double the position of
                // hidden state for encoder net.
                initializeFullStates<<<BLOCKS_D, THREADS>>>(
                    state_e_gpu.d_mz, state_e_gpu.d_Sz, state_e_gpu.d_ma,
                    state_e_gpu.d_Sa, state_e_gpu.d_J,
                    net_d.nodes[0] * net_d.batch_size,
                    net_e.z_pos[net_e.nodes.size() - 1], state_d_gpu.d_mz,
                    state_d_gpu.d_Sz, state_d_gpu.d_ma, state_d_gpu.d_Sa,
                    state_d_gpu.d_J);

                // Feed forward for decoder
                feedForward(net_d, theta_d_gpu, idx_d_gpu, state_d_gpu);

                // Feed backward for hidden state and parameters of decoder
                stateBackward(net_d, theta_d_gpu, state_d_gpu, idx_d_gpu,
                              op_d_gpu, d_state_d_gpu);
                paramBackward(net_d, theta_d_gpu, state_d_gpu, d_state_d_gpu,
                              idx_d_gpu, d_theta_d_gpu);

                // Update parameter for decoder
                globalParamUpdate(d_theta_d_gpu, theta_d.mw.size(),
                                  theta_d.mb.size(), theta_d.mw_sc.size(),
                                  theta_d.mb_sc.size(), THREADS, theta_d_gpu);

                // TODO: Replace delta_m with delta_mz. Load output data for
                // encoder
                getInputDeltaState<<<BLOCKS_D, THREADS>>>(
                    d_state_d_gpu.d_delta_mz, d_state_d_gpu.d_delta_Sz,
                    net_d.nodes[0] * net_d.batch_size, op_e_gpu.d_y_batch,
                    op_e_gpu.d_V_batch);

                // Feed backward for hidden state and parameters of encoder
                stateBackward(net_e, theta_e_gpu, state_e_gpu, idx_e_gpu,
                              op_e_gpu, d_state_e_gpu);
                paramBackward(net_e, theta_e_gpu, state_e_gpu, d_state_e_gpu,
                              idx_e_gpu, d_theta_e_gpu);

                // Update parameter for encoder
                globalParamUpdate(d_theta_e_gpu, theta_e.mw.size(),
                                  theta_e.mb.size(), theta_e.mw_sc.size(),
                                  theta_e.mb_sc.size(), THREADS, theta_e_gpu);
            }
        }
        //  auto end_time = std::chrono::steady_clock::now();
        //  std::cout << "Time/epoch in seconds: "<<
        //  std::chrono::duration_cast<std::chrono::seconds>(end_time -
        //  start_time).count()<< " sec"<<"\n";

        d_state_e_gpu.copy_device_to_host();
        theta_e_gpu.copy_device_to_host(theta_e);
        d_state_d_gpu.copy_device_to_host();
        theta_d_gpu.copy_device_to_host(theta_d);
        state_e_gpu.copy_device_to_host(state_e);
        state_d_gpu.copy_device_to_host(state_d);
        std::cout << "deltaMz_0"
                  << "\n"
                  << std::endl;
        for (int i = 0; i < 12; i++) {
            std::cout << std::fixed;
            std::cout << std::setprecision(10);
            std::cout << d_state_d_gpu.delta_mz[i] << '\n';
        }
        std::cout << std::endl;

        std::cout << "deltaSz_0"
                  << "\n"
                  << std::endl;
        for (int i = 0; i < theta_e.mw.size(); i++) {
            std::cout << std::fixed;
            std::cout << std::setprecision(10);
            std::cout << theta_e.mw[i] << '\n';
        }
        std::cout << std::endl;

        // Save results
        if (debug) {
            std::string res_path_e = path.debug_path + "/saved_result_enc/";
            save_inference_results(res_path_e, d_state_e_gpu, theta_e);

            std::string res_path_d = path.debug_path + "/saved_result_dec/";
            save_inference_results(res_path_d, d_state_d_gpu, theta_d);
        }
    }

    // // Generate image from test set
    // for (int i = 0; i < test_n_iter; i++) {
    //     // TODO: set momentum for normalization layer when i > i
    //     net_e.ra_mt = 0.0f;
    //     net_d.ra_mt = 0.0f;

    //     // Load input data for encoder and output data for decoder
    //     get_batch_idx(data_idx, i, net_e.batch_size, batch_idx);
    //     get_batch_data(imdb.images, batch_idx, net_e.nodes[0], x_batch);
    //     get_batch_data(imdb.labels, batch_idx, 1, label_batch);
    //     ip_gpu.copy_host_to_device(x_batch, Sx_batch);
    //     op_d_gpu.copy_host_to_device(x_batch, idx_ud_batch, V_batch);

    //     // Initialize input of encoder
    //     initializeStates<<<BLOCKS, THREADS>>>(
    //         ip_gpu.d_x_batch, ip_gpu.d_Sx_batch, state_e_gpu.d_mz,
    //         state_e_gpu.d_Sz, state_e_gpu.d_ma, state_e_gpu.d_Sa,
    //         state_e_gpu.d_J, net_e.batch_size * net_e.nodes[0]);

    //     // Feed forward for encoder
    //     feedForward(net_e, theta_e_gpu, idx_e_gpu, state_e_gpu);

    //     // Initialize the decoder's input. TODO: double the position of
    //     // hidden state for encoder net.
    //     initializeFullStates<<<BLOCKS_D, THREADS>>>(
    //         state_e_gpu.d_mz, state_e_gpu.d_Sz, state_e_gpu.d_ma,
    //         state_e_gpu.d_Sa, state_e_gpu.d_J,
    //         net_d.nodes[0] * net_d.batch_size,
    //         net_e.z_pos[net_e.nodes.size() - 1], state_d_gpu.d_mz,
    //         state_d_gpu.d_Sz, state_d_gpu.d_ma, state_d_gpu.d_Sa,
    //         state_d_gpu.d_J);

    //     // Feed forward for decoder
    //     feedForward(net_d, theta_d_gpu, idx_d_gpu, state_d_gpu);
    // }
    // state_d_gpu.copy_device_to_host(state_d);
}

///////////////////////////////////////////////////////////////////////
// CLASSIFICATION
///////////////////////////////////////////////////////////////////////
void classification(Network &net, IndexOut &idx, NetState &state, Param &theta,
                    ImageData &imdb, ImageData &test_imdb, int n_epochs,
                    int n_classes, SavePath &path, bool train_mode, bool debug)
/*Classification task
  Args:
    Net: Network architecture
    idx: Indices of network
    theta: Weights & biases of network
    imdb: Image database
    n_epochs: Number of epochs
    n_classes: Number of classes of image data
    res_path: Directory stored the final results
    debug: Debugging mode allows saving inference data
 */
{
    // Number of bytes
    size_t id_bytes, od_bytes, ode_bytes, max_n_s_bytes;
    compute_net_memory(net, id_bytes, od_bytes, ode_bytes, max_n_s_bytes);

    // Input and output layer
    auto hrs = class_to_obs(n_classes);
    std::vector<float> x_batch, Sx_batch, y_batch, V_batch;
    std::vector<int> data_idx = create_range(imdb.num_data);
    std::vector<int> test_data_idx = create_range(test_imdb.num_data);
    std::vector<int> batch_idx(net.batch_size);
    std::vector<int> idx_ud_batch(net.nye * net.batch_size, 0);
    std::vector<int> label_batch(net.batch_size, 0);

    x_batch.resize(net.batch_size * net.nodes[0], 0);
    Sx_batch.resize(net.batch_size * net.nodes[0], 0);
    y_batch.resize(net.batch_size * hrs.n_obs, 0);
    V_batch.resize(net.batch_size * hrs.n_obs, net.sigma_v);

    IndexGPU idx_gpu;
    StateGPU state_gpu;
    ParamGPU theta_gpu;
    DeltaStateGPU d_state_gpu;
    DeltaParamGPU d_theta_gpu;

    initialize_network_to_device(net, idx, state, theta, idx_gpu, state_gpu,
                                 theta_gpu, d_state_gpu, d_theta_gpu);

    // Data transfer for input and output data
    InputGPU ip_gpu(net.nodes[0], net.batch_size);
    ip_gpu.allocate_cuda_memory();

    ObsGPU op_gpu(net.nodes[net.nodes.size() - 1], net.nye, net.batch_size);
    op_gpu.allocate_cuda_memory();

    // Initialization
    int wN = theta.mw.size();
    int bN = theta.mb.size();
    int wN_sc = theta.mw_sc.size();
    int bN_sc = theta.mb_sc.size();

    int THREADS = 16;
    unsigned int BLOCKS =
        (net.batch_size * net.nodes[0] + THREADS - 1) / THREADS;
    int mt_idx = 0;
    int n_iter = 1;       // imdb.num_data / net.batch_size;
    int test_n_iter = 1;  // test_imdb.num_data / net.batch_size;

    // Error rate for training
    std::vector<int> error_rate(imdb.num_data, 0);
    std::vector<float> prob_class(imdb.num_data * n_classes);
    std::vector<int> error_rate_batch;
    std::vector<float> prob_class_batch;

    // Error rate for testing
    std::vector<int> test_epoch_error_rate(n_epochs, 0);
    std::vector<int> test_error_rate(test_imdb.num_data, 0);
    std::vector<float> prob_class_test(test_imdb.num_data * n_classes);
    std::vector<float> ma_output(
        net.batch_size * net.nodes[net.nodes.size() - 1], 0);
    std::vector<float> Sa_output(
        net.batch_size * net.nodes[net.nodes.size() - 1], 0);

    for (int e = 0; e < n_epochs; e++) {
        if (train_mode) {
            std::cout << "%%%%%%%%%%"
                      << "\n"
                      << std::endl;
            std::cout << "Epoch #" << e << "\n" << std::endl;
            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < n_iter; i++) {
                // TODO: Make a cleaner way to handle both cases
                if (i == 0) {
                    net.ra_mt = 0.0f;
                } else {
                    net.ra_mt = 0.9f;
                }

                // Load data
                get_batch_idx(data_idx, i, net.batch_size, batch_idx);
                get_batch_data(imdb.images, batch_idx, net.nodes[0], x_batch);
                get_batch_data(imdb.obs_label, batch_idx, hrs.n_obs, y_batch);
                get_batch_data(imdb.obs_idx, batch_idx, hrs.n_obs,
                               idx_ud_batch);
                get_batch_data(imdb.labels, batch_idx, 1, label_batch);
                ip_gpu.copy_host_to_device(x_batch, Sx_batch);
                op_gpu.copy_host_to_device(y_batch, idx_ud_batch, V_batch);

                // Initialize input
                initializeStates<<<BLOCKS, THREADS>>>(
                    ip_gpu.d_x_batch, ip_gpu.d_Sx_batch, state_gpu.d_mz,
                    state_gpu.d_Sz, state_gpu.d_ma, state_gpu.d_Sa,
                    state_gpu.d_J, net.batch_size * net.nodes[0]);

                // Feed forward. TODO:  Need to update mra_prev and Sra_prev
                // in state GPU
                feedForward(net, theta_gpu, idx_gpu, state_gpu);

                // Feed backward for hidden states
                stateBackward(net, theta_gpu, state_gpu, idx_gpu, op_gpu,
                              d_state_gpu);

                // Feed backward for parameters
                paramBackward(net, theta_gpu, state_gpu, d_state_gpu, idx_gpu,
                              d_theta_gpu);

                // Update model parameters. TODO: Double check if we need to
                // duplicate parameters when updating
                globalParamUpdate(d_theta_gpu, wN, bN, wN_sc, bN_sc, THREADS,
                                  theta_gpu);

                // Compute error rate
                state_gpu.copy_device_to_host(state);
                get_output_states(state.ma, state.Sa, ma_output, Sa_output,
                                  net.z_pos[net.nodes.size() - 1]);
                std::tie(error_rate_batch, prob_class_batch) =
                    get_error(ma_output, Sa_output, label_batch, hrs, n_classes,
                              net.batch_size);
                mt_idx = i * net.batch_size;
                update_vector(error_rate, error_rate_batch, mt_idx, 1);
                // update_vector(prob_class, prob_class_batch, mt_idx,
                // n_classes);

                if (i % 100 == 0) {
                    int curr_idx = mt_idx + net.batch_size;
                    auto avg_error =
                        compute_average_error_rate(error_rate, curr_idx, 100);
                    std::cout << "#############"
                              << "\n";
                    std::cout << "Error rate: ";
                    std::cout << std::fixed;
                    std::cout << std::setprecision(10);
                    std::cout << avg_error << "\n" << std::endl;
                }
            }
            auto end = std::chrono::steady_clock::now();
            std::cout << "Elapsed time in seconds: "
                      << std::chrono::duration_cast<std::chrono::seconds>(end -
                                                                          start)
                             .count()
                      << " sec"
                      << "\n";
        } else {
            for (int i = 0; i < test_n_iter; i++) {
                // TODO: set = 0.9 when i > 0
                net.ra_mt = 0.0f;

                // Load data
                get_batch_idx(data_idx, i, net.batch_size, batch_idx);
                get_batch_data(test_imdb.images, batch_idx, net.nodes[0],
                               x_batch);
                get_batch_data(test_imdb.obs_label, batch_idx, hrs.n_obs,
                               y_batch);
                get_batch_data(test_imdb.obs_idx, batch_idx, hrs.n_obs,
                               idx_ud_batch);
                get_batch_data(test_imdb.labels, batch_idx, 1, label_batch);
                ip_gpu.copy_host_to_device(x_batch, Sx_batch);
                op_gpu.copy_host_to_device(y_batch, idx_ud_batch, V_batch);

                // Initialize input
                initializeStates<<<BLOCKS, THREADS>>>(
                    ip_gpu.d_x_batch, ip_gpu.d_Sx_batch, state_gpu.d_mz,
                    state_gpu.d_Sz, state_gpu.d_ma, state_gpu.d_Sa,
                    state_gpu.d_J, net.batch_size * net.nodes[0]);

                // Feed forward. TODO:  Need to update mra_prev and Sra_prev
                // in state GPU
                feedForward(net, theta_gpu, idx_gpu, state_gpu);

                // Compute error rate
                state_gpu.copy_device_to_host(state);
                get_output_states(state.ma, state.Sa, ma_output, Sa_output,
                                  net.z_pos[net.nodes.size() - 1]);
                std::tie(error_rate_batch, prob_class_batch) =
                    get_error(ma_output, Sa_output, label_batch, hrs, n_classes,
                              net.batch_size);
                mt_idx = i * net.batch_size;
                update_vector(test_error_rate, error_rate_batch, mt_idx, 1);
                // update_vector(prob_class, prob_class_batch, mt_idx,
                // n_classes);
            }
        }
        auto test_avg_error = compute_average_error_rate(
            test_error_rate, test_imdb.num_data, test_imdb.num_data);
        test_epoch_error_rate[e] = test_avg_error;
    }
    d_state_gpu.copy_device_to_host();
    theta_gpu.copy_device_to_host(theta);
    std::cout << "prob"
              << "\n"
              << std::endl;
    for (int i = 0; i < prob_class_batch.size(); i++) {
        std::cout << prob_class_batch[i] << '\n';
    }
    std::cout << std::endl;

    // Save debugging data
    if (debug) {
        save_inference_results(path.saved_inference_path, d_state_gpu, theta);
    }
}

///////////////////////////////////////////////////////////////////////
// REGRESSION
///////////////////////////////////////////////////////////////////////
void regression(Network &net, IndexOut &idx, NetState &state, Param &theta,
                std::vector<float> &x, std::vector<float> &y, int n_iter,
                int n_epochs)
/*Classification task
Args:
    Net: Network architecture
    idx: Indices of network
    theta: Weights & biases of network
    x: Covariates i.e. input
    y: Observations i.e. output
    n_iter: Number of iteration for each epoch
    n_epochs: Number of epochs
    n_classes: Number of classes of image data
    res_path: Directory stored the final results
*/
{
    // Compute number of data
    int n_data = x.size() / net.nodes[0];

    // Number of bytes
    size_t id_bytes, od_bytes, ode_bytes, max_n_s_bytes;
    compute_net_memory(net, id_bytes, od_bytes, ode_bytes, max_n_s_bytes);

    // Copie data
    std::vector<float> x_batch, Sx_batch, y_batch, V_batch;
    std::vector<int> data_idx = create_range(n_data);
    std::vector<int> batch_idx(net.batch_size);
    std::vector<int> idx_ud_batch(net.nye * net.batch_size, 0);

    x_batch.resize(net.batch_size * net.nodes[0], 0);
    Sx_batch.resize(net.batch_size * net.nodes[0], 0);
    y_batch.resize(net.batch_size * net.nodes[net.nodes.size() - 1], 0);
    V_batch.resize(net.batch_size * net.nodes[net.nodes.size() - 1],
                   net.sigma_v);

    // Data transfer
    StateGPU state_gpu;
    ParamGPU theta_gpu;
    IndexGPU idx_gpu;
    DeltaStateGPU d_state_gpu;
    DeltaParamGPU d_theta_gpu;

    initialize_network_to_device(net, idx, state, theta, idx_gpu, state_gpu,
                                 theta_gpu, d_state_gpu, d_theta_gpu);

    // Data transfer for input and output data
    InputGPU ip_gpu(net.nodes[0], net.batch_size);
    ip_gpu.allocate_cuda_memory();

    ObsGPU op_gpu(net.nodes[net.nodes.size() - 1], net.nye, net.batch_size);
    op_gpu.allocate_cuda_memory();

    int wN = theta.mw.size();
    int bN = theta.mb.size();
    int wN_sc = theta.mw_sc.size();
    int bN_sc = theta.mb_sc.size();

    int THREADS = 16;
    unsigned int BLOCKS =
        (net.batch_size * net.nodes[0] + THREADS - 1) / THREADS;

    for (int e = 0; e < n_epochs; e++) {
        for (int i = 0; i < n_iter; i++) {
            // Load data
            get_batch_idx(data_idx, i, net.batch_size, batch_idx);
            get_batch_data(x, batch_idx, net.nodes[0], x_batch);
            get_batch_data(y, batch_idx, net.nodes[net.nodes.size() - 1],
                           y_batch);
            ip_gpu.copy_host_to_device(x_batch, Sx_batch);
            op_gpu.copy_host_to_device(y_batch, idx_ud_batch, V_batch);

            for (int k = 0; k < y_batch.size(); k++) {
                std::cout << y_batch[k] << "\n";
            }
            std::cout << std::endl;

            // Initialize input
            initializeStates<<<BLOCKS, THREADS>>>(
                ip_gpu.d_x_batch, ip_gpu.d_Sx_batch, state_gpu.d_mz,
                state_gpu.d_Sz, state_gpu.d_ma, state_gpu.d_Sa, state_gpu.d_J,
                net.batch_size * net.nodes[0]);

            // Feed forward
            feedForward(net, theta_gpu, idx_gpu, state_gpu);

            // Feed backward for hidden states
            stateBackward(net, theta_gpu, state_gpu, idx_gpu, op_gpu,
                          d_state_gpu);

            // Feed backward for parameters
            paramBackward(net, theta_gpu, state_gpu, d_state_gpu, idx_gpu,
                          d_theta_gpu);

            // Update model parameters
            globalParamUpdate(d_theta_gpu, wN, bN, wN_sc, bN_sc, THREADS,
                              theta_gpu);
        }
    }
    state_gpu.copy_device_to_host(state);
    theta_gpu.copy_device_to_host(theta);
    std::cout << "delta_m"
              << "\n"
              << std::endl;
    for (int i = 0; i < state.ma.size(); i++) {
        std::cout << state.ma[i] << '\n';
    }
    std::cout << std::endl;

    std::cout << "mw"
              << "\n"
              << std::endl;
    for (int i = 0; i < theta.mw.size(); i++) {
        std::cout << theta.mw[i] << '\n';
    }
    std::cout << std::endl;
}

///////////////////////////////////////////////////////////////////////
// TASK MAIN
///////////////////////////////////////////////////////////////////////
void set_task(std::string &user_input_file, SavePath &path) {
    /* Assign different tasks and its parameters

    Args:
        user_input_file: User-specified inputs
        res_path: Directory path where results are stored under *.csv file
    */
    auto user_input = load_userinput(user_input_file);

    if (user_input.task_name == "classification") {
        // Network
        bool train_mode = true;
        IndexOut idx;
        Network net;
        Param theta;
        NetState state;
        net_init(user_input.encoder_net_name, net, theta, state, idx);

        // Data
        auto hrs = class_to_obs(user_input.num_classes);
        auto imdb = get_images(user_input.data_name, user_input.x_train_dir,
                               user_input.y_train_dir, user_input.mu,
                               user_input.sigma, net.widths[0], net.heights[0],
                               net.filters[0], hrs, user_input.num_train_data);
        auto test_imdb = get_images(
            user_input.data_name, user_input.x_test_dir, user_input.y_test_dir,
            user_input.mu, user_input.sigma, net.widths[0], net.heights[0],
            net.filters[0], hrs, user_input.num_test_data);

        // Load param
        if (user_input.load_param) {
            load_net_param(user_input.model_name, user_input.net_name,
                           path.saved_param_path, theta);
        }

        // Saved debug data
        if (user_input.debug) {
            std::string param_path = path.debug_path + "/saved_param/";
            std::string idx_path = path.debug_path + "/saved_idx/";
            save_net_prop(param_path, idx_path, theta, idx);
        }

        classification(net, idx, state, theta, imdb, test_imdb,
                       user_input.num_epochs, user_input.num_classes, path,
                       train_mode, user_input.debug);

        // Save net's parameters
        save_net_param(user_input.model_name, user_input.net_name,
                       path.saved_param_path, theta);

    } else if (user_input.task_name == "autoencoder") {
        // Encoder
        IndexOut idx_e;
        Network net_e;
        Param theta_e;
        NetState state_e;
        net_init(user_input.encoder_net_name, net_e, theta_e, state_e, idx_e);
        net_e.is_output_ud = false;

        // Decoder
        IndexOut idx_d;
        Network net_d;
        Param theta_d;
        NetState state_d;
        net_init(user_input.decoder_net_name, net_d, theta_d, state_d, idx_d);
        net_d.is_idx_ud = false;
        // It eable to infer the input's hidden states
        net_d.last_backward_layer = 0;

        // Load data
        auto hrs = class_to_obs(user_input.num_classes);
        auto imdb =
            get_images(user_input.data_name, user_input.x_train_dir,
                       user_input.y_train_dir, user_input.mu, user_input.sigma,
                       net_e.widths[0], net_e.heights[0], net_e.filters[0], hrs,
                       user_input.num_train_data);
        auto test_imdb = get_images(
            user_input.data_name, user_input.x_test_dir, user_input.y_test_dir,
            user_input.mu, user_input.sigma, net_e.widths[0], net_e.heights[0],
            net_e.filters[0], hrs, user_input.num_test_data);

        // Load param
        if (user_input.load_param) {
            load_net_param(user_input.model_name, user_input.encoder_net_name,
                           path.saved_param_path, theta_e);
            load_net_param(user_input.model_name, user_input.decoder_net_name,
                           path.saved_param_path, theta_d);
        }

        // Save data for debugging
        save_autoencoder_net_prop(theta_e, theta_d, idx_e, idx_d,
                                  path.debug_path);

        // Train network
        bool train_mode = true;
        autoencoder(net_e, idx_e, state_e, theta_e, net_d, idx_d, state_d,
                    theta_d, imdb, test_imdb, user_input.num_epochs,
                    user_input.num_classes, path, train_mode, user_input.debug);

        save_net_param(user_input.model_name, user_input.encoder_net_name,
                       path.saved_param_path, theta_e);
        save_net_param(user_input.model_name, user_input.decoder_net_name,
                       path.saved_param_path, theta_d);

    } else if (user_input.task_name == "regression") {
        // Data
        std::vector<float> x = {1, 2, 3, 4};
        std::vector<float> y = {2.2};
        int n_iter = 1;
        int n_epochs = 1;

        // Network
        IndexOut idx;
        Network net;
        Param theta;
        NetState state;
        net_init(user_input.encoder_net_name, net, theta, state, idx);

        // Load param
        if (user_input.load_param) {
            load_net_param(user_input.model_name, user_input.net_name,
                           path.saved_param_path, theta);
        }

        // Save network's parameter to debug data
        std::string param_path = path.debug_path + "/saved_param/";
        save_param(param_path, theta);

        // Run regression
        regression(net, idx, state, theta, x, y, n_iter, n_epochs);

        // Save net's parameters
        save_net_param(user_input.model_name, user_input.net_name,
                       path.saved_param_path, theta);
    } else {
        throw std::invalid_argument("Task name does not exist - task.cu");
    }
}