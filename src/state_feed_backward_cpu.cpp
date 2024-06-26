///////////////////////////////////////////////////////////////////////////
// File:         state_feed_backward_cpu.cpp
// Description:  CPU version for backward pass for hidden state
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 18, 2022
// Updated:      March 01, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////

#include "../include/state_feed_backward_cpu.h"

void delta_mzSz_with_indices(std::vector<float> &ma, std::vector<float> &Sa,
                             std::vector<float> &Sz, std::vector<float> &J,
                             std::vector<float> &y, std::vector<float> &Sv,
                             std::vector<int> &udIdx, int zpos, int ny, int nye,
                             int n, std::vector<float> &delta_mz,
                             std::vector<float> &delta_Sz)
/* Update output layer based on selected indices.

Args:
    Sz: Variance of hidden states
    ma: Mean of activation units
    Sa: Variance of activation units
    J: Jacobian vector
    y: Observation
    Sv: Observation noise
    udIdx: Selected indiced to update
    delta_mz: Updated quantities for the mean of output's hidden states
    delta_Sz: Updated quantities for the varaince of output's hidden states
    z_pos: Hidden state's position for output layer
    ny: Size of the output layer
    nye: Number of observation to be updated
    n: Number of batches x size of output layer
 */
{
    float zeroPad = 0;
    float tmp = 0;
    int idx = 0;
    for (int col = 0; col < n; col++) {
        // minus 1 due to matlab's indexing
        idx = udIdx[col] + (col / nye) * ny - 1;
        tmp = (J[idx + zpos] * Sz[idx + zpos]) / (Sa[idx + zpos] + Sv[col]);
        if (isinf(tmp) || isnan(tmp)) {
            delta_mz[idx] = zeroPad;
            delta_Sz[idx] = zeroPad;
        } else {
            delta_mz[idx] = tmp * (y[col] - ma[idx + zpos]);
            delta_Sz[idx] = -tmp * (J[idx + zpos] * Sz[idx + zpos]);
        }
    }
}

void delta_mzSz(std::vector<float> &ma, std::vector<float> &Sa,
                std::vector<float> &Sz, std::vector<float> &J,
                std::vector<float> &y, std::vector<float> &Sv, int z_pos, int n,
                std::vector<float> &delta_mz, std::vector<float> &delta_Sz) {
    float zeroPad = 0;
    float tmp = 0;
    for (int col = 0; col < n; col++) {
        tmp = (J[col + z_pos] * Sz[col + z_pos]) / (Sa[col + z_pos] + Sv[col]);
        if (isinf(tmp) || isnan(tmp)) {
            delta_mz[col] = zeroPad;
            delta_Sz[col] = zeroPad;
        } else {
            delta_mz[col] = tmp * (y[col] - ma[col + z_pos]);
            delta_Sz[col] = -tmp * (J[col + z_pos] * Sz[col + z_pos]);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
/// REMAX
////////////////////////////////////////////////////////////////////////////////
void delta_z_y_check_cpu(std::vector<float> &mu_a, std::vector<float> &var_a,
                         std::vector<float> &cov_y_y_check,
                         std::vector<float> &y, std::vector<float> &var_noise,
                         int no, int B, int z_pos,
                         std::vector<float> &delta_mu_zy_check,
                         std::vector<float> &delta_var_zy_check)
/*Compute updating quantities for \check{y}*/
{
    float tmp = 0, zero_pad = 0;
    int col;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < no; j++) {
            col = i * no + j;
            tmp = cov_y_y_check[col] / (var_a[col + z_pos] + var_noise[col]);
            if (isinf(tmp) || isnan(tmp)) {
                delta_mu_zy_check[col] = zero_pad;
                delta_var_zy_check[col] = zero_pad;
            } else {
                delta_mu_zy_check[col] = tmp * (y[col] - mu_a[col + z_pos]);
                delta_var_zy_check[col] = -tmp * cov_y_y_check[col];
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////
/// MULTITHREAD VERSION
//////////////////////////////////////////////////////////////////////
void delta_mzSz_with_indices_worker(
    std::vector<float> &ma, std::vector<float> &Sa, std::vector<float> &Sz,
    std::vector<float> &J, std::vector<float> &y, std::vector<float> &Sv,
    std::vector<int> &udIdx, int z_pos, int ny, int nye, int start_idx,
    int end_idx, std::vector<float> &delta_mz, std::vector<float> &delta_Sz)

{
    float zeroPad = 0;
    float tmp = 0;
    int idx = 0;
    for (int col = start_idx; col < end_idx; col++) {
        // minus 1 due to matlab's indexing
        idx = udIdx[col] + (col / nye) * ny - 1;
        tmp = (J[idx + z_pos] * Sz[idx + z_pos]) / (Sa[idx + z_pos] + Sv[col]);
        if (isinf(tmp) || isnan(tmp)) {
            delta_mz[idx] = zeroPad;
            delta_Sz[idx] = zeroPad;
        } else {
            delta_mz[idx] = tmp * (y[col] - ma[idx + z_pos]);
            delta_Sz[idx] = -tmp * (J[idx + z_pos] * Sz[idx + z_pos]);
        }
    }
}

void delta_mzSz_with_indices_multithreading(
    std::vector<float> &ma, std::vector<float> &Sa, std::vector<float> &Sz,
    std::vector<float> &J, std::vector<float> &y, std::vector<float> &Sv,
    std::vector<int> &udIdx, int z_pos, int ny, int nye, int n,
    unsigned int NUM_THREADS, std::vector<float> &delta_mz,
    std::vector<float> &delta_Sz)

{
    const int n_batch = n / NUM_THREADS;
    const int rem_batch = n % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(delta_mzSz_with_indices_worker, std::ref(ma),
                                 std::ref(Sa), std::ref(Sz), std::ref(J),
                                 std::ref(y), std::ref(Sv), std::ref(udIdx),
                                 z_pos, ny, nye, start_idx, end_idx,
                                 std::ref(delta_mz), std::ref(delta_Sz));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void delta_mzSz_worker(std::vector<float> &ma, std::vector<float> &Sa,
                       std::vector<float> &Sz, std::vector<float> &J,
                       std::vector<float> &y, std::vector<float> &Sv, int z_pos,
                       int start_idx, int end_idx, std::vector<float> &delta_mz,
                       std::vector<float> &delta_Sz) {
    float zeroPad = 0;
    float tmp = 0;
    for (int col = start_idx; col < end_idx; col++) {
        tmp = (J[col + z_pos] * Sz[col + z_pos]) / (Sa[col + z_pos] + Sv[col]);
        if (isinf(tmp) || isnan(tmp)) {
            delta_mz[col] = zeroPad;
            delta_Sz[col] = zeroPad;
        } else {
            delta_mz[col] = tmp * (y[col] - ma[col + z_pos]);
            delta_Sz[col] = -tmp * (J[col + z_pos] * Sz[col + z_pos]);
        }
    }
}

void delta_mzSz_multithreading(std::vector<float> &ma, std::vector<float> &Sa,
                               std::vector<float> &Sz, std::vector<float> &J,
                               std::vector<float> &y, std::vector<float> &Sv,
                               int z_pos, int n, unsigned int NUM_THREADS,
                               std::vector<float> &delta_mz,
                               std::vector<float> &delta_Sz) {
    const int n_batch = n / NUM_THREADS;
    const int rem_batch = n % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(delta_mzSz_worker, std::ref(ma), std::ref(Sa),
                                 std::ref(Sz), std::ref(J), std::ref(y),
                                 std::ref(Sv), z_pos, start_idx, end_idx,
                                 std::ref(delta_mz), std::ref(delta_Sz));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

///////////////////////////////////////////////////////////////////////////
/// NOISE INFERENCE
///////////////////////////////////////////////////////////////////////////
void get_obs_noise_variance_with_idx_cpu(std::vector<float> &Sa,
                                         std::vector<int> &udIdx, int ny,
                                         int nye, std::vector<float> &Sv)
/*Get observation noise variance from the last output layer

Args:
    Sa: Variance predicted using network
    udIdx: Selected indiced to update
    ny: Number of hidden states of the output layer without hidden states
        for noise observation
    nye: Number of observation to be updated for an observation
    Sv: Observation variance i.e., V = [nye x 1]
*/
{
    int idx = 0;
    for (int i = 0; i < udIdx.size(); i++) {
        idx = udIdx[i] + (i / nye) * ny - 1;
        Sv[i] += Sa[idx];
    }
}

void join_output_hidden_states_cpu(std::vector<float> &z_mu,
                                   std::vector<float> &z_v2, int ny,
                                   std::vector<float> &z)
/* Attach noise's hidden states with the mean's hidden states.

Args:
    z_mu: Hidden states of the output
    z_v2: Hidden states of observation noise
    ny: Number of hidden states of the output layer including hidden states
        for noise observation
    z: Hidden states of the output layer (Output + noise's hidden states)
 */
{
    int h = ny / 2;
    int m, k;
    for (int i = 0; i < z_mu.size(); i++) {
        m = (i / h) * ny + i % h;
        k = (i / h) * ny + i % h + h;
        z[m] = z_mu[i];
        z[k] = z_v2[i];
    }
}

void transfer_updated_values_cpu(std::vector<float> &d_z_mu,
                                 std::vector<float> &d_z)
/*Transfer the updated values from noise state to delta state. This is required
   for the case of the homoscedastic nosie in order to update the hidden state
   of the output layer*/
{
    for (int i = 0; i < d_z_mu.size(); i++) {
        d_z[i] = d_z_mu[i];
    }
}

void delta_mz_Sz_backward_cpu(
    std::vector<float> &ma_prior, std::vector<float> &Sa_prior,
    std::vector<float> &J, std::vector<float> &Cza_prior,
    std::vector<float> &ma_post, std::vector<float> &Sa_post,
    std::vector<float> &delta_mz, std::vector<float> &delta_Sz)
/*Compute the updated quantities for hidden states using the backward update
  i.e. smoother algorithm

Args:
    ma_prior: Prior mean of activation unit
    Sa_prior: Prior variance of activation unit
    J: Jacobian matrix
    Cza_prior: Covariance between hidden state and activation units
    ma_post: Posterior mean of activation units
    Sa_post: Posterior variance of activation units
    delta_mz: Updated quantities of mean for the hidden states
    delta_Sz: Updated quantities of variance for the hidden states
 */
{
    float Jz = 0.0f;
    for (int i = 0; i < ma_prior.size(); i++) {
        Jz = J[i] * Cza_prior[i] / Sa_prior[i];
        delta_mz[i] = Jz * (ma_post[i] - ma_prior[i]);
        delta_Sz[i] = Jz * (Sa_post[i] - Sa_prior[i]) * Jz;
    }
}

void delta_mz_Sz_with_indices_backward_cpu(
    std::vector<float> &ma_prior, std::vector<float> &Sa_prior,
    std::vector<float> &J, std::vector<float> &Cza_prior,
    std::vector<float> &ma_post, std::vector<float> &Sa_post,
    std::vector<int> &ud_idx, int ny, int nye, std::vector<float> &delta_mz,
    std::vector<float> &delta_Sz)
/*Compute the updated quantities for specified hidden states using the backward
  update i.e. smoother algorithm

Args:
    ma_prior: Prior mean of activation unit
    Sa_prior: Prior variance of activation unit
    J: Jacobian matrix
    Cza_prior: Covariance between hidden state and activation units
    ma_post: Posterior mean of activation units
    Sa_post: Posterior variance of activation units
    up_idx: Indices for the hidden states to be updated
    ny: Total number of hidden states for the output layer w/o noise's hidden
        states
    nye: Totoal number of hidden states to be updated for the output layer
    delta_mz: Updated values of mean for the hidden states
    delta_Sz: Updated values of variance for the hidden states
 */
{
    float Jz = 0.0f;
    int idx = 0;
    for (int i = 0; i < ud_idx.size(); i++) {
        idx = ud_idx[i] + (i / nye) * ny - 1;
        Jz = J[idx] * Cza_prior[idx] / Sa_prior[idx];
        delta_mz[idx] = Jz * (ma_post[idx] - ma_prior[idx]);
        delta_Sz[idx] = Jz * (Sa_post[idx] - Sa_prior[idx]) * Jz;
    }
}

void compute_posterior_for_v_squared_cpu(std::vector<float> &delta_mv,
                                         std::vector<float> &delta_Sv,
                                         std::vector<float> &ma_v2,
                                         std::vector<float> &mz_v2,
                                         std::vector<float> &Sz_v2)
/* Compute the posterior distribution for the v squared.

Args:
    delta_mv: Updated value of the mean for the observation noise (v)
    delta_Sv: Updated value of the variance of the observation noise
    ma_v2: Mean of activation units for the observation noise squared (v^2)
    Sa_v2: Variance of activation units for the observation noise squared
    mz_v2: Mean of hidden states for the observation noise squared
    Sz_v2: Variance of hidden states for the observation noise squared
 */
{
    int n = delta_mv.size();
    float Sv_p;
    for (int i = 0; i < n; i++) {
        Sv_p = ma_v2[i] + delta_Sv[i];
        mz_v2[i] = pow(delta_mv[i], 2) + Sv_p;
        Sz_v2[i] = 2 * pow(Sv_p, 2) + 4 * pow(delta_mv[i], 2) * Sv_p;
    }
}

void compute_prior_for_v_squared_cpu(std::vector<float> &ma_v2b,
                                     std::vector<float> &Sa_v2b,
                                     std::vector<float> &Sa_v2)
/* Compute the posterior distribition for observation noise v.

Args:
    ma_v2: Mean of activation units for the observation noise squared (v^2)
    Sa_v2: Variance of activation units for the observation noise squared
 */
{
    int n = Sa_v2.size();
    for (int i = 0; i < n; i++) {
        Sa_v2[i] = 3 * Sa_v2b[i] + 2 * pow(ma_v2b[i], 2);
    }
}

void delta_mz_Sz_output_dist_cpu(std::vector<float> &y, std::vector<float> &Sv,
                                 NoiseState &noise_state)
/*Compute the updated quantities for the output distribution. The
   observation is defined following
                        y = x + v, v ~ N(0, \sigma_v^2),
   where y is the observation and x is the output distribution i.e.,
   x ~ N(\mu_x, Sx).

Args:
    y: Observation vector
    Sv: Observation noise
    noise_state: Noise state for the output layer
*/
{
    // Update hidden states for the mean
    delta_mzSz(noise_state.ma_mu, noise_state.Sa_mu, noise_state.Sz_mu,
               noise_state.J_mu, y, noise_state.ma_v2b_prior, 0,
               noise_state.ma_v2b_prior.size(), noise_state.delta_mz_mu,
               noise_state.delta_Sz_mu);

    // Update hidden states for observation noise's hidden states
    delta_mzSz(noise_state.ma_mu, noise_state.Sa_mu, noise_state.ma_v2b_prior,
               noise_state.J_v, y, noise_state.ma_v2b_prior, 0,
               noise_state.ma_v2b_prior.size(), noise_state.delta_mv,
               noise_state.delta_Sv);
}

void delta_mz_Sz_noise_dist_cpu(NoiseState &noise_state, std::string noise_type)
/*Compute the updated quantities for the heteroscedastic & homoscedastic noise
   distribution for the observation noise squared (v^2). The observation is
   defined following
                    y = x + v, v ~ N(0, \sigma_v^2)

Args:
    noise_state: Noise state for the output layer
    noise_type: Type of noise i.e., homoscedastic or heteroscedastic noises
*/
{
    // Update hidden stats for observation noise squared
    int z_pos_v = noise_state.ma_v2b_prior.size();
    compute_posterior_for_v_squared_cpu(
        noise_state.delta_mv, noise_state.delta_Sv, noise_state.ma_v2b_prior,
        noise_state.ma_v2_post, noise_state.Sa_v2_post);

    compute_prior_for_v_squared_cpu(noise_state.ma_v2b_prior,
                                    noise_state.Sa_v2b_prior,
                                    noise_state.Sa_v2_prior);

    // NOTE: We do not apply the activatation function i.e., exponential
    // function for the hidden states representing the observation noise for the
    // homoscedastic case so that we have to handle both following cases
    // differently.
    // Heteroscedastic case
    if (noise_type.compare("heteros") == 0) {
        delta_mz_Sz_backward_cpu(
            noise_state.ma_v2b_prior, noise_state.Sa_v2_prior, noise_state.J_v2,
            noise_state.Cza_v2, noise_state.ma_v2_post, noise_state.Sa_v2_post,
            noise_state.delta_mz_v2b, noise_state.delta_Sz_v2b);
    }
    // Homoscedastic case
    else if (noise_type.compare("homosce") == 0) {
        delta_mz_Sz_backward_cpu(
            noise_state.ma_v2b_prior, noise_state.Sa_v2_prior, noise_state.J_v,
            noise_state.Sa_v2b_prior, noise_state.ma_v2_post,
            noise_state.Sa_v2_post, noise_state.delta_mz_v2b,
            noise_state.delta_Sz_v2b);
    } else {
        throw std::invalid_argument(
            "Noise inference type is invalid - delta_mz_Sz_noise_dist_cpu");
    }
}

void delta_mz_Sz_with_idx_output_dist_cpu(std::vector<float> &y,
                                          std::vector<float> &Sv,
                                          std::vector<int> &ud_idx, int ny,
                                          int nye, NoiseState &noise_state)
/*Compute the updated quantities for the output distribution specified by
 indices

 Args:
    y: Observation vector
    Sv: Observation noise
    up_idx: Indices for the hidden states to be updated
    ny: Total number of hidden states for the output layer w/o noise's hidden
        states
    nye: Totoal number of hidden states to be updated for the output layer
    noise_state: Noise state for the output layer

 */
{
    // Get number of hidden states for the output layer without the hidden
    // states for the observation noise
    int z_pos = 0;

    // Compute the observation noise variance
    get_obs_noise_variance_with_idx_cpu(noise_state.ma_v2b_prior, ud_idx, ny,
                                        nye, Sv);

    // Update hidden states for the mean
    delta_mzSz_with_indices(noise_state.ma_mu, noise_state.Sa_mu,
                            noise_state.Sz_mu, noise_state.J_mu, y, Sv, ud_idx,
                            z_pos, ny, nye, y.size(), noise_state.delta_mz_mu,
                            noise_state.delta_Sz_mu);

    // Update hidden states for observation noise (v)
    delta_mzSz_with_indices(noise_state.ma_mu, noise_state.Sa_mu,
                            noise_state.ma_v2b_prior, noise_state.J_v, y, Sv,
                            ud_idx, z_pos, ny, nye, y.size(),
                            noise_state.delta_mv, noise_state.delta_Sv);
}

void delta_mz_Sz_with_idx_noise_dist(NoiseState &noise_state,
                                     std::string noise_type,
                                     std::vector<int> &ud_idx, int ny, int nye)
/*Compute the updated quantities for the heteroscedastic & homoscedastic noise
   distribution for the specified observation noise.

Args:
    noise_state: Noise state for the output layer
    noise_type: Type of noise i.e., homoscedastic or heteroscedastic noises
    up_idx: Indices for the hidden states to be updated
    ny: Total number of hidden states for the output layer w/o noise's hidden
        states
    nye: Total number of hidden states to be updated for the output layer
 */
{
    // Update hidden state for observation noise squared (v^2)
    compute_posterior_for_v_squared_cpu(
        noise_state.delta_mv, noise_state.delta_Sv, noise_state.ma_v2b_prior,
        noise_state.ma_v2_post, noise_state.Sa_v2_post);

    compute_prior_for_v_squared_cpu(noise_state.ma_v2b_prior,
                                    noise_state.Sa_v2b_prior,
                                    noise_state.Sa_v2_prior);

    // Heteroscedastic case
    if (noise_type.compare("heteros") == 0) {
        delta_mz_Sz_with_indices_backward_cpu(
            noise_state.ma_v2b_prior, noise_state.Sa_v2_prior, noise_state.J_v2,
            noise_state.Cza_v2, noise_state.ma_v2_post, noise_state.Sa_v2_post,
            ud_idx, ny, nye, noise_state.delta_mz_v2b,
            noise_state.delta_Sz_v2b);
    }
    // Homoscedastic case
    else if (noise_type.compare("homosce") == 0) {
        delta_mz_Sz_with_indices_backward_cpu(
            noise_state.ma_v2b_prior, noise_state.Sa_v2_prior, noise_state.J_v,
            noise_state.Sa_v2b_prior, noise_state.ma_v2_post,
            noise_state.Sa_v2_post, ud_idx, ny, nye, noise_state.delta_mz_v2b,
            noise_state.delta_Sz_v2b);
    } else {
        throw std::invalid_argument(
            "Noise inference type is invalid - "
            "delta_mz_Sz_with_idx_noise_dist");
    }
}

///////////////////////////////////////////////////////////////////////////
/// UPDATE VALUES OF HIDDEN STATES FOR OUTPUT LAYER
///////////////////////////////////////////////////////////////////////////
void reset_updated_values(int n, std::vector<float> &z) {
    for (int i = 0; i < n; i++) {
        z[i] = 0.0f;
    }
}

void update_homosce_noise_cpu(NoiseState &noise_state, int ny, int B)
/* Compute the updated values for homoscedastic noise squared by summing up the
   mini-batches of updated values of each noise observation squared.
 */
{
    float tmp_m = 0.0f;
    float tmp_S = 0.0f;
    for (int i = 0; i < ny * B; i++) {
        tmp_m = 0.0f;
        tmp_S = 0.0f;
        for (int j = 0; j < B; j++) {
            tmp_m += noise_state.delta_mz_v2b[(j % B) * ny + i % ny];
            tmp_S += noise_state.delta_Sz_v2b[(j % B) * ny + i % ny];
        }

        noise_state.ma_v2b_prior[i] += tmp_m;
        noise_state.Sa_v2b_prior[i] += tmp_S;
    }
}

void output_delta_mz_Sz_with_noise_inferenece_cpu(NetState &state, Network &net,
                                                  Obs &obs, DeltaState &d_state)
/* Compute the updated value for the output layer including the noise
   observation's hidden states.
 */
{
    int z_pos = net.z_pos.back();
    if (net.is_idx_ud) {
        // Reset the updated values to zeros
        reset_updated_values(state.noise_state.delta_mz_mu.size(),
                             state.noise_state.delta_mz_mu);
        reset_updated_values(state.noise_state.delta_Sz_mu.size(),
                             state.noise_state.delta_Sz_mu);
        reset_updated_values(state.noise_state.delta_mv.size(),
                             state.noise_state.delta_mv);
        reset_updated_values(state.noise_state.delta_Sv.size(),
                             state.noise_state.delta_Sv);
        reset_updated_values(state.noise_state.delta_mz_v2b.size(),
                             state.noise_state.delta_mz_v2b);
        reset_updated_values(state.noise_state.delta_Sz_v2b.size(),
                             state.noise_state.delta_Sz_v2b);

        int ny_B = net.n_y * net.batch_size;
        d_state.reset_updated_values(ny_B);

        // Compute updated values for the output distribution
        delta_mz_Sz_with_idx_output_dist_cpu(obs.y_batch, obs.V_batch,
                                             obs.idx_ud_batch, net.n_y, net.nye,
                                             state.noise_state);

        // Compute updated values for the noise observation of the output
        // distribution
        delta_mz_Sz_with_idx_noise_dist(state.noise_state, net.noise_type,
                                        obs.idx_ud_batch, net.n_y, net.nye);
    } else {
        // Compute updated values for the output distribution
        delta_mz_Sz_output_dist_cpu(obs.y_batch, obs.V_batch,
                                    state.noise_state);

        // Compute updated values for the noise observation of the output
        // distribution
        delta_mz_Sz_noise_dist_cpu(state.noise_state, net.noise_type);
    }

    // Join updated values (outputs + its observation noise)
    if (net.noise_type.compare("heteros") == 0) {
        join_output_hidden_states_cpu(state.noise_state.delta_mz_mu,
                                      state.noise_state.delta_mz_v2b,
                                      net.nodes.back(), d_state.delta_mz);

        join_output_hidden_states_cpu(state.noise_state.delta_Sz_mu,
                                      state.noise_state.delta_Sz_v2b,
                                      net.nodes.back(), d_state.delta_Sz);
    } else if (net.noise_type.compare("homosce") == 0) {
        transfer_updated_values_cpu(state.noise_state.delta_mz_mu,
                                    d_state.delta_mz);
        transfer_updated_values_cpu(state.noise_state.delta_Sz_mu,
                                    d_state.delta_Sz);
        update_homosce_noise_cpu(state.noise_state, net.nodes.back(),
                                 net.batch_size);
    } else {
        throw std::invalid_argument(
            "Noise inference type is invalid - "
            "output_delta_mz_Sz_with_noise_inferenece_cpu");
    }
}

void output_delta_mz_Sz_cpu(Network &net, NetState &state, Obs &obs,
                            DeltaState &d_state)
/* Compute the updated value for the hidden states of the output layer

 Args:
    net: Network architecture
    state: Hidden state of network
    obs: Observations
    d_state: Updated quantities for network's hidden states
 */
{
    int n_state_last_layer = net.batch_size * net.nodes.back();
    if (!net.is_idx_ud) {
        if (n_state_last_layer > net.min_operations && net.multithreading) {
            delta_mzSz_multithreading(
                state.ma, state.Sa, state.Sz, state.J, obs.y_batch, obs.V_batch,
                net.z_pos.back(), n_state_last_layer, net.num_cpu_threads,
                d_state.delta_mz, d_state.delta_Sz);
        } else {
            delta_mzSz(state.ma, state.Sa, state.Sz, state.J, obs.y_batch,
                       obs.V_batch, net.z_pos.back(), n_state_last_layer,
                       d_state.delta_mz, d_state.delta_Sz);
        }
    } else {
        // Reset updated values to zeros
        int ny_B = net.n_y * net.batch_size;
        d_state.reset_updated_values(ny_B);

        int n_state_last_layer_e = net.nye * net.batch_size;
        if (n_state_last_layer > net.min_operations && net.multithreading) {
            delta_mzSz_with_indices_multithreading(
                state.ma, state.Sa, state.Sz, state.J, obs.y_batch, obs.V_batch,
                obs.idx_ud_batch, net.z_pos.back(), net.nodes.back(), net.nye,
                n_state_last_layer_e, net.num_cpu_threads, d_state.delta_mz,
                d_state.delta_Sz);
        } else {
            delta_mzSz_with_indices(
                state.ma, state.Sa, state.Sz, state.J, obs.y_batch, obs.V_batch,
                obs.idx_ud_batch, net.z_pos.back(), net.nodes.back(), net.nye,
                n_state_last_layer_e, d_state.delta_mz, d_state.delta_Sz);
        }
    }
}

void remax_output_delta_z_cpu(Network &net, NetState &state, Obs &obs,
                              DeltaState &d_state)
/*Compute updating quantities of hidden states for softmax layer*/
{
    int no = net.nodes.back();
    int B = net.batch_size;
    int z_pos = net.z_pos.back();
    // Covariance between m and \check{a}
    compute_cov_m_a_check_cpu(state.remax.var_log, state.remax.cov_log_logsum,
                              state.remax.mu_m, 0, 0, no, B,
                              state.remax.cov_m_a_check);

    // Covariance between m and a
    compute_cov_m_a_cpu(state.remax.cov_m_a_check, state.ma, state.remax.var_m,
                        state.Sz, state.remax.J_m, 0, z_pos, no, B,
                        state.remax.cov_m_a);

    // Updating quantities for hidden states
    delta_z_y_check_cpu(state.ma, state.Sa, state.remax.cov_m_a, obs.y_batch,
                        obs.V_batch, no, B, z_pos, d_state.delta_mz,
                        d_state.delta_Sz);
}

void update_output_hidden_states_cpu(Network &net, NetState &state, Obs &obs,
                                     DeltaState &d_state)
/*Compute updated quantities for the output layer's hidden state

 Args:
    net: Network architecture
    state: Hidden state of network
    d_state: Updated quantities for network's hidden states
    obs: Observations
 */
{
    if (net.is_output_ud) {
        if (net.noise_type.compare("homosce") != 0 &&
            net.noise_type.compare("heteros") != 0 &&
            (net.activations.back() != net.act_names.remax)) {
            output_delta_mz_Sz_cpu(net, state, obs, d_state);
        } else if (net.activations.back() == net.act_names.remax) {
            remax_output_delta_z_cpu(net, state, obs, d_state);
        } else {
            output_delta_mz_Sz_with_noise_inferenece_cpu(state, net, obs,
                                                         d_state);
        }
    } else {
        d_state.delta_mz = obs.y_batch;
        d_state.delta_Sz = obs.V_batch;
    }
}

///////////////////////////////////////////////////////////////////////////
/// STATE BACKWARD
///////////////////////////////////////////////////////////////////////////
void state_backward_cpu(Network &net, Param &theta, NetState &state,
                        IndexOut &idx, Obs &obs, DeltaState &d_state)
/*Compute the updated quantities for network's hidden states using TAGI.

  Args:
    net: Network architecture
    theta: Network's weights and biases
    state: Hidden state of network
    idx: Indices for network e.g. see indices.cpp
    obs: Observations

  Returns:
    d_state: Updated quantities for network's hidden states
*/
{
    // Compute updated quantities for the output layer's hidden state
    int n_state_last_layer = net.batch_size * net.nodes.back();
    update_output_hidden_states_cpu(net, state, obs, d_state);

    // Compute inovation vector
    if (n_state_last_layer > net.min_operations && net.multithreading) {
        inovation_multithreading(state.Sz, d_state.delta_mz, d_state.delta_Sz,
                                 net.z_pos.back(), net.z_pos.back(),
                                 n_state_last_layer, net.num_cpu_threads,
                                 d_state.delta_m, d_state.delta_S);
    } else {
        inovation_mean(state.Sz, d_state.delta_mz, net.z_pos.back(),
                       net.z_pos.back(), n_state_last_layer, d_state.delta_m);
        inovation_var(state.Sz, d_state.delta_Sz, net.z_pos.back(),
                      net.z_pos.back(), n_state_last_layer, d_state.delta_S);
    }

    int no, ni, niB, z_pos_in, z_pos_out, w_pos_in;
    int B = net.batch_size;
    for (int k = net.layers.size() - 2; k >= net.last_backward_layer; k--) {
        no = net.nodes[k + 1];
        ni = net.nodes[k];
        // Handle multiple input sequences from LSTM layer
        if (net.layers[k] == net.layer_names.lstm) {
            ni = net.nodes[k] * net.input_seq_len;
        }
        z_pos_out = net.z_pos[k + 1];
        z_pos_in = net.z_pos[k];
        w_pos_in = net.w_pos[k];
        niB = ni * B;

        //**
        // 1: Fully connected
        //
        if (net.layers[k + 1] == net.layer_names.fc) {
            if (niB > net.min_operations && net.multithreading) {
                fc_delta_mzSz_multithreading(
                    theta.mw, state.Sz, state.J, d_state.delta_m,
                    d_state.delta_S, w_pos_in, z_pos_in, z_pos_out, ni, no, B,
                    net.num_cpu_threads, d_state.delta_mz, d_state.delta_Sz);

            } else {
                fc_delta_mz(theta.mw, state.Sz, state.J, d_state.delta_m,
                            w_pos_in, z_pos_in, z_pos_out, ni, no, B,
                            d_state.delta_mz);
                fc_delta_Sz(theta.mw, state.Sz, state.J, d_state.delta_S,
                            w_pos_in, z_pos_in, z_pos_out, ni, no, B,
                            d_state.delta_Sz);
            }
        }
        //**
        // 7: LSTM
        //
        else if (net.layers[k + 1] == net.layer_names.lstm) {
            lstm_state_update_cpu(net, state, theta, d_state, k);
        }
        //**
        // 8: MHA
        //
        else if (net.layers[k + 1] == net.layer_names.mha) {
            update_self_attention_state(net, state, theta, d_state, k);
        }

        if (niB > net.min_operations && net.multithreading) {
            inovation_multithreading(state.Sz, d_state.delta_mz,
                                     d_state.delta_Sz, z_pos_in, z_pos_in, niB,
                                     net.num_cpu_threads, d_state.delta_m,
                                     d_state.delta_S);
        } else {
            inovation_mean(state.Sz, d_state.delta_mz, z_pos_in, z_pos_in, niB,
                           d_state.delta_m);
            inovation_var(state.Sz, d_state.delta_Sz, z_pos_in, z_pos_in, niB,
                          d_state.delta_S);
        }
    }
}