# import libraries
import os
import sys

import fire
import numpy as np
from tqdm import tqdm

import pytagi.metric as metric
from pytagi import exponential_scheduler
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential

from examples.data_loader import GlobalTimeSeriesDataloader

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)


class TimeSeriesEmbeddings:
    """
    Class to handle embedding operations with mean and variance.
    """

    def __init__(self, embedding_dim: tuple, encoding_type: str = "normal"):
        self.embedding_dim = embedding_dim
        if encoding_type == "normal":
            self.mu_embedding = np.random.randn(*embedding_dim)
            self.var_embedding = np.full(embedding_dim, 1.0)
        elif encoding_type == "onehot":
            epsilon = 1e-6
            self.mu_embedding = np.full(embedding_dim, epsilon)
            np.fill_diagonal(self.mu_embedding, 1.0)
            self.var_embedding = np.ones(embedding_dim)
        else:
            raise ValueError("Encoding type not supported.")

    def update(self, idx: int, mu_delta: np.ndarray, var_delta: np.ndarray):
        self.mu_embedding[idx] = self.mu_embedding[idx] + mu_delta
        self.var_embedding[idx] = self.var_embedding[idx] + var_delta

    def get_embedding(self, idx: int) -> tuple:
        return self.mu_embedding[idx], self.var_embedding[idx]


# TODO: Create a EmbeddingHandler class to handle the embeddings (build, reduce, input)
def build_vector(x: int, num_features_len: int, embedding_dim: int) -> np.ndarray:
    vector = np.zeros(x)
    cycle_length = num_features_len + embedding_dim

    # Iterate through the vector in steps of the cycle length
    for i in range(0, x, cycle_length):
        # Find the starting position of the embedding section in the current cycle
        embedding_start = i + num_features_len

        # Ensure the embedding section doesn't go out of bounds
        if embedding_start < x:
            # Set the values of the embedding section to ones
            end_position = min(embedding_start + embedding_dim, x)
            vector[embedding_start:end_position] = np.ones(end_position - embedding_start)

    return vector


def reduce_vector(x: np.ndarray, vector: np.ndarray, embedding_dim: int) -> np.ndarray:
    x = (x + 1) * vector
    x = x[x != 0] - 1  # remove zeros and reset index
    return x.reshape(-1, embedding_dim)


def input_embeddings(x, embeddings, num_features, embedding_dim):
    """
    Reads embeddings into the input vector.
    """
    x_var = x.copy()
    counter = 0
    last_idx = 0

    for item in x:
        if counter % num_features == 0 and counter != 0 and counter + last_idx < len(x):
            idx = int(x[counter + last_idx])
            embed_x, embed_var = embeddings.get_embedding(idx)
            (x[counter + last_idx:counter + last_idx + embedding_dim],
             x_var[counter + last_idx:counter + last_idx + embedding_dim]) = (embed_x.tolist(),
                                                                              embed_var.tolist())
            last_idx = counter + embedding_dim + last_idx
            counter = 0
        else:
            counter += 1

    return np.array(x, dtype=np.float32), np.array(x_var, dtype=np.float32)


def main(num_epochs: int = 1, batch_size: int = 64, sigma_v: float = 0.5, lstm_nodes: int = 40):
    """
    Run training for a time-series forecasting global model.
    Training is done on one complete time series at a time.
    """
    # Dataset
    embedding_dim = 2  # dimension of the embedding
    nb_ts = 10  # for electricity 370 and 963 for traffic
    ts_idx = np.arange(0, nb_ts)
    ts_idx_test = np.arange(0, nb_ts)  # unshuffled ts_idx for testing
    output_col = [0]
    num_features = 3
    input_seq_len = 5
    output_seq_len = 1
    seq_stride = 1
    rolling_window = 5  # for rolling window predictions in the test set
    embeddings = TimeSeriesEmbeddings((nb_ts, embedding_dim))  # initialize embeddings

    # Network
    net = Sequential(
        LSTM((num_features + embedding_dim), lstm_nodes, input_seq_len),
        LSTM(lstm_nodes, lstm_nodes, input_seq_len),
        LSTM(lstm_nodes, lstm_nodes, input_seq_len),
        Linear(lstm_nodes * input_seq_len, 1),
    )
    # net.to_device("cuda")
    # net.set_threads(8)
    out_updater = OutputUpdater(net.device)

    # input state update
    net.input_state_update = True

    # Create an output directory
    out_dir = ("dw_out/traffic_" + str(num_epochs) + "_" + str(batch_size) + "_" + str(sigma_v)
               + "_" + str(lstm_nodes) + "_method1")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # -------------------------------------------------------------------------#
    # Training
    mses = []
    mses_val = []  # to save mse_val for plotting
    ll_val = []  # to save log likelihood for plotting

    # options for early stopping
    log_lik_optim = -1E100
    mse_optim = 1E100
    epoch_optim = 1
    early_stopping_criteria = 'mse'  # 'log_lik' or 'mse'
    patience = 10
    global_mse = []
    global_log_lik = []

    pbar = tqdm(range(num_epochs), desc="Training Progress")

    covar_means = [0] * nb_ts
    covar_stds = [1.0] * nb_ts

    w_dir = '/Users/davidwardan/Library/CloudStorage/OneDrive-Personal/Projects/cuTAGI_DW'

    for epoch in pbar:

        # Decaying observation's variance
        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=0.01, decaying_factor=0.99, curr_iter=epoch
        )
        var_y = np.full((batch_size * len(output_col),), sigma_v ** 2, dtype=np.float32)

        for ts in ts_idx:
            train_dtl = GlobalTimeSeriesDataloader(
                x_file=w_dir + "/data/traffic/traffic_2008_01_14_train.csv",
                date_time_file=w_dir + "/data/traffic/traffic_2008_01_14_train_datetime.csv",
                output_col=output_col,
                input_seq_len=input_seq_len,
                output_seq_len=output_seq_len,
                num_features=num_features,
                stride=seq_stride,
                ts_idx=ts,
                time_covariates=['hour_of_day', 'day_of_week'],
                scale_covariates=True,
                embedding_dim=embedding_dim,
            )

            # store covariate means and stds
            covar_means[ts] = train_dtl.covariate_means
            covar_stds[ts] = train_dtl.covariate_stds

            batch_iter = train_dtl.create_data_loader(batch_size, shuffle=False)

            mse = []
            for x, y in batch_iter:
                # build a zero vector with the same length as x with 1 where the embedding is stored
                zero_vector = build_vector(x.shape[0], num_features, embedding_dim)
                # get the indices of the time series stored in the embedding
                x_ts_idx = reduce_vector(x, zero_vector, embedding_dim)
                # reduce the vector to get only the indices (needed if embedding_dim > 1)
                x_ts_idx = np.mean(x_ts_idx, axis=1).tolist()

                # replace the embedding section with the actual embedding
                x, x_var = input_embeddings(x, embeddings, num_features, embedding_dim)

                # only leave variance for the embedding section
                x_var = x_var * zero_vector

                # Feed forward
                m_pred, _ = net(x, x_var)

                # Update output layer
                out_updater.update(
                    output_states=net.output_z_buffer,
                    mu_obs=y,
                    var_obs=var_y,
                    delta_states=net.input_delta_z_buffer,
                )

                # Feed backward
                net.backward()
                net.step()

                # get the updated input state
                (mu_delta, var_delta) = net.get_input_states()

                # update the embedding
                x_update = mu_delta * x_var
                var_update = x_var * var_delta * x_var

                # reduce the vector to get only embeddings
                x_update = reduce_vector(x_update, zero_vector, embedding_dim)
                var_update = reduce_vector(var_update, zero_vector, embedding_dim)

                # store the updated embedding
                vec_loc = 0
                for ts_idx_ in x_ts_idx:
                    ts_idx_ = int(ts_idx_)
                    embeddings.update(ts_idx_, x_update[vec_loc], var_update[vec_loc])
                    vec_loc += 1

                # Compute MSE
                mse.append(metric.mse(m_pred, y))
            mses.append(np.mean(mse))

        #-------------------------------------------------------------------------#
        # validation
        # define validation progress inside the training progress
        for ts in ts_idx:
            embed_mu, embed_var = embeddings.get_embedding(ts)

            val_dtl = GlobalTimeSeriesDataloader(
                x_file=w_dir + "/data/traffic/traffic_2008_01_14_val.csv",
                date_time_file=w_dir + "/data/traffic/traffic_2008_01_14_val_datetime.csv",
                output_col=output_col,
                input_seq_len=input_seq_len,
                output_seq_len=output_seq_len,
                num_features=num_features,
                stride=seq_stride,
                ts_idx=ts,
                time_covariates=['hour_of_day', 'day_of_week'],
                scale_covariates=True,
                covariate_means=covar_means[ts],
                covariate_stds=covar_stds[ts],
                embedding=embed_mu,
            )

            val_batch_iter = val_dtl.create_data_loader(batch_size, shuffle=False)

            mu_preds = []
            var_preds = []
            y_val = []
            x_val = []

            for x, y in val_batch_iter:
                # Prediction
                m_pred, v_pred = net(x)

                mu_preds.extend(m_pred)
                var_preds.extend(v_pred + sigma_v ** 2)
                x_val.extend(x)
                y_val.extend(y)

            mu_preds = np.array(mu_preds)
            std_preds = np.array(var_preds) ** 0.5
            y_val = np.array(y_val)
            x_val = np.array(x_val)

            # Compute log-likelihood for validation set
            mse_val = metric.mse(mu_preds, y_val)
            log_lik_val = metric.log_likelihood(
                prediction=mu_preds, observation=y_val, std=std_preds
            )

            global_mse.append(mse_val)
            global_log_lik.append(log_lik_val)

        mse_val = np.mean(global_mse)
        log_lik_val = np.mean(global_log_lik)

        mses_val.append(mse_val)
        ll_val.append(log_lik_val)

        pbar.set_postfix(mse=f"{np.mean(mses):.4f}", mse_val=f"{mse_val:.4f}", log_lik_val=f"{log_lik_val:.4f}",
                         sigma_v=f"{sigma_v:.4f}")

        # early-stopping
        if early_stopping_criteria == 'mse':
            if mse_val < mse_optim:
                mse_optim = mse_val
                log_lik_optim = log_lik_val
                epoch_optim = epoch
                net_optim = net.get_state_dict()
        elif early_stopping_criteria == 'log_lik':
            if log_lik_val > log_lik_optim:
                mse_optim = mse_val
                log_lik_optim = log_lik_val
                epoch_optim = epoch
                net_optim = net.get_state_dict()
        if epoch - epoch_optim > patience:
            break

        # shuffle ts_idx
        np.random.shuffle(ts_idx)

    # save validation metrics into csv
    df = np.array([mses_val, ll_val]).T
    np.savetxt(out_dir + "/validation_metrics.csv", df, delimiter=",")

    # save the model
    net.save_csv(out_dir + "/param/traffic_2008_01_14_net_pyTAGI.csv")

    # Testing
    pbar = tqdm(ts_idx_test, desc="Testing Progress")

    ytestPd = np.full((168, nb_ts), np.nan)
    SytestPd = np.full((168, nb_ts), np.nan)
    ytestTr = np.full((168, nb_ts), np.nan)
    for ts in pbar:
        embed_mu, embed_var = embeddings.get_embedding(ts)

        test_dtl = GlobalTimeSeriesDataloader(
            x_file=w_dir + "/data/traffic/traffic_2008_01_14_test.csv",
            date_time_file=w_dir + "/data/traffic/traffic_2008_01_14_test_datetime.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            time_covariates=['hour_of_day', 'day_of_week'],
            scale_covariates=True,
            covariate_means=covar_means[ts],
            covariate_stds=covar_stds[ts],
            embedding=embed_mu,
        )

        # test_batch_iter = test_dtl.create_data_loader(batch_size, shuffle=False)
        test_batch_iter = test_dtl.create_data_loader(1, shuffle=False)

        mu_preds = []
        var_preds = []
        y_test = []
        x_test = []

        net.load_state_dict(net_optim)  # load optimal net

        # Rolling window predictions
        for RW_idx_, (x, y) in enumerate(test_batch_iter):
            # Rolling window predictions
            RW_idx = RW_idx_ % rolling_window
            if RW_idx > 0:
                x[-RW_idx * (num_features + embedding_dim)::(num_features + embedding_dim)] = mu_preds[-RW_idx:]

            # Prediction
            m_pred, v_pred = net(x)

            mu_preds.extend(m_pred)
            var_preds.extend(v_pred + sigma_v ** 2)
            x_test.extend(x)
            y_test.extend(y)

        mu_preds = np.array(mu_preds)
        std_preds = np.array(var_preds) ** 0.5
        y_test = np.array(y_test)
        x_test = np.array(x_test)

        # save test predictions for each time series
        ytestPd[:, ts] = mu_preds.flatten()
        SytestPd[:, ts] = std_preds.flatten() ** 2
        ytestTr[:, ts] = y_test.flatten()

    np.savetxt(out_dir + "/traffic_2008_01_14_ytestPd_pyTAGI.csv", ytestPd, delimiter=",")
    np.savetxt(out_dir + "/traffic_2008_01_14_SytestPd_pyTAGI.csv", SytestPd, delimiter=",")
    np.savetxt(out_dir + "/traffic_2008_01_14_ytestTr_pyTAGI.csv", ytestTr, delimiter=",")

    # -------------------------------------------------------------------------#
    # calculate metrics
    p50_tagi = metric.computeND(ytestTr, ytestPd)
    p90_tagi = metric.compute90QL(ytestTr, ytestPd, SytestPd)
    RMSE_tagi = metric.computeRMSE(ytestTr, ytestPd)

    # save metrics into a text file
    with open(out_dir + "/metrics.txt", "w") as f:
        f.write(f'ND/p50:    {p50_tagi}\n')
        f.write(f'p90:    {p90_tagi}\n')
        f.write(f'RMSE:    {RMSE_tagi}\n')
        f.write(f'Epoch:    {epoch_optim}\n')
        f.write(f'Batch size:    {batch_size}\n')
        f.write(f'Sigma_v:    {sigma_v}\n')
        f.write(f'LSTM nodes:    {lstm_nodes}\n')

    # rename the directory
    out_dir_ = "dw_out/traffic_" + str(epoch_optim) + "_" + str(batch_size) + "_" + str(
        round(sigma_v, 3)) + "_" + str(lstm_nodes) + "_method1"
    os.rename(out_dir, out_dir_)


if __name__ == "__main__":
    fire.Fire(main)
