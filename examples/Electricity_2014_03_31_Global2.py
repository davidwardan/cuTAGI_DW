import os
import sys
import fire
import numpy as np
from tqdm import tqdm

import pytagi.metric as metric
from pytagi import exponential_scheduler
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential, EvenExp
from pytagi import Normalizer as normalizer

from examples.data_loader import GlobalTimeSeriesDataloader
from examples.embedding import *

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)


def main(
    num_epochs: int = 100,
    batch_size: int = 32,
    sigma_v: float = 0.2,
    lstm_nodes: int = 40,
    embedding_dim: int = 25,
):
    """
    Run training for a time-series forecasting global model.
    Training is done on shuffling batches from all series.
    """

    # Dataset
    nb_ts = 370  # for electricity 370 and 963 for traffic
    ts_idx = np.arange(0, nb_ts)
    output_col = [0]
    num_features = 3
    input_seq_len = 24
    output_seq_len = 1
    seq_stride = 1
    rolling_window = 24  # for rolling window predictions in the test set
    embeddings = TimeSeriesEmbeddings(
        (nb_ts, embedding_dim), "normal"
    )  # initialize embeddings

    pbar = tqdm(ts_idx, desc="Loading Data Progress")

    factors = [1.0] * nb_ts
    mean_train = [0.0] * nb_ts
    std_train = [1.0] * nb_ts
    covar_means = [0.0] * nb_ts
    covar_stds = [1.0] * nb_ts

    for ts in pbar:
        train_dtl_ = GlobalTimeSeriesDataloader(
            x_file="data/electricity/electricity_2014_03_31_train.csv",
            date_time_file="data/electricity/electricity_2014_03_31_train_datetime.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            time_covariates=["hour_of_day", "day_of_week"],
            global_scale="deepAR",
            scale_covariates=True,
            embedding_dim=embedding_dim,
        )

        # Store scaling factors----------------------#
        factors[ts] = train_dtl_.scale_i
        # mean_train[ts] = train_dtl_.x_mean
        # std_train[ts] = train_dtl_.x_std
        # -------------------------------------------#

        # store covariate means and stds
        covar_means[ts] = train_dtl_.covariate_means
        covar_stds[ts] = train_dtl_.covariate_stds

        val_dtl_ = GlobalTimeSeriesDataloader(
            x_file="data/electricity/electricity_2014_03_31_val.csv",
            date_time_file="data/electricity/electricity_2014_03_31_val_datetime.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            # x_mean=mean_train[ts],
            # x_std=std_train[ts],
            time_covariates=["hour_of_day", "day_of_week"],
            global_scale="deepAR",
            scale_i=factors[ts],
            scale_covariates=True,
            covariate_means=covar_means[ts],
            covariate_stds=covar_stds[ts],
            embedding_dim=embedding_dim,
        )

        if ts == 0:
            train_dtl = train_dtl_
            val_dtl = val_dtl_
        else:
            train_dtl = concat_ts_sample(train_dtl, train_dtl_)
            val_dtl = concat_ts_sample(val_dtl, val_dtl_)

    # Network
    if sigma_v is None:
        net = Sequential(
            LSTM((num_features + embedding_dim), lstm_nodes, input_seq_len),
            LSTM(lstm_nodes, lstm_nodes, input_seq_len),
            LSTM(lstm_nodes, lstm_nodes, input_seq_len),
            Linear(lstm_nodes * input_seq_len, 2),
            EvenExp(),
        )
    else:
        net = Sequential(
            LSTM((num_features + embedding_dim), lstm_nodes, input_seq_len),
            LSTM(lstm_nodes, lstm_nodes, input_seq_len),
            LSTM(lstm_nodes, lstm_nodes, input_seq_len),
            Linear(lstm_nodes * input_seq_len, 1),
        )
    net.to_device("cuda")
    # net.set_threads(8)
    out_updater = OutputUpdater(net.device)

    # input state update
    net.input_state_update = True

    # Create output directory
    out_dir = (
        "out/electricity_"
        + str(num_epochs)
        + "_"
        + str(batch_size)
        + "_"
        + str(sigma_v)
        + "_"
        + str(lstm_nodes)
        + "_method2"
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # -------------------------------------------------------------------------#
    # calculate the weights for each time series
    # weights = train_dtl.dataset["weights"] / np.sum(train_dtl.dataset["weights"])

    # Training
    mses = []
    mses_val = []  # to save mse_val for plotting
    ll_val = []  # to save log likelihood for plotting

    # options for early stopping
    log_lik_optim = -1e100
    mse_optim = 1e100
    epoch_optim = 1
    early_stopping_criteria = "mse"  # 'log_lik' or 'mse'
    patience = 10
    net_optim = []  # to save optimal net at the optimal epoch

    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:

        batch_iter = train_dtl.create_data_loader(
            batch_size,
            shuffle=True,
            # weighted_sampling=True,
            # weights=weights,
            # num_samples=500000,
        )

        if sigma_v != None:
            # Decaying observation's variance
            sigma_v = exponential_scheduler(
                curr_v=sigma_v, min_v=0.01, decaying_factor=0.99, curr_iter=epoch
            )
            var_y = np.full(
                (batch_size * len(output_col),), sigma_v**2, dtype=np.float32
            )

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

            if sigma_v is None:
                m_pred = m_pred[::2]  # Even positions
                # Update output layer
                out_updater.update_heteros(
                    output_states=net.output_z_buffer,
                    mu_obs=y,
                    delta_states=net.input_delta_z_buffer,
                )
            else:
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

            # unscale the predictions
            # pred = m_pred * factors[ts]
            # obs = y * factors[ts]
            # pred = normalizer.unstandardize(
            #     m_pred, mean_train[ts], std_train[ts]
            # )
            # obs = normalizer.unstandardize(
            #     y, mean_train[ts], std_train[ts]
            # )

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

            pred = m_pred
            obs = y

            # Training metric
            mse = metric.mse(pred, obs)
            mses.append(mse)

        # -------------------------------------------------------------------------#
        # # Validation
        val_batch_iter = val_dtl.create_data_loader(batch_size, shuffle=True)

        mu_preds = []
        var_preds = []
        y_val = []
        x_val = []

        for x, y in val_batch_iter:
            # replace the embedding section with the actual embedding
            x, x_var = input_embeddings(x, embeddings, num_features, embedding_dim)
            zero_vector = build_vector(x.shape[0], num_features, embedding_dim)

            # only leave variance for the embedding section
            x_var = x_var * zero_vector

            # Prediction
            m_pred, v_pred = net(x, x_var)

            if sigma_v is None:
                mu_preds.extend(m_pred[::2])
                var_preds.extend(v_pred[::2] + m_pred[1::2])
            else:
                mu_preds.extend(m_pred)
                var_preds.extend(v_pred + sigma_v**2)

            x_val.extend(x)
            y_val.extend(y)

        mu_preds = np.array(mu_preds)
        std_preds = np.array(var_preds) ** 0.5
        y_val = np.array(y_val)
        x_val = np.array(x_val)

        # Unscale the predictions
        # mu_preds = mu_preds * factors[ts]
        # std_preds = std_preds * factors[ts] ** 0.5
        # y_val = y_val * factors[ts]

        # mu_preds = normalizer.unstandardize(
        #     mu_preds, mean_train[ts], std_train[ts]
        # )
        # std_preds = normalizer.unstandardize_std(std_preds, std_train[ts])
        #
        # y_val = normalizer.unstandardize(
        #     y_val, mean_train[ts], std_train[ts]
        # )

        # Compute log-likelihood for validation set
        mse_val = metric.mse(mu_preds, y_val)
        log_lik_val = metric.log_likelihood(
            prediction=mu_preds, observation=y_val, std=std_preds
        )

        # Save mse_val and log likelihood for plotting
        mses_val.append(mse_val)
        ll_val.append(log_lik_val)

        # Progress bar
        pbar.set_postfix(
            mse=f"{np.mean(mses):.4f}",
            mse_val=f"{mse_val:.4f}",
            log_lik_val=f"{log_lik_val:.4f}",
        )
        # pbar.set_postfix(mse=f"{np.mean(mses):.4f}", sigma_v=f"{sigma_v:.4f}")

        # early-stopping
        if early_stopping_criteria == "mse":
            if mse_val < mse_optim:
                mse_optim = mse_val
                log_lik_optim = log_lik_val
                epoch_optim = epoch
                net_optim = net.get_state_dict()
        elif early_stopping_criteria == "log_lik":
            if log_lik_val > log_lik_optim:
                mse_optim = mse_val
                log_lik_optim = log_lik_val
                epoch_optim = epoch
                net_optim = net.get_state_dict()
        if epoch - epoch_optim > patience:
            break

    # save validation metrics into csv
    df = np.array([mses_val, ll_val]).T
    np.savetxt(out_dir + "/validation_metrics.csv", df, delimiter=",")

    # load optimal model
    net.load_state_dict(net_optim)

    # save the model
    net.save_csv(out_dir + "/param/electricity_2014_03_31_net_pyTAGI.csv")

    # save the embeddings
    embeddings.save(out_dir)

    # Testing
    pbar = tqdm(ts_idx, desc="Testing Progress")

    ytestPd = np.full((168, nb_ts), np.nan)
    SytestPd = np.full((168, nb_ts), np.nan)
    ytestTr = np.full((168, nb_ts), np.nan)

    for ts in pbar:

        test_dtl = GlobalTimeSeriesDataloader(
            x_file="data/electricity/electricity_2014_03_31_test.csv",
            date_time_file="data/electricity/electricity_2014_03_31_test_datetime.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            # x_mean=mean_train[ts],
            # x_std=std_train[ts],
            time_covariates=["hour_of_day", "day_of_week"],
            global_scale="deepAR",
            scale_i=factors[ts],
            scale_covariates=True,
            covariate_means=covar_means[ts],
            covariate_stds=covar_stds[ts],
            embedding_dim=embedding_dim,
        )

        # test_batch_iter = test_dtl.create_data_loader(batch_size, shuffle=False)
        test_batch_iter = test_dtl.create_data_loader(1, shuffle=False)

        mu_preds = []
        var_preds = []
        y_test = []
        x_test = []

        # net = net_optim

        # Rolling window predictions
        for RW_idx_, (x, y) in enumerate(test_batch_iter):
            # Rolling window predictions
            RW_idx = RW_idx_ % rolling_window
            if RW_idx > 0:
                x[
                    -RW_idx
                    * (num_features + embedding_dim) :: (num_features + embedding_dim)
                ] = mu_preds[-RW_idx:]

            # replace the embedding section with the actual embedding
            x, x_var = input_embeddings(x, embeddings, num_features, embedding_dim)
            zero_vector = build_vector(x.shape[0], num_features, embedding_dim)

            # only leave variance for the embedding section
            x_var = x_var * zero_vector

            # Prediction
            m_pred, v_pred = net(x, x_var)

            if sigma_v is None:
                mu_preds.extend(m_pred[::2])
                var_preds.extend(v_pred[::2] + m_pred[1::2])
            else:
                mu_preds.extend(m_pred)
                var_preds.extend(v_pred + sigma_v**2)

            x_test.extend(x)
            y_test.extend(y)

        mu_preds = np.array(mu_preds)
        std_preds = np.array(var_preds) ** 0.5
        y_test = np.array(y_test)
        x_test = np.array(x_test)

        # Unscale the predictions
        mu_preds = mu_preds * factors[ts]
        std_preds = std_preds * factors[ts]
        y_test = y_test * factors[ts]
        # mu_preds = normalizer.unstandardize(
        #     mu_preds, mean_train[ts], std_train[ts]
        # )
        # std_preds = normalizer.unstandardize_std(std_preds, std_train[ts])

        # y_test = normalizer.unstandardize(
        #     y_test, mean_train[ts], std_train[ts]
        # )

        # save test predictions for each time series
        ytestPd[:, ts] = mu_preds.flatten()
        SytestPd[:, ts] = std_preds.flatten() ** 2
        ytestTr[:, ts] = y_test.flatten()

    np.savetxt(out_dir + "/ytestPd.csv", ytestPd, delimiter=",")
    np.savetxt(out_dir + "/SytestPd.csv", SytestPd, delimiter=",")
    np.savetxt(out_dir + "/ytestTr.csv", ytestTr, delimiter=",")

    # calculate metrics
    p50_tagi = metric.computeND(ytestTr, ytestPd)
    p90_tagi = metric.compute90QL(ytestTr, ytestPd, SytestPd)
    RMSE_tagi = metric.computeRMSE(ytestTr, ytestPd)
    # MASE_tagi = metric.computeMASE(ytestTr, ytestPd, ytrain, seasonality) # TODO: check if ytrain is correct in MASE

    # save metrics into a text file
    with open(out_dir + "/electricity_2014_03_31_metrics.txt", "w") as f:
        f.write(f"ND/p50:    {p50_tagi}\n")
        f.write(f"p90:    {p90_tagi}\n")
        f.write(f"RMSE:    {RMSE_tagi}\n")
        # f.write(f'MASE:    {MASE_tagi}\n')
        f.write(f"Epoch:    {epoch_optim}\n")
        f.write(f"Batch size:    {batch_size}\n")
        f.write(f"Sigma_v:    {sigma_v}\n")
        f.write(f"LSTM nodes:    {lstm_nodes}\n")
        f.write(f"global_scale:    {test_dtl.global_scale}\n")

    # rename the directory
    out_dir_ = (
        "out/electricity_"
        + str(epoch)
        + "_"
        + str(batch_size)
        + "_"
        + str(round(sigma_v, 3))
        + "_"
        + str(lstm_nodes)
        + "_method2_seed"
        + str(np.random.randint(50))
    )
    os.rename(out_dir, out_dir_)


def concat_ts_sample(data, data_add):
    """Concatenate two time series samples"""
    x_combined = np.concatenate(
        (data.dataset["value"][0], data_add.dataset["value"][0]), axis=0
    )
    y_combined = np.concatenate(
        (data.dataset["value"][1], data_add.dataset["value"][1]), axis=0
    )
    time_combined = np.concatenate(
        (data.dataset["date_time"], data_add.dataset["date_time"])
    )
    # weights_combined = np.concatenate(
    #     (data.dataset["weights"], data_add.dataset["weights"])
    # )
    data.dataset["value"] = (x_combined, y_combined)
    data.dataset["date_time"] = time_combined
    # data.dataset["weights"] = weights_combined
    return data


def random_weighted_sampling(
    data: np.ndarray, weights: np.ndarray, num_samples: int
) -> np.ndarray:
    """Random-weighted sampling"""
    return np.random.choice(data, num_samples, p=weights)


if __name__ == "__main__":
    fire.Fire(main(sigma_v=None))
