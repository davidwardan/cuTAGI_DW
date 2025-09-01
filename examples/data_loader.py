from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd

from pytagi import Normalizer, Utils


class DataloaderBase(ABC):
    """Dataloader template"""

    normalizer: Normalizer = Normalizer()

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    @abstractmethod
    def process_data(self) -> dict:
        raise NotImplementedError

    def create_data_loader(self, raw_input: np.ndarray, raw_output: np.ndarray) -> list:
        """Create dataloader based on batch size"""
        num_input_data = raw_input.shape[0]
        num_output_data = raw_output.shape[0]
        assert num_input_data == num_output_data

        # Even indices
        even_indices = self.split_evenly(num_input_data, self.batch_size)

        if np.mod(num_input_data, self.batch_size) != 0:
            # Remider indices
            rem_indices = self.split_reminder(num_input_data, self.batch_size)
            even_indices.append(rem_indices)

        indices = np.stack(even_indices)
        input_data = raw_input[indices]
        output_data = raw_output[indices]
        dataset = []
        for x_batch, y_batch in zip(input_data, output_data):
            dataset.append((x_batch, y_batch))
        return dataset

    @staticmethod
    def split_data(data: int, test_ratio: float = 0.2, val_ratio: float = 0.0) -> dict:
        """Split data into training, validation, and test sets"""
        num_data = data.shape[1]
        splited_data = {}
        if val_ratio != 0.0:
            end_val_idx = num_data - int(test_ratio * num_data)
            end_train_idx = int(end_val_idx - val_ratio * end_val_idx)
            splited_data["train"] = data[:end_train_idx]
            splited_data["val"] = data[end_train_idx:end_val_idx]
            splited_data["test"] = data[end_val_idx:]
        else:
            end_train_idx = num_data - int(test_ratio * num_data)
            splited_data["train"] = data[:end_train_idx]
            splited_data["val"] = []
            splited_data["test"] = data[end_train_idx:]

        return splited_data

    @staticmethod
    def load_data_from_csv(data_file: str) -> pd.DataFrame:
        """Load data from csv file"""

        data = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

        return data.values

    @staticmethod
    def split_evenly(num_data, chunk_size: int):
        """split data evenly"""
        indices = np.arange(int(num_data - np.mod(num_data, chunk_size)))

        return np.split(indices, int(np.floor(num_data / chunk_size)))

    @staticmethod
    def split_reminder(num_data: int, chunk_size: int):
        """Pad the reminder"""
        indices = np.arange(num_data)
        reminder_start = int(num_data - np.mod(num_data, chunk_size))
        num_samples = chunk_size - (num_data - reminder_start)
        random_idx = np.random.choice(indices, size=num_samples, replace=False)
        reminder_idx = indices[reminder_start:]

        return np.concatenate((random_idx, reminder_idx))


class RegressionDataLoader:
    """Load and format data that are feeded to the neural network.
    The user must provide the input and output data file in *csv"""

    def __init__(
        self,
        x_file: str,
        y_file: str,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
        y_mean: Optional[np.ndarray] = None,
        y_std: Optional[np.ndarray] = None,
    ) -> None:
        self.x_file = x_file
        self.y_file = y_file
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

        self.dataset = self.process_data()

    def load_data_from_csv(self, data_file: str) -> pd.DataFrame:
        """Load data from csv file"""

        data = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

        return data.values

    @staticmethod
    def batch_generator(
        input_data: np.ndarray,
        output_data: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> Generator[Tuple[np.ndarray, ...], None, None]:
        """
        Generator function to yield batches of data.
        """
        num_data = input_data.shape[0]
        indices = np.arange(num_data)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, num_data, batch_size):
            # if start_idx + batch_size > num_data:
            #     continue
            end_idx = min(start_idx + batch_size, num_data)
            idx = indices[start_idx:end_idx]
            yield input_data[idx].flatten(), output_data[idx].flatten()

    def process_data(self) -> dict:
        """Process data from the csv file"""

        # Load data
        x = self.load_data_from_csv(self.x_file)
        y = self.load_data_from_csv(self.y_file)

        # Normalizer
        if self.x_mean is None:
            self.x_mean, self.x_std = Normalizer.compute_mean_std(x)
            self.y_mean, self.y_std = Normalizer.compute_mean_std(y)

        x = Normalizer.standardize(data=x, mu=self.x_mean, std=self.x_std)
        y = Normalizer.standardize(data=y, mu=self.y_mean, std=self.y_std)

        # Dataloader
        dataset = {}
        dataset["value"] = (np.float32(x), np.float32(y))

        return dataset

    def create_data_loader(self, batch_size: int, shuffle: bool = True):
        return self.batch_generator(*self.dataset["value"], batch_size, shuffle)


class MnistDataLoader:
    """Data loader for MNIST."""

    def __init__(
        self,
        x_file: Optional[str] = None,
        y_file: Optional[str] = None,
        num_images: int = None,
    ):
        if x_file is None and y_file is None:
            # Load from default location
            x_file = ("data/mnist/train-images-idx3-ubyte",)
            y_file = ("data/mnist/train-labels-idx1-ubyte",)
            num_images = 60000
        self.dataset = self.process_data(x_file, y_file, num_images)

    @staticmethod
    def load_mnist_images(image_file: str, label_file: str, num_images: int):
        """Load raw data"""
        utils = Utils()
        images, labels = utils.load_mnist_images(
            image_file=image_file, label_file=label_file, num_images=num_images
        )
        return images, labels

    @staticmethod
    def batch_generator(
        input_data: np.ndarray,
        output_data: np.ndarray,
        output_idx: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> Generator[Tuple[np.ndarray, ...], None, None]:
        """
        Generator function to yield batches of data.
        """
        num_data = input_data.shape[0]
        indices = np.arange(num_data)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, num_data, batch_size):
            end_idx = min(start_idx + batch_size, num_data)
            idx = indices[start_idx:end_idx]
            yield input_data[idx].flatten(), output_data[idx].flatten(), output_idx[
                idx
            ].flatten(), labels[idx].flatten()

    def process_data(
        self, x_train_file: str, y_train_file: str, num_images: int
    ) -> dict:
        """Process MNIST images."""
        # Load training and test data
        utils = Utils()
        images, labels = self.load_mnist_images(x_train_file, y_train_file, num_images)

        y, y_idx, num_enc_obs = utils.label_to_obs(labels=labels, num_classes=10)
        x_mean, x_std = Normalizer.compute_mean_std(images)
        x_std = 1

        # Normalizer
        x = Normalizer.standardize(data=images, mu=x_mean, std=x_std)

        y = np.float32(y.reshape((num_images, num_enc_obs)))
        y_idx = y_idx.reshape((num_images, num_enc_obs))
        x = x.reshape((num_images, 28, 28))

        return {"value": (x, y, y_idx, labels)}

    def create_data_loader(self, batch_size: int, shuffle: bool = True):
        return self.batch_generator(*self.dataset["value"], batch_size, shuffle)


class MnistOneHotDataloader(DataloaderBase):
    """Data loader for mnist dataset"""

    def process_data(
        self,
        x_train_file: str,
        y_train_file: str,
        x_test_file: str,
        y_test_file: str,
    ) -> dict:
        """Process mnist images"""
        # Initialization
        utils = Utils()
        num_train_images = 60000
        num_test_images = 10000

        # Traininng set
        train_images, train_labels = utils.load_mnist_images(
            image_file=x_train_file,
            label_file=y_train_file,
            num_images=num_train_images,
        )

        y_train = utils.label_to_one_hot(labels=train_labels, num_classes=10)
        x_mean, x_std = self.normalizer.compute_mean_std(train_images)
        x_std = 1

        # Test set
        test_images, test_labels = utils.load_mnist_images(
            image_file=x_test_file,
            label_file=y_test_file,
            num_images=num_test_images,
        )

        # Normalizer
        x_train = self.normalizer.standardize(data=train_images, mu=x_mean, std=x_std)
        x_test = self.normalizer.standardize(data=test_images, mu=x_mean, std=x_std)

        y_train = y_train.reshape((num_train_images, 10))
        x_train = x_train.reshape((num_train_images, 28, 28))
        x_test = x_test.reshape((num_test_images, 28, 28))

        # Data loader
        dataset = {}
        dataset["train"] = (x_train, y_train, train_labels)
        dataset["test"] = self.create_data_loader(
            raw_input=x_test, raw_output=test_labels
        )
        dataset["x_norm_param_1"] = x_mean
        dataset["x_norm_param_2"] = x_std

        return dataset


class TimeSeriesDataloader:
    """Data loader for time series"""

    def __init__(
        self,
        x_file: str,
        date_time_file: str,
        output_col: np.ndarray,
        input_seq_len: int,
        output_seq_len: int,
        num_features: int,
        stride: int,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
        ts_idx: Optional[int] = None,
        time_covariates: Optional[str] = None,
        keep_last_time_cov: Optional[bool] = False,
    ) -> None:
        self.x_file = x_file
        self.date_time_file = date_time_file
        self.output_col = output_col
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.num_features = num_features
        self.stride = stride
        self.x_mean = x_mean
        self.x_std = x_std
        self.ts_idx = ts_idx  # add time series index when data having multiple ts
        self.time_covariates = time_covariates  # for adding time covariates
        self.keep_last_time_cov = keep_last_time_cov
        self.dataset = self.process_data()

    def load_data_from_csv(self, data_file: str) -> pd.DataFrame:
        """Load data from csv file"""

        data = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

        return data.values

    @staticmethod
    def batch_generator(
        input_data: np.ndarray,
        output_data: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> Generator[Tuple[np.ndarray, ...], None, None]:
        """
        Generator function to yield batches of data.
        """
        num_data = input_data.shape[0]
        indices = np.arange(num_data)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, num_data, batch_size):
            # if start_idx + batch_size > num_data:
            #     continue
            end_idx = min(start_idx + batch_size, num_data)
            idx = indices[start_idx:end_idx]
            yield input_data[idx].flatten(), output_data[idx].flatten()

    def process_data(self) -> dict:
        """Process time series"""
        # Initialization
        utils = Utils()

        # Load data
        x = self.load_data_from_csv(self.x_file)
        if self.ts_idx is not None:
            x = x[:, self.ts_idx : self.ts_idx + 1]  # choose time series column
        date_time = self.load_data_from_csv(self.date_time_file)

        # Add time covariates
        if self.time_covariates is not None:
            date_time = np.array(date_time, dtype="datetime64")
            for time_cov in self.time_covariates:
                if time_cov == "hour_of_day":
                    hour_of_day = date_time.astype("datetime64[h]").astype(int) % 24
                    x = np.concatenate((x, hour_of_day), axis=1)
                elif time_cov == "day_of_week":
                    day_of_week = date_time.astype("datetime64[D]").astype(int) % 7
                    x = np.concatenate((x, day_of_week), axis=1)
                elif time_cov == "week_of_year":
                    week_of_year = (
                        date_time.astype("datetime64[W]").astype(int) % 52 + 1
                    )
                    x = np.concatenate((x, week_of_year), axis=1)
                elif time_cov == "month_of_year":
                    month_of_year = (
                        date_time.astype("datetime64[M]").astype(int) % 12 + 1
                    )
                    x = np.concatenate((x, month_of_year), axis=1)
                elif time_cov == "quarter_of_year":
                    month_of_year = (
                        date_time.astype("datetime64[M]").astype(int) % 12 + 1
                    )
                    quarter_of_year = (month_of_year - 1) // 3 + 1
                    x = np.concatenate((x, quarter_of_year), axis=1)

        # Normalizer
        if self.x_mean is None and self.x_std is None:
            self.x_mean, self.x_std = Normalizer.compute_mean_std(x)
        x = Normalizer.standardize(data=x, mu=self.x_mean, std=self.x_std)

        # Create rolling windows
        x_rolled, y_rolled = utils.create_rolling_window(
            data=x,
            output_col=self.output_col,
            input_seq_len=self.input_seq_len,
            output_seq_len=self.output_seq_len,
            num_features=self.num_features,
            stride=self.stride,
        )

        # remove time covariates, only keep the time cov at the last time step
        if self.keep_last_time_cov:
            x_rolled = self.remove_time_cov(x_rolled)

        # Dataloader
        dataset = {}
        dataset["value"] = (x_rolled, y_rolled)

        # NOTE: Datetime is saved for the visualization purpose
        dataset["date_time"] = [np.datetime64(date) for date in np.squeeze(date_time)]

        return dataset

    def remove_time_cov(self, x):
        x_new = np.zeros(
            (len(x), self.input_seq_len + self.num_features - 1),
            dtype=np.float32,
        )
        for i in range(0, len(x)):
            x_ = x[i]
            keep_idx = np.arange(0, len(x_), self.num_features)
            x_new[i] = np.concatenate((x_[keep_idx], x_[-self.num_features + 1 :]))
        return x_new

    def create_data_loader(self, batch_size: int, shuffle: bool = True):
        return self.batch_generator(*self.dataset["value"], batch_size, shuffle)


# class TimeSeriesDataloader:
#     """Data loader for time series"""

#     def __init__(
#         self,
#         x_file: str,
#         date_time_file: str,
#         output_col: np.ndarray,
#         input_seq_len: int,
#         output_seq_len: int,
#         num_features: int,
#         stride: int,
#         x_mean: Optional[np.ndarray] = None,
#         x_std: Optional[np.ndarray] = None,
#     ) -> None:
#         self.x_file = x_file
#         self.date_time_file = date_time_file
#         self.output_col = output_col
#         self.input_seq_len = input_seq_len
#         self.output_seq_len = output_seq_len
#         self.num_features = num_features
#         self.stride = stride
#         self.x_mean = x_mean
#         self.x_std = x_std

#         self.dataset = self.process_data()

#     def load_data_from_csv(self, data_file: str) -> pd.DataFrame:
#         """Load data from csv file"""

#         data = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

#         return data.values

#     @staticmethod
#     def batch_generator(
#         input_data: np.ndarray,
#         output_data: np.ndarray,
#         batch_size: int,
#         shuffle: bool = True,
#     ) -> Generator[Tuple[np.ndarray, ...], None, None]:
#         """
#         Generator function to yield batches of data.
#         """
#         num_data = input_data.shape[0]
#         indices = np.arange(num_data)
#         if shuffle:
#             np.random.shuffle(indices)

#         for start_idx in range(0, num_data, batch_size):
#             if start_idx + batch_size > num_data:
#                 continue
#             end_idx = min(start_idx + batch_size, num_data)
#             idx = indices[start_idx:end_idx]
#             yield input_data[idx].flatten(), output_data[idx].flatten()

#     def process_data(self) -> dict:
#         """Process time series"""
#         # Initialization
#         utils = Utils()

#         # Load data
#         x = self.load_data_from_csv(self.x_file)
#         date_time = self.load_data_from_csv(self.date_time_file)

#         # Normalizer
#         if self.x_mean is None and self.x_std is None:
#             self.x_mean, self.x_std = Normalizer.compute_mean_std(x)
#         x = Normalizer.standardize(data=x, mu=self.x_mean, std=self.x_std)

#         # Create rolling windows
#         x_rolled, y_rolled = utils.create_rolling_window(
#             data=x,
#             output_col=self.output_col,
#             input_seq_len=self.input_seq_len,
#             output_seq_len=self.output_seq_len,
#             num_features=self.num_features,
#             stride=self.stride,
#         )

#         # Dataloader
#         dataset = {}
#         dataset["value"] = (x_rolled, y_rolled)

#         # NOTE: Datetime is saved for the visualization purpose
#         dataset["date_time"] = [np.datetime64(date) for date in np.squeeze(date_time)]

#         return dataset

#     def create_data_loader(self, batch_size: int, shuffle: bool = True):
#         return self.batch_generator(*self.dataset["value"], batch_size, shuffle)


class GlobalTimeSeriesDataloader:
    """Similar to TimeSeriesDataloader but with global normalization"""

    def __init__(
        self,
        x_file: str,
        date_time_file: str,
        output_col: np.ndarray,
        input_seq_len: int,
        output_seq_len: int,
        num_features: int,
        stride: int,
        ts_idx: Optional[int] = 0,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
        time_covariates: Optional[str] = None,
        scale_i: Optional[float] = None,
        global_scale: Optional[str] = None,  # other options: 'standard', 'deepAR'
        idx_as_feature: Optional[bool] = False,
        min_max_scaler: Optional[list] = None,
        scale_covariates: Optional[bool] = False,
        covariate_means: Optional[np.ndarray] = None,
        covariate_stds: Optional[np.ndarray] = None,
        embedding_dim: Optional[int] = None,
        embedding: Optional[np.ndarray] = None,
        embed_at_end: Optional[
            bool
        ] = False,  # if True, embedding is added at the end of the input sequence
        skip_if_invalid: Optional[bool] = False,
        fillna: Optional[str] = None,
    ) -> None:
        self.x_file = x_file
        self.date_time_file = date_time_file
        self.output_col = output_col
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.num_features = num_features
        self.stride = stride
        self.ts_idx = ts_idx  # add time series index when data having multiple ts
        self.time_covariates = time_covariates  # for adding time covariates
        self.scale_i = scale_i  # scaling factor for ith time series
        self.global_scale = global_scale
        self.x_mean = x_mean
        self.x_std = x_std
        self.idx_as_feature = idx_as_feature
        self.min_max_scaler = min_max_scaler
        self.scale_covariates = scale_covariates
        self.covariate_means = covariate_means
        self.covariate_stds = covariate_stds
        self.embedding_dim = embedding_dim
        self.embedding = embedding
        self.embed_at_end = embed_at_end
        self.skip_if_invalid = skip_if_invalid
        self.fillna = fillna
        self.dataset = self.process_data()
    @staticmethod
    def _repair_nans(x: np.ndarray, method: Optional[str] = None) -> np.ndarray:
        """Optionally repair NaNs in 2-D array.
        method: None | 'ffill_bfill' | 'zero'
        """
        if method is None:
            return x
        x = x.copy()
        if method == 'zero':
            x[~np.isfinite(x)] = 0.0
            return x
        if method == 'ffill_bfill':
            # forward fill then back fill per column
            n, m = x.shape
            for j in range(m):
                col = x[:, j]
                mask = ~np.isfinite(col)
                if mask.all():
                    # all-NaN/Inf: set to zero to avoid breakage
                    col[:] = 0.0
                else:
                    # forward fill
                    idx = np.where(~mask, np.arange(n), 0)
                    np.maximum.accumulate(idx, out=idx)
                    col = col[idx]
                    # back fill leading NaNs (if any remained because first values were NaN)
                    first_valid = np.where(~np.isnan(col) & np.isfinite(col))[0]
                    if first_valid.size:
                        col[: first_valid[0]] = col[first_valid[0]]
                x[:, j] = col
            return x
        # unknown method -> no change
        return x

    @staticmethod
    def load_data_from_csv(data_file: str) -> np.ndarray:
        """Load data from csv file"""

        data = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

        return data.values

    @staticmethod
    def batch_generator(
        input_data: np.ndarray,
        output_data: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        weights: Optional[np.ndarray] = None,
        num_samples: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, ...], None, None]:
        """
        Generator function to yield batches of data.
        """

        num_data = input_data.shape[0]
        indices = np.arange(num_data)
        if shuffle:
            np.random.shuffle(indices)

        if num_samples is not None:
            num_samples = min(num_data, num_samples)
            if weights is None:
                selected_indices = np.random.choice(
                    indices, size=num_samples, replace=False
                )
            else:
                selected_indices = np.random.choice(
                    indices, size=num_samples, replace=True, p=weights
                )
        else:
            selected_indices = indices

        # Calculate the number of batches
        total_batches = int(np.ceil(len(selected_indices) / batch_size))

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = start_idx + batch_size
            batch_indices = selected_indices[start_idx:end_idx]
            yield input_data[batch_indices].flatten(), output_data[
                batch_indices
            ].flatten()

    @staticmethod
    def _strip_trailing_rows_sync(x: np.ndarray, dt: Optional[np.ndarray]):
        """Trim trailing rows where the *first column* of x is NaN.
        Returns (x_trimmed, dt_trimmed_or_none).
        """
        if x.ndim == 1:
            mask = ~np.isnan(x)
            if not mask.any():
                raise ValueError("[GlobalTimeSeriesDataloader] series is all-NaN")
            last = np.where(mask)[0].max()
            x_trim = x[: last + 1]
            dt_trim = dt[: last + 1] if dt is not None else None
            return x_trim, dt_trim
        else:
            # use leading/target column (col 0) to decide valid extent
            mask = ~np.isnan(x[:, 0])
            if not mask.any():
                raise ValueError("[GlobalTimeSeriesDataloader] series is all-NaN in first column")
            last = np.where(mask)[0].max()
            x_trim = x[: last + 1]
            dt_trim = dt[: last + 1] if dt is not None else None
            return x_trim, dt_trim

    @staticmethod
    def _validate_series_for_windows(
        x: np.ndarray,
        input_len: int,
        output_len: int,
        stride: int,
        num_features: int,
    ) -> np.ndarray:
        """Ensure x is 2-D, finite, long enough, correct feature count, and float32 contiguous."""
        if stride <= 0:
            raise ValueError(f"[GlobalTimeSeriesDataloader] stride must be > 0 (got {stride})")

        if x.ndim != 2:
            raise ValueError(f"[GlobalTimeSeriesDataloader] expected 2-D array, got shape {x.shape}")

        if x.shape[1] != num_features:
            raise ValueError(
                f"[GlobalTimeSeriesDataloader] num_features mismatch: x has {x.shape[1]} columns, expected {num_features}. "
                "Make sure `num_features` includes covariates that were concatenated."
            )

        if not np.isfinite(x).all():
            # find first few offending indices for easier debugging
            bad = np.argwhere(~np.isfinite(x))
            bad_list = [tuple(b) for b in bad[:5]]
            raise ValueError(f"[GlobalTimeSeriesDataloader] non-finite values detected at {bad_list} (showing up to 5)")

        min_len = input_len + output_len
        if x.shape[0] < min_len:
            raise ValueError(
                f"[GlobalTimeSeriesDataloader] too short after trimming: len={x.shape[0]} < {min_len} (input+output)"
            )

        return np.ascontiguousarray(x.astype(np.float32))

    def process_data(self) -> dict:
        """Process time series"""
        # Initialization
        utils = Utils()

        # Load data
        x = self.load_data_from_csv(self.x_file)
        date_time = self.load_data_from_csv(self.date_time_file)
        # choose target column
        x = x[:, self.ts_idx : self.ts_idx + 1]
        date_time = date_time[:, self.ts_idx : self.ts_idx + 1]

        # Trim trailing padded NaNs using the target (first) column, and keep date_time in sync
        x, date_time = self._strip_trailing_rows_sync(x, date_time)


        # Add time covariates
        if self.time_covariates is not None:
            date_time = np.array(date_time, dtype="datetime64")
            for time_cov in self.time_covariates:
                if time_cov == "hour_of_day":
                    hour_of_day = date_time.astype("datetime64[h]").astype(int) % 24
                    x = np.concatenate((x, hour_of_day), axis=1)
                elif time_cov == "day_of_week":
                    day_of_week = date_time.astype("datetime64[D]").astype(int) % 7
                    x = np.concatenate((x, day_of_week), axis=1)
                elif time_cov == "week_of_year":
                    week_of_year = (
                        date_time.astype("datetime64[W]").astype(int) % 52 + 1
                    )
                    x = np.concatenate((x, week_of_year), axis=1)
                elif time_cov == "month_of_year":
                    month_of_year = (
                        date_time.astype("datetime64[M]").astype(int) % 12 + 1
                    )
                    x = np.concatenate((x, month_of_year), axis=1)
                elif time_cov == "quarter_of_year":
                    month_of_year = (
                        date_time.astype("datetime64[M]").astype(int) % 12 + 1
                    )
                    quarter_of_year = (month_of_year - 1) // 3 + 1
                    x = np.concatenate((x, quarter_of_year), axis=1)

        # standardize covariates
        if self.scale_covariates:
            # Compute means and stds for covariates if missing
            if self.covariate_means is None or self.covariate_stds is None:
                self.covariate_means = np.nanmean(x, axis=0)
                self.covariate_stds = np.nanstd(x, axis=0)
            # Avoid division by zero for any zero std entries
            cov_idx = slice(1, self.num_features)
            stds = self.covariate_stds[cov_idx]
            stds = np.where(stds == 0.0, 1.0, stds)
            # Standardize all covariate columns at once
            x[:, cov_idx] = (x[:, cov_idx] - self.covariate_means[cov_idx]) / stds

        # scale the observations using time series dependent scaling factors
        if self.global_scale:
            mode = self.global_scale.strip().lower()

            # --- deepAR scaling ---
            if mode == "deepar":
                if self.scale_i is None:
                    # deepAR often uses mean absolute value; here we add 1 to avoid zero‐division
                    self.scale_i = 1.0 + np.nanmean(x[:, 0])
                x[:, 0] /= self.scale_i

            # --- standard z‑score scaling ---
            elif mode == "standard":
                # recompute if either is missing
                if self.x_mean is None or self.x_std is None:
                    self.x_mean, self.x_std = Normalizer.compute_mean_std(x[:, 0])
                x[:, 0] = Normalizer.standardize(
                    data=x[:, 0], mu=self.x_mean, std=self.x_std
                )

        # Optional NaN repair
        x = self._repair_nans(x, method=self.fillna)

        # Finalize features and validate before windowing
        try:
            x = np.asarray(x, dtype=np.float32)  # ensures float32
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            # If num_features was passed incorrectly, fail fast with a clear message
            if self.num_features is None:
                self.num_features = x.shape[1]

            # Validate series (length, finiteness, matching num_features, stride)
            # x = self._validate_series_for_windows(
            #     x=x,
            #     input_len=self.input_seq_len,
            #     output_len=self.output_seq_len,
            #     stride=self.stride,
            #     num_features=self.num_features,
            # )

            # Create rolling windows
            x_rolled, y_rolled = utils.create_rolling_window(
                data=x,
                output_col=self.output_col,
                input_seq_len=self.input_seq_len,
                output_seq_len=self.output_seq_len,
                num_features=self.num_features,
                stride=self.stride,
            )

            x_rolled = np.ascontiguousarray(x_rolled, dtype=np.float32)
            y_rolled = np.ascontiguousarray(y_rolled, dtype=np.float32)
        except Exception as e:
            if self.skip_if_invalid:
                print(f"[WARN] Skipping TS {self.ts_idx} due to: {e}")
                # Produce empty shapes consistent with expected downstream usage
                x_rolled = np.zeros((0, self.input_seq_len * self.num_features), dtype=np.float32)
                y_rolled = np.zeros((0, self.output_seq_len), dtype=np.float32)
            else:
                raise

        # Create embedding for time series index
        if self.embedding_dim is not None or self.embedding is not None:
            if self.embedding_dim is not None and self.embedding is None:
                self.embedding = np.full((self.embedding_dim,), self.ts_idx)
            if self.embed_at_end:
                # TODO: add embedding at the end of the sequence
                x_rolled = np.concatenate(
                    [
                        x_rolled,
                        np.tile(self.embedding, (x_rolled.shape[0], 1)),
                    ],
                    axis=1,
                )
            else:
                x_rolled = self.roll_embedding(x_rolled, self.embedding)

        # Dataloader
        dataset = {}
        dataset["value"] = (x_rolled, y_rolled)

        # NOTE: Datetime is saved for the visualization purpose
        # dataset["date_time"] = [np.datetime64(date) for date in np.squeeze(date_time)]

        # store weights for weighted sampling. Default is uniform sampling
        if self.global_scale == "deepAR":
            dataset["weights"] = self.scale_i * np.ones(x_rolled.shape[0])

        return dataset

    def roll_embedding(self, x_rolled: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        # Assuming embedding is a 1D vector; if not, adjust accordingly
        emb_dim = embedding.shape[0] if embedding.ndim == 1 else embedding.shape[1]

        # Reshape the rolled data into a (num_data, seq_len, features) array.
        x_rolled_reshaped = x_rolled.reshape(
            x_rolled.shape[0], self.input_seq_len, self.num_features
        )

        # Reshape and broadcast the embedding vector to match the sequence length.
        embedding_vector = embedding.reshape(1, 1, -1)
        embedding_broadcasted = np.broadcast_to(
            embedding_vector, (x_rolled.shape[0], self.input_seq_len, emb_dim)
        )

        # Concatenate the original features and the embedding along the last dimension.
        concatenated = np.concatenate(
            [x_rolled_reshaped, embedding_broadcasted], axis=2
        )

        # Reshape back to a 2D array with shape (x_rolled.shape[0], self.input_seq_len * (features + emb_dim)).
        new_x_rolled = concatenated.reshape(
            x_rolled.shape[0], self.input_seq_len * (self.num_features + emb_dim)
        )

        return new_x_rolled.astype(np.float32)

    def create_data_loader(
        self,
        batch_size: int,
        shuffle: bool = True,
        weighted_sampling: bool = False,
        weights: Optional[np.ndarray] = None,
        num_samples: Optional[int] = None,
    ):
        return self.batch_generator(
            *self.dataset["value"],
            batch_size,
            shuffle,
            weights,
            num_samples,
        )
