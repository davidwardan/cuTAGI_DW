from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple, List, Dict

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
        date_time = self.load_data_from_csv(self.date_time_file)

        if self.ts_idx is not None:
            x = x[:, self.ts_idx : self.ts_idx + 1]
            date_time = date_time[:, self.ts_idx : self.ts_idx + 1]

            # remove trailing NaNs
            x, date_time = self._trim_trailing_nans(x, date_time)

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

    # Helpers
    @staticmethod
    def _trim_trailing_nans(
        x: np.ndarray, dt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Trim padded trailing NaNs in the *target* series, keep the same cut for datetime."""
        if len(x) == 0:
            return x, dt
        valid = ~np.isnan(x)
        if not np.any(valid):
            return np.array([], dtype=np.float32), np.array([], dtype="datetime64[ns]")
        last = np.where(valid)[0][-1]
        x = x[: last + 1]
        dt = dt[: last + 1]
        # ensure datetime dtype
        if not np.issubdtype(dt.dtype, np.datetime64):
            dt = np.array(dt, dtype="datetime64[ns]")
        return x.astype(np.float32), dt


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
    """
    Multi-TS dataloader with configurable ordering of (x, y) windows.

    Each column in x_file is a separate time series (target is col 0 before covariates are appended).
    date_time_file must have the same shape and correspond to x_file column-wise for covariates.

    order_mode:
        - "by_window": Take window #0 from ts0, ts1, ..., tsN-1; then window #1 from ts0, ts1, ..., tsN-1; etc.
                       (Preserves CSV series order; this is the default.)
        - "by_series": All windows of ts0, then all windows of ts1, ..., tsN-1.
    """

    def __init__(
        self,
        x_file: str,
        date_time_file: str,
        output_col: np.ndarray,
        input_seq_len: int,
        output_seq_len: int,
        num_features: int,  # target + covariates count expected (no embeddings)
        stride: int,
        scale_method: Optional[str] = None,  # 'standard' | 'deepAR' | None
        x_mean: Optional[List[float]] = None,
        x_std: Optional[List[float]] = None,
        scale_covariates: bool = True,  # Can set False for debugging purposes
        covariate_means: Optional[np.ndarray] = None,
        covariate_stds: Optional[np.ndarray] = None,
        time_covariates: Optional[
            List[str]
        ] = None,  # e.g. ["hour_of_day", "day_of_week"]
        keep_last_time_cov: bool = True,
        min_len_guard: bool = False,  # skip series shorter than in+out, instead of raising
        order_mode: str = "by_window",  # "by_window" | "by_series"
        random_seed: Optional[int] = None,  # Needed to allow for same shuffled batches
    ) -> None:
        self.x_file = x_file
        self.date_time_file = date_time_file
        self.output_col = output_col
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.stride = stride
        self.num_features = int(num_features)
        self.time_covariates = time_covariates or []
        self.keep_last_time_cov = keep_last_time_cov

        self.scale_method = (scale_method or "").strip().lower() or None
        self.x_mean = x_mean
        self.x_std = x_std
        self.scale_covariates = scale_covariates
        self.covariate_means = covariate_means
        self.covariate_stds = covariate_stds

        self.min_len_guard = min_len_guard

        self.order_mode = order_mode
        self._rng = np.random.default_rng(random_seed)

        # processed outputs
        self.dataset = self._process_all()

    @staticmethod
    def load_data_from_csv(data_file: str) -> np.ndarray:
        """Load CSV -> 2D numpy (T, N). Skips the first row (header)."""
        df = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
        return df.values

    def create_data_loader(
        self,
        batch_size: int,
        shuffle: bool = False,  # default False to preserve defined ordering
        weighted_sampling: bool = False,  # allows to sample batches depending on assigned weights
        weights: Optional[np.ndarray] = None,
        num_samples: Optional[int] = None,
        include_ids: bool = False,
        shuffle_series_blocks: bool = False,  # Will shuffle while respecting series blocks
    ) -> Generator[Tuple[np.ndarray, ...], None, None]:
        x_rolled, y_rolled = self.dataset["value"]
        series_id = self.dataset.get("series_id", None)
        window_id = self.dataset.get("window_id", None)

        # Prepare sampling indices
        n = x_rolled.shape[0]
        indices = np.arange(n)
        if shuffle:
            self._rng.shuffle(indices)

        if num_samples is not None:
            num_samples = min(n, num_samples)
            if weighted_sampling and (weights is not None):
                chosen = self._rng.choice(
                    indices, size=num_samples, replace=True, p=weights / np.sum(weights)
                )
            else:
                chosen = self._rng.choice(indices, size=num_samples, replace=False)
        else:
            chosen = indices

        # Code block to shuffle order of series per window
        if (
            self.order_mode == "by_window"
            and not shuffle
            and not weighted_sampling
            and num_samples is None
            and window_id is not None
            and len(window_id) == len(indices)
            and len(indices) > 0
            and shuffle_series_blocks
        ):
            boundaries = np.flatnonzero(np.diff(window_id)) + 1
            segments = (
                np.split(indices.copy(), boundaries)
                if boundaries.size
                else [indices.copy()]
            )
            for segment in segments:
                if segment.size > 1:
                    self._rng.shuffle(segment)
            indices = np.concatenate(segments)
            chosen = indices

        # Code block to shuffle order of series
        if (
            batch_size == 1
            and self.order_mode == "by_series"
            and not shuffle
            and not weighted_sampling
            and num_samples is None
            and series_id is not None
            and len(series_id) == len(indices)
            and len(indices) > 0
            and shuffle_series_blocks
        ):
            sid = series_id
            boundaries = np.flatnonzero(np.diff(sid)) + 1
            if boundaries.size:
                segments = np.split(indices, boundaries)
                perm = self._rng.permutation(len(segments))
                indices = np.concatenate([segments[i] for i in perm])
                chosen = indices

        total_batches = int(np.ceil(len(chosen) / batch_size))

        for b in range(total_batches):
            sl = chosen[b * batch_size : (b + 1) * batch_size]
            Xb = x_rolled[sl].reshape(len(sl), -1)
            Yb = y_rolled[sl].reshape(len(sl), -1)
            if include_ids and (series_id is not None) and (window_id is not None):
                Sb = series_id[sl]
                Kb = window_id[sl]
                yield Xb, Yb, Sb, Kb
            else:
                yield Xb, Yb

    # Core processing
    def _process_all(self) -> Dict[str, np.ndarray]:
        # Load matrices (T, N)
        X_all = self.load_data_from_csv(self.x_file)
        DT_all = self.load_data_from_csv(self.date_time_file)
        assert (
            X_all.shape == DT_all.shape
        ), "Values CSV and datetime CSV must have identical shapes (T, N)."

        T, N = X_all.shape  # T is time steps, N is number of series

        # Precompute covariates per series; also collect to compute global cov stats if needed
        per_ts_series = []
        cov_collect = []  # for global cov means/stds if scale_covariates is True

        for j in range(N):
            xj, dtj = X_all[:, j], DT_all[:, j]
            # TODO: debug this
            xj, dtj = self._trim_trailing_nans(
                xj, dtj
            )  # needed for some series that are shorter than T

            # Build [target, covs...]
            covs = self._time_covariates_from_datetime_column(dtj)

            if covs is None:
                Xj = xj.reshape(-1, 1).astype(np.float32)
            else:
                Xj = np.concatenate(
                    [xj.reshape(-1, 1), covs.astype(np.float32)], axis=1
                )

            per_ts_series.append({"X": Xj, "dt": dtj})
            if self.scale_covariates and Xj.shape[1] > 1:
                cov_collect.append(Xj[:, 1:])

        # Compute global cov means/stds if requested
        if self.scale_covariates and len(cov_collect) > 0:
            all_cov = np.concatenate([c for c in cov_collect if c.size > 0], axis=0)
            if self.covariate_means is None or self.covariate_stds is None:
                self.covariate_means = np.nanmean(all_cov, axis=0)
                self.covariate_stds = np.nanstd(all_cov, axis=0)
                self.covariate_stds[self.covariate_stds == 0.0] = 1.0

        # Compute per-series mean/std for 'standard' scaling
        if self.scale_method == "standard":
            # use if provided, else compute if missing
            need_compute = (
                self.x_mean is None
                or self.x_std is None
                or (
                    isinstance(self.x_mean, (list, np.ndarray))
                    and len(self.x_mean) == 0
                )
                or (isinstance(self.x_std, (list, np.ndarray)) and len(self.x_std) == 0)
            )
            if need_compute:
                per_means: List[float] = []
                per_stds: List[float] = []
                for s in per_ts_series:
                    t = s["X"][:, 0]
                    # robust to potential NaNs
                    mu = float(np.nanmean(t)) if t.size else 0.0
                    sd = float(np.nanstd(t)) if t.size else 1.0
                    if sd == 0.0 or np.isnan(sd):
                        sd = 1.0
                    per_means.append(mu)
                    per_stds.append(sd)
                self.x_mean = per_means
                self.x_std = per_stds

        # Roll per series (with scaling, cov scaling)
        rolled_per_ts = []
        scale_is_per_window = []  # deepAR weights, aligned later
        for j, s in enumerate(per_ts_series):
            Xj = s["X"].copy()

            # Covariate scaling
            if self.scale_covariates and Xj.shape[1] > 1:
                mu, sig = self.covariate_means, self.covariate_stds
                Xj[:, 1:] = (Xj[:, 1:] - mu) / sig

            # Target scaling
            scale_i_for_deepar = None
            if self.scale_method == "deepar":
                # per-series scale following deepAR style (mean abs + 1)
                scale_i_for_deepar = 1.0 + float(np.nanmean(np.abs(Xj[:, 0])))
                Xj[:, 0] = Xj[:, 0] / scale_i_for_deepar
            elif self.scale_method == "standard":
                # Per-series standardization
                if isinstance(self.x_mean, (list, np.ndarray)) and isinstance(
                    self.x_std, (list, np.ndarray)
                ):
                    mu_j = float(self.x_mean[j])
                    sd_j = float(self.x_std[j])
                else:
                    # Fallback if a single scalar was provided
                    mu_j = float(self.x_mean) if self.x_mean is not None else 0.0
                    sd_j = float(self.x_std) if self.x_std not in (None, 0.0) else 1.0
                if sd_j == 0.0 or np.isnan(sd_j):
                    sd_j = 1.0
                Xj[:, 0] = (Xj[:, 0] - mu_j) / sd_j

            # Roll windows for this series
            xw, yw = self._create_rolling_windows(
                Xj,
                input_seq_len=self.input_seq_len,
                output_seq_len=self.output_seq_len,
                stride=self.stride,
            )

            # Optionally keep only last-step covariates (depends on lstm definition)
            if self.keep_last_time_cov and Xj.shape[1] > 1:
                xw = self._keep_last_cov_only(xw, Xj.shape[1])

            rolled_per_ts.append((xw.astype(np.float32), yw.astype(np.float32)))

            # Store deepAR weights for each window (series-specific)
            if self.scale_method == "deepar":
                scale_is_per_window.append(
                    np.full((len(yw),), scale_i_for_deepar, dtype=np.float32)
                )
            else:
                scale_is_per_window.append(None)

        # ordering of windows (effective if shuffle=False)
        if self.order_mode == "by_window":
            X, Y, W, S, K = self._order_by_window_index(
                rolled_per_ts, scale_is_per_window
            )
        elif self.order_mode == "by_series":
            X, Y, W, S, K = self._order_by_series(rolled_per_ts, scale_is_per_window)
        else:
            raise ValueError(f"Unknown order_mode: {self.order_mode}")

        dataset: Dict[str, np.ndarray] = {
            "value": (X, Y),
            "series_id": S,
            "window_id": K,
        }
        if W is not None:
            dataset["weights"] = W
        dataset["date_time"] = None
        return dataset

    # Helpers
    @staticmethod
    def _trim_trailing_nans(
        x: np.ndarray, dt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Trim padded trailing NaNs in the *target* series, keep the same cut for datetime."""
        if len(x) == 0:
            return x, dt
        valid = ~np.isnan(x)
        if not np.any(valid):
            return np.array([], dtype=np.float32), np.array([], dtype="datetime64[ns]")
        last = np.where(valid)[0][-1]
        x = x[: last + 1]
        dt = dt[: last + 1]
        # ensure datetime dtype
        if not np.issubdtype(dt.dtype, np.datetime64):
            dt = np.array(dt, dtype="datetime64[ns]")
        return x.astype(np.float32), dt

    def _time_covariates_from_datetime_column(
        self, dt_col: np.ndarray
    ) -> Optional[np.ndarray]:
        """Build covariates for a single datetime column."""
        if not self.time_covariates:
            return None
        dt = dt_col.astype("datetime64[ns]")
        cols = []
        for cov in self.time_covariates:
            c = cov.lower()
            if c == "hour_of_day":
                cols.append(
                    (dt.astype("datetime64[h]").astype(int) % 24).reshape(-1, 1)
                )
            elif c == "day_of_week":
                cols.append((dt.astype("datetime64[D]").astype(int) % 7).reshape(-1, 1))
            elif c == "week_of_year":
                # ISO weeks are tricky; this approximates with 0-51
                cols.append(
                    ((dt.astype("datetime64[W]").astype(int) % 52) + 1).reshape(-1, 1)
                )
            elif c == "month_of_year":
                cols.append(
                    ((dt.astype("datetime64[M]").astype(int) % 12) + 1).reshape(-1, 1)
                )
            elif c == "quarter_of_year":
                month = (dt.astype("datetime64[M]").astype(int) % 12) + 1
                quarter = ((month - 1) // 3 + 1).reshape(-1, 1)
                cols.append(quarter)
            else:
                # silently skip unknown covariates
                pass
        if not cols:
            return None
        return np.concatenate(cols, axis=1).astype(np.float32)

    @staticmethod
    def _create_rolling_windows(
        X: np.ndarray,
        input_seq_len: int,
        output_seq_len: int,
        stride: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        X: (T, F) with first column the (scaled) target.
        Returns:
            x_windows: (Nw, input_seq_len * F)
            y_windows: (Nw, output_seq_len)    # only target for outputs
        """
        T, F = X.shape
        min_len = input_seq_len + output_seq_len
        if T < min_len:
            return np.empty((0, input_seq_len * F), dtype=np.float32), np.empty(
                (0, output_seq_len), dtype=np.float32
            )

        n_win = 1 + (T - min_len) // stride
        xw = np.zeros((n_win, input_seq_len * F), dtype=np.float32)
        yw = np.zeros((n_win, output_seq_len), dtype=np.float32)

        for i in range(n_win):
            s = i * stride
            x_slice = X[s : s + input_seq_len, :]  # (L, F)
            y_slice = X[
                s + input_seq_len : s + input_seq_len + output_seq_len, 0
            ]  # target only
            xw[i] = x_slice.reshape(-1)
            yw[i] = y_slice.reshape(-1)
        return xw, yw

    def _keep_last_cov_only(self, xw: np.ndarray, feat_count: int) -> np.ndarray:
        """
        From xw shaped (Nw, input_seq_len * F), keep:
          - all target values over the input window (the first column of each step),
          - and ONLY the covariates (F-1) from the **last** time step.
        Output shape: (Nw, input_seq_len + (F-1))
        """
        Nw = xw.shape[0]
        L = self.input_seq_len
        F = feat_count
        out = np.zeros((Nw, L + (F - 1)), dtype=np.float32)
        # indices of target over the window
        tgt_idx = np.arange(0, L * F, F)
        # last step covariates:
        last_cov_start = (L - 1) * F + 1
        last_cov = xw[:, last_cov_start : last_cov_start + (F - 1)]
        out[:, :L] = xw[:, tgt_idx]
        out[:, L:] = last_cov
        return out

    # Ordering helpers
    @staticmethod
    def _order_by_window_index(
        rolled_per_ts: List[Tuple[np.ndarray, np.ndarray]],
        per_ts_deepar_weights: List[Optional[np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        n_ts = len(rolled_per_ts)
        counts = [len(yw) for (_, yw) in rolled_per_ts]
        if max(counts, default=0) == 0:
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0, 0), dtype=np.float32),
                None,
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )

        sample_x = next((x for (x, y) in rolled_per_ts if len(y) > 0), None)
        sample_y = next((y for (x, y) in rolled_per_ts if len(y) > 0), None)
        feat_x = sample_x.shape[1]
        feat_y = sample_y.shape[1]

        total = sum(counts)
        X = np.zeros((total, feat_x), dtype=np.float32)
        Y = np.zeros((total, feat_y), dtype=np.float32)
        S = np.zeros((total,), dtype=np.int32)
        K = np.zeros((total,), dtype=np.int32)
        W = (
            np.zeros((total,), dtype=np.float32)
            if any(w is not None for w in per_ts_deepar_weights)
            else None
        )

        write = 0
        max_k = max(counts)
        for k in range(max_k):
            for j in range(n_ts):
                xj, yj = rolled_per_ts[j]
                if k < len(yj):
                    X[write] = xj[k]
                    Y[write] = yj[k]
                    S[write] = j
                    K[write] = k
                    if W is not None and per_ts_deepar_weights[j] is not None:
                        W[write] = per_ts_deepar_weights[j][k]
                    write += 1

        S, K = S[:write], K[:write]
        if W is not None:
            return X[:write], Y[:write], W[:write], S, K
        return X[:write], Y[:write], None, S, K

    @staticmethod
    def _order_by_series(
        rolled_per_ts: List[Tuple[np.ndarray, np.ndarray]],
        per_ts_deepar_weights: List[Optional[np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        xs, ys, ws = [], [], []
        S_parts, K_parts = [], []
        for j, (xj, yj) in enumerate(rolled_per_ts):
            n = len(yj)
            if n == 0:
                continue
            xs.append(xj)
            ys.append(yj)
            S_parts.append(np.full((n,), j, dtype=np.int32))
            K_parts.append(np.arange(n, dtype=np.int32))
            if per_ts_deepar_weights[j] is not None:
                ws.append(per_ts_deepar_weights[j])

        if not xs:
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0, 0), dtype=np.float32),
                None,
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )

        X = np.concatenate(xs, axis=0).astype(np.float32)
        Y = np.concatenate(ys, axis=0).astype(np.float32)
        S = np.concatenate(S_parts, axis=0)
        K = np.concatenate(K_parts, axis=0)
        W = np.concatenate(ws, axis=0).astype(np.float32) if ws else None

        return X, Y, W, S, K
