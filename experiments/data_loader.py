from typing import Generator, Optional, Tuple, List, Dict
import numpy as np
import pandas as pd

from pytagi import Normalizer

import warnings


class GlobalTimeSeriesDataloader:
    def __init__(
        self,
        x_file: str,
        date_time_file: str,
        input_seq_len: int,
        output_seq_len: int,
        stride: int,
        order_mode: str = "by_window",
        scale_method: Optional[str] = None,
        x_mean: Optional[
            List[np.ndarray]
        ] = None,  # List of means, one array per series
        x_std: Optional[List[np.ndarray]] = None,  # List of stds, one array per series
        time_covariates: Optional[List[str]] = None,
        ts_to_use: Optional[List[int]] = None,
    ) -> None:
        self.x_file = x_file
        self.date_time_file = date_time_file
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.stride = stride
        self.order_mode = order_mode
        self.scale_method = scale_method
        self.x_mean = x_mean
        self.x_std = x_std
        self.time_covariates = time_covariates or []
        self.ts_to_use = ts_to_use
        self.num_features = 1 + len(self.time_covariates)

        # Metadata
        self.max_len = None

        # processed outputs
        self.dataset = self._process_all()

    # --- Data Processing Helpers ---
    @staticmethod
    def _load_data_from_csv(
        data_file: str, col_to_use: Optional[List[int]] = None
    ) -> np.ndarray:
        """Load CSV -> 2D numpy (T, N). Skips the first row (header)."""
        df = pd.read_csv(
            data_file, skiprows=1, delimiter=",", header=None, usecols=col_to_use
        )
        return df.values

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
        if not np.issubdtype(dt.dtype, np.datetime64):
            dt = np.array(dt, dtype="datetime64[ns]")
        return x.astype(np.float32), dt

    @staticmethod
    def _create_rolling_windows(
        X: np.ndarray,
        input_seq_len: int,
        output_seq_len: int,
        stride: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        X: (T, F) with first column = (scaled) target, and remaining columns = time covariates.
        Keeps time covariates only from the last timestep in each input window.
        """
        T, F = X.shape
        num_cov = F - 1
        min_len = input_seq_len + output_seq_len

        if T < min_len:
            return (
                np.empty((0, input_seq_len + num_cov), dtype=np.float32),
                np.empty((0, output_seq_len), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        n_win = 1 + (T - min_len) // stride
        xw = np.zeros((n_win, input_seq_len + num_cov), dtype=np.float32)
        yw = np.zeros((n_win, output_seq_len), dtype=np.float32)
        window_id = np.zeros(n_win, dtype=np.int32)

        for i in range(n_win):
            s = i * stride
            x_slice = X[s : s + input_seq_len, 0]  # target only
            cov_last = X[s + input_seq_len - 1, 1:] if num_cov > 0 else np.empty(0)
            y_slice = X[s + input_seq_len : s + input_seq_len + output_seq_len, 0]

            xw[i, :input_seq_len] = x_slice
            if num_cov > 0:
                xw[i, input_seq_len:] = cov_last
            yw[i] = y_slice
            window_id[i] = s  # index of the start of this window

        return xw, yw, window_id

    def _time_covariates_from_datetime_column(
        self, dt_col: np.ndarray
    ) -> Optional[np.ndarray]:
        """Build covariates for a single datetime column."""
        if not self.time_covariates:
            return None
        if not np.issubdtype(dt_col.dtype, np.datetime64):
            dt = dt_col.astype("datetime64[ns]")
        else:
            dt = dt_col

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
                dt_series = pd.Series(dt)
                cols.append(dt_series.dt.isocalendar().week.values.reshape(-1, 1))
            elif c == "month_of_year":
                cols.append(
                    ((dt.astype("datetime64[M]").astype(int) % 12) + 1).reshape(-1, 1)
                )
            elif c == "quarter_of_year":
                month = (dt.astype("datetime64[M]").astype(int) % 12) + 1
                quarter = ((month - 1) // 3 + 1).reshape(-1, 1)
                cols.append(quarter)
            else:
                assert False, f"Unknown time covariate: {cov}"

        if not cols:
            return None
        return np.concatenate(cols, axis=1).astype(np.float32)

    def _standard_scale_series(
        self, X: np.ndarray, mu: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        """Standard scale a single series X (T, F) with provided mu, sd (F,)."""
        # Add a small epsilon to std dev to avoid division by zero
        X_scaled = Normalizer.standardize(X, mu=mu, std=std)
        return X_scaled

    def _process_all(self) -> Dict[str, np.ndarray]:
        X_all = self._load_data_from_csv(self.x_file, col_to_use=self.ts_to_use)
        DT_all = self._load_data_from_csv(
            self.date_time_file, col_to_use=self.ts_to_use
        )

        assert (
            X_all.shape == DT_all.shape
        ), "Values CSV and datetime CSV must have identical shapes (T, N)."

        T, N = X_all.shape
        self.max_len = T
        ts_indices = self.ts_to_use if self.ts_to_use is not None else list(range(N))

        rolled_per_ts = []
        scaling_info_per_ts = []  # To store calculated (mu, sd) tuples

        for j in range(N):
            xj, dtj = X_all[:, j], DT_all[:, j]
            xj, dtj = self._trim_trailing_nans(xj, dtj)
            series_idx = ts_indices[j]
            covs = self._time_covariates_from_datetime_column(dtj)

            if covs is None:
                Xj = xj.reshape(-1, 1).astype(np.float32)
            else:
                Xj = np.concatenate(
                    [xj.reshape(-1, 1), covs.astype(np.float32)], axis=1
                )

            if Xj.shape[0] == 0:
                continue

            if Xj.shape[1] != self.num_features:
                raise RuntimeError(
                    f"Internal error: Feature count mismatch for Series {series_idx}. "
                    f"Expected {self.num_features}, got {Xj.shape[1]}."
                )

            if self.scale_method == "standard":
                if self.x_mean is None or self.x_std is None:
                    # Calculate and store stats
                    mu = np.nanmean(Xj, axis=0)
                    std = np.nanstd(Xj, axis=0)
                    scaling_info_per_ts.append((mu, std))
                else:
                    # Use provided stats
                    if len(self.x_mean) != N or len(self.x_std) != N:
                        raise ValueError(
                            f"x_mean/x_std lists must have length {N} (matching loaded series),"
                            f" but got {len(self.x_mean)}/{len(self.x_std)}."
                        )
                    mu = self.x_mean[j]
                    std = self.x_std[j]

                if len(mu) != self.num_features or len(std) != self.num_features:
                    raise ValueError(
                        f"Series {series_idx}: Mismatch in feature count for scaling. "
                        f"Data has {self.num_features} features, but mean/std have {len(mu)}/{len(std)}."
                    )

                Xj = self._standard_scale_series(Xj, mu, std)
            elif self.scale_method is not None:
                raise ValueError(f"Unknown scale_method: {self.scale_method}")

            x_rolled, y_rolled, window_ids = self._create_rolling_windows(
                Xj,
                input_seq_len=self.input_seq_len,
                output_seq_len=self.output_seq_len,
                stride=self.stride,
            )

            series_ids = np.full((len(window_ids),), series_idx, dtype=np.int32)

            rolled_per_ts.append(
                (
                    x_rolled.astype(np.float32),
                    y_rolled.astype(np.float32),
                    window_ids.astype(np.int32),
                    series_ids.astype(np.int32),
                )
            )

        if (
            self.scale_method == "standard"
            and self.x_mean is None
            and self.x_std is None
            and scaling_info_per_ts  # Check list is not empty
        ):
            means, stds = zip(*scaling_info_per_ts)
            self.x_mean = list(means)
            self.x_std = list(stds)

        # --- Concatenation Logic ---
        if self.order_mode == "by_series":
            xs, ys = [], []
            S_parts, K_parts = [], []
            for j, (xj, yj, kj, sj) in enumerate(rolled_per_ts):
                if len(yj) == 0:
                    continue
                xs.append(xj)
                ys.append(yj)
                K_parts.append(kj)
                S_parts.append(sj)

            if not xs:
                return {
                    "value": (
                        np.empty(
                            (0, self.input_seq_len + self.num_features - 1),
                            dtype=np.float32,
                        ),
                        np.empty((0, self.output_seq_len), dtype=np.float32),
                    ),
                    "series_id": np.empty((0,), dtype=np.int32),
                    "window_id": np.empty((0,), dtype=np.int32),
                }

            X = np.concatenate(xs, axis=0).astype(np.float32)
            Y = np.concatenate(ys, axis=0).astype(np.float32)
            S = np.concatenate(S_parts, axis=0)
            K = np.concatenate(K_parts, axis=0)

            return {
                "value": (X, Y),
                "series_id": S,
                "window_id": K,
            }

        elif self.order_mode == "by_window":
            n_ts = len(rolled_per_ts)
            counts = [len(yw) for (_, yw, _, _) in rolled_per_ts]
            if max(counts, default=0) == 0:
                return {
                    "value": (
                        np.empty(
                            (0, self.input_seq_len + self.num_features - 1),
                            dtype=np.float32,
                        ),
                        np.empty((0, self.output_seq_len), dtype=np.float32),
                    ),
                    "series_id": np.empty((0,), dtype=np.int32),
                    "window_id": np.empty((0,), dtype=np.int32),
                }

            feat_x = self.input_seq_len + self.num_features - 1
            feat_y = self.output_seq_len
            total = sum(counts)

            X = np.zeros((total, feat_x), dtype=np.float32)
            Y = np.zeros((total, feat_y), dtype=np.float32)
            S = np.zeros((total,), dtype=np.int32)
            K = np.zeros((total,), dtype=np.int32)

            write = 0
            max_k = max(counts)
            for k in range(max_k):
                for j in range(n_ts):
                    xj, yj, kj, sj = rolled_per_ts[j]
                    if k < len(yj):
                        X[write] = xj[k]
                        Y[write] = yj[k]
                        S[write] = sj[k]
                        K[write] = kj[k]
                        write += 1

            S, K = S[:write], K[:write]

            return {
                "value": (X[:write], Y[:write]),
                "series_id": S,
                "window_id": K,
            }
        else:
            raise ValueError(f"Unknown order_mode: {self.order_mode}")

    # --- Data Loader Functions ---
    @staticmethod
    def _loader_by_window(
        X: np.ndarray,
        Y: np.ndarray,
        S: np.ndarray,
        K: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> Generator[
        Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray], None, None
    ]:
        """
        Generator for 'by_window' mode.
        - Batches contain samples with the *same* window_id.
        - Iterates sequentially through window_ids (k=0, k=1, k=2...).
        - If shuffle=True, shuffles the series *within* each window_id group.
        - Pads the last batch of each window group with NaN (for X/Y),
          -1 (for S), and k_id (for K) to ensure it fills the batch_size.
        """
        if len(X) == 0:
            return

        x_feat_dim = X.shape[1]
        y_feat_dim = Y.shape[1]
        unique_window_ids, start_indices = np.unique(K, return_index=True)
        end_indices = np.append(start_indices[1:], len(K))

        for k_id, start, end in zip(unique_window_ids, start_indices, end_indices):
            x_window_group = X[start:end]
            y_window_group = Y[start:end]
            s_window_group = S[start:end]
            k_window_group = K[start:end]

            num_series_in_group = len(x_window_group)
            group_indices = np.arange(num_series_in_group)

            if shuffle:
                np.random.shuffle(group_indices)

            for i in range(0, num_series_in_group, batch_size):
                batch_indices = group_indices[i : i + batch_size]

                x_batch_real = x_window_group[batch_indices]
                y_batch_real = y_window_group[batch_indices]
                s_batch_real = s_window_group[batch_indices]
                k_batch_real = k_window_group[batch_indices]

                current_batch_size = len(x_batch_real)
                num_to_pad = batch_size - current_batch_size

                if num_to_pad > 0:
                    x_pad = np.full(
                        (num_to_pad, x_feat_dim), np.nan, dtype=x_batch_real.dtype
                    )
                    y_pad = np.full(
                        (num_to_pad, y_feat_dim), np.nan, dtype=y_batch_real.dtype
                    )
                    s_pad = np.full((num_to_pad,), -1, dtype=s_batch_real.dtype)
                    k_pad = np.full((num_to_pad,), k_id, dtype=k_batch_real.dtype)

                    x_batch = np.concatenate((x_batch_real, x_pad), axis=0)
                    y_batch = np.concatenate((y_batch_real, y_pad), axis=0)
                    s_batch = np.concatenate((s_batch_real, s_pad), axis=0)
                    k_batch = np.concatenate((k_batch_real, k_pad), axis=0)

                    yield (x_batch, y_batch), s_batch, k_batch
                else:
                    yield (x_batch_real, y_batch_real), s_batch_real, k_batch_real

    @staticmethod
    def _loader_by_series(
        X: np.ndarray,
        Y: np.ndarray,
        S: np.ndarray,
        K: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> Generator[
        Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray], None, None
    ]:
        """
        Generator for 'by_series' mode.
        - Follows the rule: Warns and forces batch_size=1.
        - Iterates through all windows of one series, then all windows of the next.
        - If shuffle=True, shuffles the order of the series.
        """
        if batch_size > 1:
            warnings.warn(
                f"Warning: 'by_series' mode only supports batch_size=1. "
                f"Forcing batch_size=1."
            )
            batch_size = 1

        unique_series_ids = np.unique(S)
        if shuffle:
            np.random.shuffle(unique_series_ids)

        for series_id in unique_series_ids:
            (series_indices,) = np.where(S == series_id)
            for idx in series_indices:
                x_batch = X[idx : idx + 1]
                y_batch = Y[idx : idx + 1]
                s_batch = S[idx : idx + 1]
                k_batch = K[idx : idx + 1]

                yield (x_batch, y_batch), s_batch, k_batch

    @staticmethod
    def create_data_loader(
        dataset: Dict[str, np.ndarray],
        order_mode: str,
        batch_size: int,
        shuffle: bool = False,
    ) -> Generator[
        Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray], None, None
    ]:
        """
        Creates a batch generator from a global time series dataset.

        Usage:
        loader_instance = GlobalTimeSeriesDataloader(...)
        my_dataset = loader_instance.dataset

        # Call the static method on the class itself
        loader = GlobalTimeSeriesDataloader.create_data_loader(
            my_dataset, 'by_window', 32
        )
        for (x, y), s, k in loader:
            #... training logic ...
        """
        (X, Y) = dataset["value"]
        S = dataset["series_id"]
        K = dataset["window_id"]

        if len(X) == 0:
            return

        # Call the other static methods using the class name
        if order_mode == "by_series":
            yield from GlobalTimeSeriesDataloader._loader_by_series(
                X, Y, S, K, batch_size, shuffle
            )
        elif order_mode == "by_window":
            yield from GlobalTimeSeriesDataloader._loader_by_window(
                X, Y, S, K, batch_size, shuffle
            )
        else:
            raise ValueError(f"Unknown order_mode: {order_mode}")
