from typing import Any, List, Optional


class Config:
    def __init__(self):
        # Seed for reproducibility
        self.seed: int = 1

        # Set data paths
        self.x_train = "data/hq/train_1.0/split_train_values.csv"
        self.dates_train = "data/hq/train_1.0/split_train_datetimes.csv"
        self.x_val = "data/hq/split_val_values.csv"
        self.dates_val = "data/hq/split_val_datetimes.csv"
        self.x_test = "data/hq/split_test_values.csv"
        self.dates_test = "data/hq/split_test_datetimes.csv"

        # Set data_loader parameters
        self.num_features: int = 2
        self.time_covariates: list = ["week_of_year"]
        self.scale_method: str = "standard"
        self.order_mode: str = "by_window"
        self.input_seq_len: int = 52
        self.batch_size: int = 16
        self.output_col: list = [0]
        self.ts_to_use: Optional[List[int]] = [i for i in range(127)]  # Use all series

        # 1. For standard (one-per-series) embeddings:
        self.embedding_size: Optional[int] = None
        self.embedding_initializer: str = "normal"

        # 2. For mapped (shared) embeddings:
        self.embedding_map_dir: Optional[str] = None
        self.embedding_map_sizes = {
            "dam_id": 3,
            "dam_type_id": 3,
            "sensor_type_id": 3,
            "direction_id": 3,
            "sensor_id": 3,
        }
        self.embedding_map_initializer = {
            "dam_id": "normal",
            "dam_type_id": "normal",
            "sensor_type_id": "normal",
            "direction_id": "normal",
            "sensor_id": "normal",
        }
        self.embedding_map_labels = {
            "dam_id": ["DRU", "GOU", "LGA", "LTU", "MAT", "M5"],
            "dam_type_id": ["Run-of-River", "Reservoir"],
            "sensor_type_id": ["PIZ", "EXT", "PEN"],
            "direction_id": ["NA", "X", "Y", "Z"],
            "sensor_id": [f"sensor_{i}" for i in self.ts_to_use],
        }

        # Set model parameters
        self.Sigma_v_bounds: tuple = (None, None)
        self.decaying_factor: float = 0.99
        self.device: str = "cuda"
        self.lstm_hidden_sizes: List[int] = [40, 40]

        # Set training parameters
        self.num_epochs: int = 100
        self.early_stopping_criteria: str = "rmse"
        self.patience: int = 10
        self.min_delta: float = 1e-4
        self.warmup_epochs: int = 0
        self.shuffle: bool = True

        # Set evaluation parameters
        self.eval_plots: bool = False
        self.eval_metrics: bool = True
        self.seansonal_period: int = 52
        self.embed_plots: bool = False

    @property
    def x_file(self) -> list:
        """Dynamically creates the list of x files."""
        return [self.x_train, self.x_val, self.x_test]

    @property
    def date_file(self) -> list:
        """Dynamically creates the list of date files."""
        return [self.dates_train, self.dates_val, self.dates_test]

    @property
    def use_AGVI(self) -> bool:
        """Determines whether to use AGVI based on Sigma_v_bounds."""
        return self.Sigma_v_bounds[0] is None and self.Sigma_v_bounds[1] is None

    @property
    def use_mapped_embeddings(self) -> bool:
        """True if mapped embeddings are configured."""
        return self.embedding_map_dir is not None

    @property
    def use_standard_embeddings(self) -> bool:
        """True if standard (one-per-series) embeddings are configured."""
        return (
            not self.use_mapped_embeddings
            and self.embedding_size is not None
            and self.embedding_size > 0
        )

    @property
    def total_embedding_size(self) -> int:
        """Calculates the total embedding dimension based on configuration."""
        if self.use_mapped_embeddings:
            return sum(self.embedding_map_sizes.values())
        if self.use_standard_embeddings:
            return self.embedding_size
        return 0  # No embeddings

    @property
    def input_size(self) -> int:
        """Calculates the total input size for the model."""
        base_size = self.num_features + self.input_seq_len - 1
        return base_size + self.total_embedding_size

    @property
    def nb_ts(self) -> int:
        """Calculates the number of time series based on ts_to_use."""
        if self.ts_to_use is not None:
            return len(self.ts_to_use)
        return 1

    @property
    def plot_embeddings(self) -> bool:
        """
        Determines whether to plot embeddings.
        """
        return self.use_standard_embeddings

    def display(self):
        print("\nConfiguration:")
        # Display both regular attributes and properties
        for name in dir(self):
            if not name.startswith("_") and not callable(getattr(self, name)):
                print(f"  {name}: {getattr(self, name)}")
        print("\n")

    def save(self, path):
        with open(path, "w") as f:
            for name in dir(self):
                if not name.startswith("_") and not callable(getattr(self, name)):
                    f.write(f"{name}: {getattr(self, name)}\n")

    def load(self, path):
        # read config from file (simple key: value pairs)
        with open(path, "r") as f:
            for line in f:
                name, value = line.strip().split(": ", 1)
                if hasattr(self, name):
                    attr = getattr(self, name)
                    # Try to infer the type
                    if isinstance(attr, int):
                        value = int(value)
                    elif isinstance(attr, float):
                        value = float(value)
                    elif isinstance(attr, bool):
                        value = value.lower() == "true"
                    elif isinstance(attr, list):
                        value = eval(value)  # Caution: using eval
                    setattr(self, name, value)

    def wandb_dict(self):
        """Converts the configuration to a dictionary for W&B logging."""
        return {
            "seed": self.seed,
            "device": self.device,
            "epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "input_seq_len": self.input_seq_len,
            "time_covariates": self.time_covariates,
            "scale_method": self.scale_method,
            "order_mode": self.order_mode,
            "nb_ts": self.nb_ts,
            "training_size": self.x_train.split("/")[-2],
            "use_AGVI": self.use_AGVI,
            "Sigma_v_bounds": self.Sigma_v_bounds,
            "decaying_factor": self.decaying_factor,
            "embedding_type": (
                "mapped"
                if self.use_mapped_embeddings
                else "standard" if self.use_standard_embeddings else "none"
            ),
            "embedding_size": self.total_embedding_size,
        }
