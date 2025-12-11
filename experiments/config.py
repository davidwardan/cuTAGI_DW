from typing import Any, List, Optional, Dict
import yaml
from pydantic import BaseModel, Field


class DataPaths(BaseModel):
    x_train: str = "data/hq/train100/split_train_values.csv"
    dates_train: str = "data/hq/train100/split_train_datetimes.csv"
    x_val: str = "data/hq/split_val_values.csv"
    dates_val: str = "data/hq/split_val_datetimes.csv"
    x_test: str = "data/hq/split_test_values.csv"
    dates_test: str = "data/hq/split_test_datetimes.csv"


class DataLoader(BaseModel):
    num_features: int = 2
    time_covariates: List[str] = ["week_of_year"]
    scale_method: str = "standard"
    order_mode: str = "by_window"
    input_seq_len: int = 52
    batch_size: int = 16
    output_col: List[int] = [0]
    nb_ts: int = 127
    ts_to_use: List[int] = []


class Data(BaseModel):
    paths: DataPaths = Field(default_factory=DataPaths)
    loader: DataLoader = Field(default_factory=DataLoader)


class StandardEmbeddings(BaseModel):
    embedding_size: Optional[int] = None
    embedding_initializer: str = "normal"
    embedding_init_file: Optional[str] = None


class MappedEmbeddings(BaseModel):
    embedding_map_dir: Optional[str] = None
    embedding_map_sizes: Dict[str, int] = {
        "dam_id": 3,
        "dam_type_id": 3,
        "sensor_type_id": 3,
        "direction_id": 3,
        "sensor_id": 3,
    }
    embedding_map_initializer: Dict[str, str] = {
        "dam_id": "normal",
        "dam_type_id": "normal",
        "sensor_type_id": "normal",
        "direction_id": "normal",
        "sensor_id": "normal",
    }
    embedding_map_labels: Dict[str, List[str]] = {
        "dam_id": ["DRU", "GOU", "LGA", "LTU", "MAT", "M5"],
        "dam_type_id": ["Run-of-River", "Reservoir"],
        "sensor_type_id": ["PIZ", "EXT", "PEN"],
        "direction_id": ["NA", "X", "Y", "Z"],
        "sensor_id": [],  # Will be populated dynamically if needed, or set in config
    }


class Embeddings(BaseModel):
    standard: StandardEmbeddings = Field(default_factory=StandardEmbeddings)
    mapped: MappedEmbeddings = Field(default_factory=MappedEmbeddings)


class Initialization(BaseModel):
    from_file: Optional[str] = None
    variance_inject: float = 0.0
    variance_threshold: float = 1.0
    variance_action: str = "add"


class Model(BaseModel):
    Sigma_v_bounds: tuple = (None, None)
    decaying_factor: float = 0.99
    device: str = "cuda"
    hidden_sizes: List[int] = [40, 40]
    initialization: Initialization = Field(default_factory=Initialization)


class Forecasting(BaseModel):
    recursive_val: bool = True
    recursive_test: bool = True
    rolling_window: bool = False
    rolling_window_size: int = 52


class Training(BaseModel):
    num_epochs: int = 100
    early_stopping_criteria: str = "loglik"
    patience: int = 10
    min_delta: float = 1e-4
    warmup_epochs: int = 0
    shuffle: bool = True
    use_look_back_predictions: bool = True


class Evaluation(BaseModel):
    eval_plots: bool = False
    eval_metrics: bool = True
    seansonal_period: int = 52
    embed_plots: bool = False


class Config(BaseModel):
    seed: Optional[int] = None
    data: Data = Field(default_factory=Data)
    embeddings: Optional[Embeddings] = None
    model: Optional[Model] = None
    training: Optional[Training] = None
    forecasting: Forecasting = Field(default_factory=Forecasting)
    evaluation: Optional[Evaluation] = None

    @property
    def ts_to_use(self) -> List[int]:
        """Returns the list of time series indices to use. If empty, returns all."""
        if self.data.loader.ts_to_use:
            return self.data.loader.ts_to_use
        return list(range(self.data.loader.nb_ts))

    @property
    def x_file(self) -> List[str]:
        """Dynamically creates the list of x files."""
        return [
            self.data.paths.x_train,
            self.data.paths.x_val,
            self.data.paths.x_test,
        ]

    @property
    def date_file(self) -> List[str]:
        """Dynamically creates the list of date files."""
        return [
            self.data.paths.dates_train,
            self.data.paths.dates_val,
            self.data.paths.dates_test,
        ]

    @property
    def use_AGVI(self) -> bool:
        """Determines whether to use AGVI based on Sigma_v_bounds."""
        return (
            self.model.Sigma_v_bounds[0] is None
            and self.model.Sigma_v_bounds[1] is None
        )

    @property
    def use_mapped_embeddings(self) -> bool:
        """True if mapped embeddings are configured."""
        if self.embeddings is None:
            return False
        return self.embeddings.mapped.embedding_map_dir is not None

    @property
    def use_standard_embeddings(self) -> bool:
        """True if standard (one-per-series) embeddings are configured."""
        if self.embeddings is None:
            return False
        return (
            not self.use_mapped_embeddings
            and self.embeddings.standard.embedding_size is not None
            and self.embeddings.standard.embedding_size > 0
        )

    @property
    def total_embedding_size(self) -> int:
        """Calculates the total embedding dimension based on configuration."""
        if self.embeddings is None:
            return 0
        if self.use_mapped_embeddings:
            return sum(self.embeddings.mapped.embedding_map_sizes.values())
        if self.use_standard_embeddings:
            return self.embeddings.standard.embedding_size
        return 0  # No embeddings

    @property
    def input_size(self) -> int:
        """Calculates the total input size for the model."""
        base_size = self.data.loader.num_features + self.data.loader.input_seq_len - 1
        return base_size + self.total_embedding_size

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Loads configuration from a YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str):
        """Saves configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def display(self):
        """Displays the configuration."""
        print("\nConfiguration:")
        print(yaml.dump(self.model_dump(), default_flow_style=False))
        print("\n")

    def wandb_dict(self) -> Dict[str, Any]:
        """Converts the configuration to a dictionary for W&B."""
        return self.model_dump()
