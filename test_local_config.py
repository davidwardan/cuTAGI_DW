import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

# Mock dependencies
sys.modules["torch"] = MagicMock()
sys.modules["pytagi"] = MagicMock()
sys.modules["pytagi.metric"] = MagicMock()
sys.modules["experiments.wandb_helpers"] = MagicMock()
sys.modules["examples.data_loader"] = MagicMock()
sys.modules["experiments.utils"] = MagicMock()

# Import Config from the shared config file, as it is now used in time_series_locals
from experiments.config import Config


def test_local_config_loading():
    config_path = "experiments/configurations/local_run.yaml"

    print(f"Testing local config loading from {config_path}...")

    try:
        config = Config.from_yaml(config_path)
        print("Config loaded successfully!")

        # Verify some values
        assert config.seed == 1
        assert config.data_loader.batch_size == 1
        assert config.model.decaying_factor == 0.999
        assert config.model.device == "cpu"

        # Verify properties
        assert config.use_AGVI is True
        assert config.nb_ts == 127

        # Verify new fields
        assert config.model.variance_inject == 0.0
        assert config.model.variance_action == "add"

        print("All assertions passed!")
        config.display()

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_local_config_loading()
