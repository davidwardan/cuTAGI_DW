from experiments.wandb_helpers import (
    init_run,
    finish_run,
)
import torch

from experiments.time_series_global import train_global_model, eval_global_model, Config

# Create configuration
config = Config()
config.seed = 1
config.batch_size = 4
config.shuffle = True
config.device = "cuda" if torch.cuda.is_available() else "cpu"
config.x_train = "data/toy_example_global/toy_ts_train_values.csv"
config.dates_train = "data/toy_example_global/toy_ts_train_dates.csv"
config.x_val = "data/toy_example_global/toy_ts_val_values.csv"
config.dates_val = "data/toy_example_global/toy_ts_val_dates.csv"
config.x_test = "data/toy_example_global/toy_ts_test_values.csv"
config.dates_test = "data/toy_example_global/toy_ts_test_dates.csv"
config.num_epochs = 50
config.eval_plots = True
config.scale_method = "standard"

config.input_seq_len = 24
config.ts_to_use = list(range(8))

config.eval_plots = False
config.embed_plots = False

if __name__ == "__main__":
    log_wandb = True

    # Convert config object to a dictionary for W&B
    config_dict = config.wandb_dict()

    # experiment name
    experiment_name = "Toy_Global_Run"

    if log_wandb:
        # Initialize W&B run
        run = init_run(
            project="tracking_weights_lstm",
            name=experiment_name,
            config=config_dict,
            reinit=True,
            save_code=True,
        )
    else:
        run = None

    train_global_model(config, experiment_name=experiment_name, wandb_run=run)

    eval_global_model(config, experiment_name=experiment_name, wandb_run=run)

    # Finish the W&B run
    if log_wandb:
        finish_run(run)
