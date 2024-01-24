from visualizer import PredictionViz

from python_examples.data_loader import RegressionDataLoader
from python_examples.model import RegressionMLP
from python_examples.regression import Regression


def main():
    """Training and testing API"""
    # User-input
    num_inputs = 1
    num_outputs = 1
    num_epochs = 50
    x_train_file = "./data/toy_example/x_train_1D.csv"
    y_train_file = "./data/toy_example/y_train_1D.csv"
    x_test_file = "./data/toy_example/x_test_1D.csv"
    y_test_file = "./data/toy_example/y_test_1D.csv"

    # Model
    net_prop = RegressionMLP()

    # Data loader
    reg_data_loader = RegressionDataLoader(
        num_inputs=num_inputs, num_outputs=num_outputs, batch_size=net_prop.batch_size
    )
    data_loader = reg_data_loader.process_data(
        x_train_file=x_train_file,
        y_train_file=y_train_file,
        x_test_file=x_test_file,
        y_test_file=y_test_file,
    )

    # Train and test
    viz = None  # PredictionViz(task_name="regression", data_name="toy1D")
    reg_task = Regression(
        num_epochs=num_epochs, data_loader=data_loader, net_prop=net_prop, viz=viz
    )
    reg_task.train()
    reg_task.predict(std_factor=3)


if __name__ == "__main__":
    main()
