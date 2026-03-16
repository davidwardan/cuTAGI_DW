import numpy as np

from pytagi.nn.data_struct import HRCSoftmax
from pytagi.tagi_utils import Utils


class HRCSoftmaxMetric:
    """Classification error metric for Hierarchical Softmax.

    This class provides methods to compute the error rate and get predicted labels
    for a classification model that uses Hierarchical Softmax.
    """

    def __init__(self, num_classes: int):
        """Initializes the HRCSoftmaxMetric.

        :param num_classes: The total number of classes in the classification problem.
        :type num_classes: int
        """
        self.num_classes = num_classes
        self.utils = Utils()
        self.hrc_softmax: HRCSoftmax = self.utils.get_hierarchical_softmax(
            num_classes=num_classes
        )

    def error_rate(
        self, m_pred: np.ndarray, v_pred: np.ndarray, label: np.ndarray
    ) -> float:
        """Computes the classification error rate.

        This method calculates the proportion of incorrect predictions by comparing
        the predicted labels against the true labels.

        :param m_pred: The mean of the predictions from the model.
        :type m_pred: np.ndarray
        :param v_pred: The variance of the predictions from the model.
        :type v_pred: np.ndarray
        :param label: The ground truth labels.
        :type label: np.ndarray
        :return: The classification error rate, a value between 0 and 1.
        :rtype: float
        """
        batch_size = m_pred.shape[0] // self.hrc_softmax.len
        pred, _ = self.utils.get_labels(
            m_pred, v_pred, self.hrc_softmax, self.num_classes, batch_size
        )
        return classification_error(pred, label)

    def get_predicted_labels(
        self, m_pred: np.ndarray, v_pred: np.ndarray
    ) -> np.ndarray:
        """Gets the predicted class labels from the model's output.

        :param m_pred: The mean of the predictions from the model.
        :type m_pred: np.ndarray
        :param v_pred: The variance of the predictions from the model.
        :type v_pred: np.ndarray
        :return: An array of predicted class labels.
        :rtype: np.ndarray
        """
        batch_size = m_pred.shape[0] // self.hrc_softmax.len
        pred, _ = self.utils.get_labels(
            m_pred, v_pred, self.hrc_softmax, self.num_classes, batch_size
        )
        return pred


def mse(prediction: np.ndarray, observation: np.ndarray) -> float:
    """Calculates the Mean Squared Error (MSE).

    MSE measures the average of the squares of the errors, i.e., the average
    squared difference between the estimated and the observed values.

    :param prediction: The predicted values.
    :type prediction: np.ndarray
    :param observation: The actual (observed) values.
    :type observation: np.ndarray
    :return: The mean squared error.
    :rtype: float
    """
    return np.nanmean((prediction - observation) ** 2)


def log_likelihood(
    prediction: np.ndarray, observation: np.ndarray, std: np.ndarray
) -> float:
    """Computes the log-likelihood.

    This function assumes the likelihood of the observation given the prediction
    is a Gaussian distribution with a given standard deviation.

    :param prediction: The predicted mean of the distribution.
    :type prediction: np.ndarray
    :param observation: The observed data points.
    :type observation: np.ndarray
    :param std: The standard deviation of the distribution.
    :type std: np.ndarray
    :return: The average log-likelihood value.
    :rtype: float
    """
    log_lik = -0.5 * np.log(2 * np.pi * (std**2)) - 0.5 * (
        ((observation - prediction) / std) ** 2
    )
    return np.nansum(log_lik)


def rmse(prediction: np.ndarray, observation: np.ndarray) -> float:
    """Calculates the Root Mean Squared Error (RMSE).

    RMSE is the square root of the mean of the squared errors.

    :param prediction: The predicted values.
    :type prediction: np.ndarray
    :param observation: The actual (observed) values.
    :type observation: np.ndarray
    :return: The root mean squared error.
    :rtype: float
    """
    mse_val = mse(prediction, observation)
    return mse_val**0.5


def mae(prediction: np.ndarray, observation: np.ndarray) -> float:
    """Calculates the Mean Absolute Error (MAE).

    MAE measures the average absolute difference between the predicted
    and observed values.

    :param prediction: The predicted values.
    :type prediction: np.ndarray
    :param observation: The actual (observed) values.
    :type observation: np.ndarray
    :return: The mean absolute error.
    :rtype: float
    """
    return np.nanmean(np.abs(prediction - observation))


def Np50(observation: np.ndarray, prediction_p50: np.ndarray) -> float:
    """Computes normalized pinball loss at the 50th percentile.

    This matches weighted quantile loss for q=0.5:
    ``2 * sum(pinball_q=0.5) / sum(abs(observation))``.

    :param observation: Ground-truth values.
    :type observation: np.ndarray
    :param prediction_p50: Predicted 50th percentile (median) values.
    :type prediction_p50: np.ndarray
    :return: Normalized pinball loss at p50.
    :rtype: float
    """

    q = 0.5
    delta = observation - prediction_p50
    pinball = np.where(delta >= 0, q * delta, (1.0 - q) * (-delta))
    denom = np.nansum(np.abs(observation))
    return (2.0 * np.nansum(pinball)) / (denom + 1e-10)


def Np90(
    observation: np.ndarray,
    prediction_mean: np.ndarray,
    prediction_std: np.ndarray,
) -> float:
    """Computes normalized pinball loss at the 90th percentile.

    The p90 quantile is approximated from Gaussian predictions:
    ``prediction_mean + z_0.9 * prediction_std`` where ``z_0.9`` is
    approximately 1.2815515655.

    :param observation: Ground-truth values.
    :type observation: np.ndarray
    :param prediction_mean: Predicted mean values.
    :type prediction_mean: np.ndarray
    :param prediction_std: Predicted standard deviations.
    :type prediction_std: np.ndarray
    :return: Normalized pinball loss at p90.
    :rtype: float
    """

    q = 0.9
    z_p90 = 1.2815515655446004
    prediction_p90 = prediction_mean + z_p90 * prediction_std
    delta = observation - prediction_p90
    pinball = np.where(delta >= 0, q * delta, (1.0 - q) * (-delta))
    denom = np.nansum(np.abs(observation))
    return (2.0 * np.nansum(pinball)) / (denom + 1e-10)


def classification_error(prediction: np.ndarray, label: np.ndarray) -> float:
    """Computes the classification error rate.

    This function calculates the fraction of predictions that do not match the
    true labels.

    :param prediction: An array of predicted labels.
    :type prediction: np.ndarray
    :param label: An array of true labels.
    :type label: np.ndarray
    :return: The classification error rate (proportion of incorrect predictions).
    :rtype: float
    """
    count = 0
    for pred, lab in zip(prediction.T, label):
        if pred != lab:
            count += 1
    return count / len(prediction)
