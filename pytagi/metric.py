import numpy as np
from scipy.stats import norm

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
    return np.nanmean(log_lik)


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


def Nrmse(prediction: np.ndarray, observation: np.ndarray) -> float:
    """Calculates the Normalized Root Mean Squared Error (NRMSE).

    NRMSE is the RMSE divided by the range of the observed values.

    :param prediction: The predicted values.
    :type prediction: np.ndarray
    :param observation: The actual (observed) values.
    :type observation: np.ndarray
    :return: The normalized root mean squared error.
    :rtype: float
    """
    mse_val = mse(prediction, observation)
    rmse_val = np.sqrt(mse_val)
    obs_range = np.nanmax(observation) - np.nanmin(observation)
    if obs_range == 0:
        return np.nan
    return rmse_val / obs_range


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


def mase(y, y_pred, y_train, seasonality: int = 1):
    """Computes the Mean Absolute Scaled Error (MASE) for a single time series.

    This metric scales the mean absolute error of the forecast by the in-sample
    naive forecast error so that scores are comparable across series.

    :param y: Observed values of shape (n_samples,).
    :type y: np.ndarray
    :param y_pred: Forecasted values matching ``y``.
    :type y_pred: np.ndarray
    :param y_train: Training data used to compute the naive seasonal differences.
    :type y_train: np.ndarray
    :param seasonality: Seasonal period; use 1 for non-seasonal data.
    :type seasonality: int
    :return: The MASE value; returns ``np.nan`` when scaling is undefined.
    :rtype: float
    """
    y_train = y_train[~np.isnan(y_train)]
    if len(y_train) <= seasonality:
        return np.nan
    scale = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    if scale == 0:
        return np.nan
    mae = np.mean(np.abs(y - y_pred))
    return mae / scale


def Np50(y, ypred):
    """Computes the normalized median absolute error.

    :param y: Observed target values.
    :type y: np.ndarray
    :param ypred: Predicted 50th percentile (median) values.
    :type ypred: np.ndarray
    :return: Median absolute error normalized by the sum of the absolute targets.
    :rtype: float
    """
    y = np.asarray(y)
    denom = np.nansum(np.abs(y)) + 1e-8
    return p50(y, ypred) / denom


def Np90(y, ypred, spred):
    """Computes the normalized tilted loss for the 90th percentile.

    :param y: Observed target values.
    :type y: np.ndarray
    :param ypred: Predicted mean values.
    :type ypred: np.ndarray
    :param spred: Predicted standard deviations.
    :type spred: np.ndarray
    :return: 90th percentile tilted loss normalized by the sum of absolute targets.
    :rtype: float
    """
    y = np.asarray(y)
    ypred = np.asarray(ypred)
    spred = np.asarray(spred)
    denom = np.nansum(np.abs(y)) + 1e-8
    return p90(y, ypred, spred) / denom


def p50(y: np.ndarray, ypred: np.ndarray) -> float:
    """Computes the (non-normalized) median absolute error.

    :param y: Observed target values.
    :type y: np.ndarray
    :param ypred: Predicted mean values.
    :type ypred: np.ndarray
    :return: Sum of absolute deviations between targets and predictions.
    :rtype: float
    """
    y = np.asarray(y)
    ypred = np.asarray(ypred)
    e = y - ypred
    return np.nansum(np.abs(e))


def p90(y: np.ndarray, ypred: np.ndarray, spred: np.ndarray) -> float:
    """Computes the (non-normalized) tilted loss for the 90th percentile.

    :param y: Observed target values.
    :type y: np.ndarray
    :param ypred: Predicted mean values.
    :type ypred: np.ndarray
    :param spred: Predicted standard deviations.
    :type spred: np.ndarray
    :return: 90th percentile tilted loss aggregated over targets.
    :rtype: float
    """
    y = np.asarray(y)
    ypred = np.asarray(ypred)
    spred = np.asarray(spred)
    ypred_90q = ypred + 1.282 * spred  # 1.282 is the z-score for the 90th percentile
    Iq = y > ypred_90q
    Iq_ = ~Iq
    e = y - ypred_90q
    return np.nansum(2 * e * (0.9 * Iq - 0.1 * Iq_))


def mae(prediction: np.ndarray, observation: np.ndarray) -> float:
    """Calculates the Mean Absolute Error (MAE).

    MAE measures the average magnitude of the errors in a set of predictions,
    without considering their direction.

    :param prediction: The predicted values.
    :type prediction: np.ndarray
    :param observation: The actual (observed) values.
    :type observation: np.ndarray
    :return: The mean absolute error.
    :rtype: float
    """
    return np.nanmean(np.abs(prediction - observation))


def CRPS(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Computes the CRPS for Gaussian predictive distributions.

    :param y: Observed target values of shape ``(n_samples,)``.
    :type y: np.ndarray
    :param mu: Predictive mean values of shape ``(n_samples,)``.
    :type mu: np.ndarray
    :param sigma: Predictive standard deviations of shape ``(n_samples,)``.
    :type sigma: np.ndarray
    :return: The average CRPS across samples.
    :rtype: float
    """
    # Ensure proper shape
    y = y.reshape(-1)
    mu = mu.reshape(-1)
    sigma = sigma.reshape(-1)

    # Standardized difference
    z = (y - mu) / sigma
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return np.nanmean(crps)
