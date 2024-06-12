import numpy as np

from pytagi import HRCSoftmax, Utils


class HRCSoftmaxMetric:
    """Classifcation error for hierarchical softmax"""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.utils = Utils()
        self.hrc_softmax: HRCSoftmax = self.utils.get_hierarchical_softmax(
            num_classes=num_classes
        )

    def error_rate(
        self, m_pred: np.ndarray, v_pred: np.ndarray, label: np.ndarray
    ) -> float:
        """Compute error rate for classifier"""
        batch_size = m_pred.shape[0] // self.hrc_softmax.len
        pred, _ = self.utils.get_labels(
            m_pred, v_pred, self.hrc_softmax, self.num_classes, batch_size
        )
        return classification_error(pred, label)


def mse(prediction: np.ndarray, observation: np.ndarray) -> float:
    """Mean squared error"""
    return np.nanmean((prediction - observation) ** 2)


def log_likelihood(
    prediction: np.ndarray, observation: np.ndarray, std: np.ndarray
) -> float:
    """Compute the averaged log-likelihood"""

    log_lik = -0.5 * np.log(2 * np.pi * (std**2)) - 0.5 * (
        ((observation - prediction) / std) ** 2
    )

    return np.nanmean(log_lik)


def rmse(prediction: np.ndarray, observation: np.ndarray) -> None:
    """Root mean squared error"""
    mse = mse(prediction, observation)

    return mse**0.5


def classification_error(prediction: np.ndarray, label: np.ndarray) -> None:
    """Compute the classification error"""
    count = 0
    for pred, lab in zip(prediction.T, label):
        if pred != lab:
            count += 1

    return count / len(prediction)

def early_stopping(early_stopping_criteria : str, patience: int, mu_preds, y_val) -> int:
    """Early stopping"""
    if early_stopping_criteria == 'mse':
        mse_val = mse(mu_preds, y_val)

def computeRMSE(y, ypred):
    ypred = ypred.astype(float)
    e = np.reshape((y - ypred) ** 2, (-1, 1))
    e = np.sqrt(np.mean(e))
    return e

#TODO: check if ytrain is correct
def computeMASE(y, ypred, ytrain, seasonality):
    nbts = y.shape[1]
    se = np.full((1, y.shape[1]), np.nan)
    for i in range(nbts):
        ytrain_ = ytrain[:, i]
        # remove nans
        ytrain_ = ytrain_[~np.isnan(ytrain_)]
        if len(ytrain_) > seasonality:
            se[0, i] = np.mean(np.abs(ytrain_[seasonality:] - ytrain_[:-seasonality]))
    se[se == 0] = np.nan
    MASE = np.mean(np.abs(y - ypred) / se)
    MASE = np.nanmean(MASE)
    return MASE

def computeND(y, ypred):
    ND = np.sum(np.abs(ypred - y)) / np.sum(np.abs(y))
    return ND

def compute90QL(y, ypred, Vpred):
    ypred_90q = ypred + 1.282 * np.sqrt(Vpred)
    Iq = y > ypred_90q
    Iq_ = y <= ypred_90q
    e = y - ypred_90q
    p90 = np.sum(2 * e * (0.9 * Iq - (1 - 0.9) * Iq_)) / np.sum(np.abs(y))
    return p90


