import numpy as np

def RSE(pred, true):
    """
    Relative Squared Error (RSE).
    :param pred: Predicted values.
    :param true: True values.
    :return: RSE score.
    """
    numerator = np.sum((true - pred) ** 2)
    denominator = np.sum((true - true.mean()) ** 2)
    return np.sqrt(numerator) / np.sqrt(denominator)

def CORR(pred, true):
    """
    Correlation coefficient between the predicted and true values.
    :param pred: Predicted values.
    :param true: True values.
    :return: Correlation coefficient.
    """
    u = np.sum((true - true.mean(0)) * (pred - pred.mean(0)), axis=0)
    d = np.sqrt(np.sum((true - true.mean(0)) ** 2, axis=0) * np.sum((pred - pred.mean(0)) ** 2, axis=0))
    return np.mean(u / d)

def MAE(pred, true):
    """
    Mean Absolute Error (MAE).
    :param pred: Predicted values.
    :param true: True values.
    :return: MAE score.
    """
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    """
    Mean Squared Error (MSE).
    :param pred: Predicted values.
    :param true: True values.
    :return: MSE score.
    """
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    """
    Root Mean Squared Error (RMSE).
    :param pred: Predicted values.
    :param true: True values.
    :return: RMSE score.
    """
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    """
    Mean Absolute Percentage Error (MAPE).
    :param pred: Predicted values.
    :param true: True values.
    :return: MAPE score.
    """
    return np.mean(np.abs((true - pred) / true))

def MSPE(pred, true):
    """
    Mean Squared Percentage Error (MSPE).
    :param pred: Predicted values.
    :param true: True values.
    :return: MSPE score.
    """
    return np.mean(np.square((true - pred) / true))

def metric(pred, true):
    """
    Computes several error metrics.
    :param pred: Predicted values.
    :param true: True values.
    :return: Tuple of MAE, MSE, RMSE, MAPE, and MSPE.
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
