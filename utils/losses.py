import torch as t
import torch.nn as nn
import numpy as np

def divide_no_nan(a, b):
    """
    Element-wise division of a by b, replacing NaN and Inf results with 0.
    :param a: Numerator tensor.
    :param b: Denominator tensor.
    :return: Result tensor with NaNs and Infs replaced by 0.
    """
    result = a / b
    result = t.where(t.isnan(result) | t.isinf(result), t.tensor(0.0, device=result.device), result)
    return result

class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float32:
        """
        Computes the Mean Absolute Percentage Error (MAPE) loss.
        :param forecast: Forecast values. Shape: (batch, time)
        :param target: Target values. Shape: (batch, time)
        :param mask: Binary mask tensor. Shape: (batch, time)
        :return: MAPE loss value.
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs(forecast - target) * weights)

class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float32:
        """
        Computes the Symmetric Mean Absolute Percentage Error (sMAPE) loss.
        :param forecast: Forecast values. Shape: (batch, time)
        :param target: Target values. Shape: (batch, time)
        :param mask: Binary mask tensor. Shape: (batch, time)
        :return: sMAPE loss value.
        """
        numerator = t.abs(forecast - target)
        denominator = t.abs(forecast) + t.abs(target)
        smape = 200 * divide_no_nan(numerator, denominator) * mask
        return t.mean(smape)

class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float32:
        """
        Computes the Mean Absolute Scaled Error (MASE) loss.
        :param insample: Insample values (historical data). Shape: (batch, time_i)
        :param freq: Seasonal frequency of the data.
        :param forecast: Forecast values. Shape: (batch, time_o)
        :param target: Target values. Shape: (batch, time_o)
        :param mask: Binary mask tensor. Shape: (batch, time_o)
        :return: MASE loss value.
        """
        # Mean absolute error of the insample over seasonal lag `freq`
        mase_denominator = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)

        # MASE is scaled by the denominator calculated above
        masked_mase_inv = divide_no_nan(mask, mase_denominator[:, None])
        return t.mean(t.abs(forecast - target) * masked_mase_inv)
