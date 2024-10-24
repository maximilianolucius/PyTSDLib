from typing import List
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    """
    Base class for time features.
    Subclasses should implement the __call__ method to extract features from a DatetimeIndex.
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Encodes the second of the minute as a value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Encodes the minute of the hour as a value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Encodes the hour of the day as a value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Encodes the day of the week as a value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Encodes the day of the month as a value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Encodes the day of the year as a value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Encodes the month of the year as a value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Encodes the week of the year as a value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features based on the given frequency string.

    Parameters
    ----------
    freq_str : str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.

    Returns
    -------
    List[TimeFeature]
        List of appropriate time feature extractors.

    Raises
    ------
    RuntimeError
        If the frequency string is not supported.
    """
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Second: [SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = """
    Unsupported frequency {freq_str}. Supported frequencies are:
        Y   - yearly (alias: A)
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely (alias: min)
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates: pd.DatetimeIndex, freq: str = 'H') -> np.ndarray:
    """
    Generate time-based features for a given DatetimeIndex.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Datetime index to extract features from.
    freq : str, default 'H'
        Frequency string to specify the granularity.

    Returns
    -------
    np.ndarray
        Matrix of time features.
    """
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
