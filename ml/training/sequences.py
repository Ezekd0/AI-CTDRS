"""Tabular -> sequence reshaping for the LSTM (numpy only, so it is testable without PyTorch).

Flow/connection records are not temporal sequences. The LSTM therefore reads the ordered feature vector as a
short sequence: features are grouped into ``features_per_step`` consecutive values per time step and the tail is
zero-padded. ``features_per_step=1`` gives one time step per feature. This is a modelling choice recorded in the
training metadata, not a property of the data.
"""
from __future__ import annotations

import numpy as np


def sequence_shape(n_features: int, features_per_step: int) -> tuple[int, int, int]:
    """Return (timesteps, features_per_step, padding)."""
    if features_per_step < 1:
        raise ValueError("features_per_step must be >= 1")
    timesteps = -(-n_features // features_per_step)  # ceil division
    return timesteps, features_per_step, timesteps * features_per_step - n_features


def to_sequences(X: np.ndarray, features_per_step: int) -> np.ndarray:
    """(n, n_features) -> (n, timesteps, features_per_step), float32, zero-padded at the end."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("expected a 2-D feature matrix")
    timesteps, step, pad = sequence_shape(X.shape[1], features_per_step)
    if pad:
        X = np.concatenate([X, np.zeros((X.shape[0], pad), dtype=np.float32)], axis=1)
    return X.reshape(X.shape[0], timesteps, step)
