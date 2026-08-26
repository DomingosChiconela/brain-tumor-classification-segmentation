

import numpy as np
from sklearn.model_selection import train_test_split


def normalize_images(X: np.ndarray) -> np.ndarray:
  
    return X.astype("float32") / 255.0


def split_train_val(X: np.ndarray, y: np.ndarray, val_split: float = 0.15, seed: int = 42):
    """

    Returns
    -------
    X_train, X_val, y_train, y_val
    """
    return train_test_split(
        X, y,
        test_size=val_split,
        random_state=seed,
        stratify=y,
    )