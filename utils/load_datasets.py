import os
import numpy as np

def load_classification_dataset(data_path=None):
    if data_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, 'dataset', 'classification', 'brain_tumor_dataset.npz')
    
    ds = np.load(data_path, allow_pickle=True)
    return (
        ds["X_train"],
        ds["y_train"],
        ds["X_test"],
        ds["y_test"],
        ds["class_names"]
    )