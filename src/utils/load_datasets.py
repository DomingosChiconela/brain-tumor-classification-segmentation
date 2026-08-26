import os
import numpy as np
from src.utils.paths import  build_path

def load_classification_dataset(data_path=None):
    if data_path is None:
       
        data_path = build_path( 'dataset','classification','brain_tumor_dataset.npz')
    
    ds = np.load(data_path, allow_pickle=True)
    return (
        ds["X_train"],
        ds["y_train"],
        ds["X_test"],
        ds["y_test"],
        ds["class_names"]
    )