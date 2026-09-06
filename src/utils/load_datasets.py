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
    
    
    
    
def load_segmentation_dataset():
    npz_path = build_path("dataset", "segmentation", "brain_tumor_segmentation_dataset.npz")
    data = np.load(npz_path, allow_pickle=True)
    return data["X_train"], data["Y_train"], data["X_test"], data["Y_test"], data["class_names"]