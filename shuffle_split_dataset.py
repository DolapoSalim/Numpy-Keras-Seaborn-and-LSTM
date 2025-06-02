import numpy as np
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def shuffle_and_split_data (data, test_ratio):
  np.random.seed(42)
  shuffled_indices = np.random.permutation(len(data)
  test_set_size = int(len(data) * test_ratio)
  test_indices = shuffled_indices[:test_set_size]
  train_indices = shuffled_indices[test_set_size:]
  return data.iloc[train_indices], data.iloc[test_indices]
