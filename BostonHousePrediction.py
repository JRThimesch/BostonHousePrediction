import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
boston = pd.read_csv('data/boston_housing.data', header=None, delimiter=' ')
print(boston)