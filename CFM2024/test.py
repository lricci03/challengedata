import pandas as pd
import numpy as np

n_obs = 10000

df = pd.read_csv("X_train_N1UvY30.csv", nrows=100*n_obs)

print(df[:10])