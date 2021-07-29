import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

df = pd.read_csv('Data.csv').dropna()
print(df)
X = df[["Age", "Salary"]].values.astype(np.float64)

standard_scaler = StandardScaler()
normalizer = Normalizer()
min_max_scaler = MinMaxScaler()

print("Standardization")
print(standard_scaler.fit_transform(X))

print("Normalizing")
print(normalizer.fit_transform(X))

print("MinMax Scaling")
print(min_max_scaler.fit_transform(X))
