import pandas as pd
from sklearn.preprocessing import Imputer
df = pd.read_csv('Data.csv')
df.head()
imputer = Imputer(missing_values='NaN', strategy='mean', axis = 0)
df.iloc[:, 1:3] = imputer.fit_transform(df.iloc[:, 1:3])
df.head()

# Label Encoder will replace every categorical variable with number. Useful for replacing yes by 1, no by 0.
# One Hot Encoder will create a separate column for every variable and give a value of 1 where the variable is present
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

lable_encoder = LabelEncoder()
temp = df.copy()
temp.iloc[:, 0] = lable_encoder.fit_transform(df.iloc[:, 0])
temp.head()
# you can pass an array of indices of categorical features
# one_hot_encoder = OneHotEncoder(categorical_features=[0])
# temp = df.copy()
# temp.iloc[:, 0] = one_hot_encoder.fit_transform(df.iloc[:, 0])

# you can achieve the same thing using get_dummies
pd.get_dummies(df.iloc[:, :-1])

from sklearn.datasets import load_iris

iris_dataset = load_iris()
X = iris_dataset.data
y = iris_dataset.target
feature_names = iris_dataset.feature_names


X[:, 1]

from sklearn.preprocessing import Binarizer
X[:, 1:2] = Binarizer(threshold=X[:, 1].mean()).fit_transform(X[:, 1].reshape(-1, 1))
X[:, 1]

