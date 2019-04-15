import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing


class K_Means:
    def __init__(self, data_set):
        self.data_set = data_set
        for cordinates in self.data_set:
            plt.scatter(cordinates[0], cordinates[1])
        plt.show()

    def fit(self, data_set):
        pass


data_set = pd.read_excel('titanic.xls')
# K_Means(data_set)
data_set.drop(['name'], 1, inplace=True)
data_set.fillna(0, inplace=True)
columns = data_set.columns.values
for column in columns:
    if data_set[column].dtype != np.int64 and data_set[column].dtype != np.float64:
        data_set[column] = data_set[column].astype('category')
        data_set[column] = data_set[column].cat.codes
print(data_set.head())

# X = data_set.drop(['survived'], 1, inplace=True)
# X = np.array(X)
# X= X.astype(float)

X = np.array(data_set.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
