import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import random


class K_Means:
	def __init__(self, k=2, max_iterations=300):
		self.k = k
		self.max_iterations = max_iterations

	def fit(self, data_set):
		self.centroids = {}
		for i in range(self.k):
			self.centroids[i] = data_set[i]

		for i in range(self.max_iterations):
			self.classifications = {}
			#classifications[centroid] = data_set[cordinate_x, cordinate_y]
			#classifications of data_set(x,y) is some centroid
			for j in range(self.k):
				self.classifications[j] = []
			for featureset in X:
				distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
			print(distances)
			distances = []
			for cordinates in data_set:
				for centroid in self.centroids:
					distances = np.linalg.norm(cordinates-self.centroids[centroid])
					print(distances)
			# 	classification = distances.index(min(distances))
			# 	self.classifications[classification].append(cordinates)
			
			# print(self.classifications)
			# 		# classification = distances.index(min(distances))
					# self.classifications[classification].append(cordinates)
			# prev_centroid = dict(self.centroids)
			break
			# for classification in self.classifications:
			# 	self.centroids[classification] = np.average(self.classifications[classification], axis=0)
			# optimized = True

			# for c in self.centroids:
			# 	orginal_centroid = prev_centroid[c]
			# 	current_centroid = self.centroids[c]

			# 	if sum((current_centroid- orginal_centroid)/orginal_centroid*100.0) > self.tolerance:
			# 		optimized = False
			# if optimized:
			# 	break

	def predict(self, data_set):
		distances = []
		for centroid in self.centroids:
			distances.append(np.linalg.norm(data_set-self.centroids[centroid]))
		classification = distances.index(min(distances))
		return classification
# data_set = pd.read_excel('titanic.xls')
# # K_Means(data_set)
# data_set.drop(['name'], 1, inplace=True)
# data_set.fillna(0, inplace=True)
# columns = data_set.columns.values
# for column in columns:
#     if data_set[column].dtype != np.int64 and data_set[column].dtype != np.float64:
#         data_set[column] = data_set[column].astype('category')
#         data_set[column] = data_set[column].cat.codes
# print(data_set.head())

# # X = data_set.drop(['survived'], 1, inplace=True)
# # X = np.array(X)
# # X= X.astype(float)

# X = np.array(data_set.drop(['survived'], 1).astype(float))
# X = preprocessing.scale(X)


X = np.array([[1, 2],
			 [1.5, 1.8],
			 [3,4]
			 ])

colors = ['r','g','b','c','k','o','y']
# for cordinates in X:
#     plt.scatter(cordinates[0], cordinates[1], c=colors[random.randint(1,3)])
# plt.show()

km = K_Means()
km.fit(X)