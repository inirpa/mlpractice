import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import random


class K_Means:
	def __init__(self, k=2, max_iterations=10, tolerance=0.001):
		self.k = k
		self.max_iterations = max_iterations
		self.tolerance = tolerance

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
			
			for cordinates in data_set:
				distances = []
				for centroid in self.centroids:
					distances.append(np.linalg.norm(cordinates-self.centroids[centroid]))					
				classification = distances.index(min(distances))
				self.classifications[classification].append(cordinates)
			
			prev_centroid = dict(self.centroids)
			# print(self.centroids)
			for classification in self.classifications:
				self.centroids[classification] = np.average(self.classifications[classification], axis=0)				
			optimized = True

			for c in self.centroids:
				orginal_centroid = prev_centroid[c]
				current_centroid = self.centroids[c]

				if sum((current_centroid- orginal_centroid)/orginal_centroid*100.0) > self.tolerance:
					optimized = False
			if optimized:
				# print("Final centroids")
				# print(self.centroids)
				break
		# print(self.classifications)
		# print(self.centroids)

	def predict(self, predict_set):
		predictions = {}
		predictions[0] = []
		predictions[1] = []
		for cordinates in predict_set:
			distances = []
			for centroid in self.centroids:
				distances.append(np.linalg.norm(cordinates-self.centroids[centroid]))
			classification = distances.index(min(distances))
			predictions[classification].append(cordinates)
		return predictions

	def plot(self, predicted_cluster):
		colors = ['r','g','b','c','k','o','y']
		for key, value in self.classifications.items():
			for cordinates in value:
				plt.scatter(cordinates[0], cordinates[1], c=colors[key])
		plt.scatter(self.centroids[0][0], self.centroids[0][1], c=colors[0], marker='s')
		plt.text(self.centroids[0][0], self.centroids[0][1], 'centroid')
		plt.scatter(self.centroids[1][0], self.centroids[1][1], c=colors[1], marker='s')
		plt.text(self.centroids[1][0], self.centroids[1][1], 'centroid')
		for key, value in predicted_cluster.items():
			for cordinates in value:
				plt.scatter(cordinates[0], cordinates[1], c=colors[key], marker='*')
				plt.text(cordinates[0], cordinates[1], 'predicted cluster')
		plt.show()

X = np.array([
			[1, 2],
			[5, 8],
			[2, 2],
			[3, 3],
			[1, 6],
			[2, 4],
			[5, 6],
			[4.5 ,7],
			[3.5, 5],
			[3.5, 4.5],
			[3.5, 6],
			[3.5, 4]
			 ])

km = K_Means()
km.fit(X)
predict_list = np.array([
	[1.5, 1],
	[3, 1],
	[5, 9]
	])
predicted_cluster =km.predict(predict_list)
km.plot(predicted_cluster)