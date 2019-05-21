#mean shift clustering example 
# code sampled from 
# https://pythonprogramming.net/mean-shift-titanic-dataset-machine-learning-tutorial/
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_excel('worldcities.xlsx')
# df.dropna(subset = ['population'] ,inplace=True) #drop rows where population is 0
df = df[['lng','lat']]
X = np.array(df)

ms = MeanShift()
print("Fit started..")
ms.fit(X)
print("Fit done..")
cluster_centers = ms.cluster_centers_
# print(cluster_centers)
print("Prediction started..")
prediction = ms.predict(X)
print("Prediction done..")

print("Plot started..")
colors = ['g','b','c','k','y','m']
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(prediction)):
	print("Plotting {} of {} ".format(i, len(prediction)))
	ax.scatter(X[i,0], X[i,1], marker='o', c=colors[prediction[i]])
ax.scatter(cluster_centers[:,0], cluster_centers[:,1], marker='x', color='red', s=300, linewidth=5, zorder=10)
plt.show()