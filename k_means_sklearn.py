import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_excel('worldcities.xlsx')
df.drop(['city','city_ascii','country','iso2','iso3','admin_name','capital','id'], 1, inplace=True)
print(len(df)) #Print total rows
print(df.isnull().sum(axis = 0)) #print out total empty values in each column
df.dropna(subset = ['population'] ,inplace=True) #drop rows where population is 0
df = df[['lng','lat']]
X = np.array(df)

print(X)
kmeans = KMeans(n_clusters=6, max_iter=10, algorithm = 'auto')
fitted = kmeans.fit(X)
prediction = kmeans.predict(X)
colors = ['r','g','b','c','k','y','m']
for i in range(len(prediction)):
	# print(i)
	plt.scatter(X[i][0], X[i][1], c=colors[prediction[i]])

plt.show()