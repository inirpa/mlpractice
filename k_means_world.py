import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_excel('worldcities.xlsx')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for index in range(len(df)):
	ax.scatter(df['lat'][index], df['lng'][index], df['population'][index], marker='o')
plt.show()
