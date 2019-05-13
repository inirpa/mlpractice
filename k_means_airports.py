import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_excel('districts_nepal.xlsx')
print(df.head())
for i in range(77):
	df['Area (km²)'] = df['Area (km²)'].str.replace(',','')
	df['Population (2011)'] = df['Population (2011)'].str.replace(',','')
	plt.scatter(float(df['Area (km²)'][i]), float(df['Population (2011)'][i]), color='k')
	plt.text(float(df['Area (km²)'][i]), float(df['Population (2011)'][i]), df['Headquarters'][i])
plt.show()
print(df.head())