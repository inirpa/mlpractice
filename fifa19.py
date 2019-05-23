import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/fifa19.csv')
print(df.head())
ages = df['Age'].unique()
amount = {}

df['release_clause'] = df['Release Clause'].astype(str).map(lambda x: x.lstrip('€').rstrip('M'))
plt.scatter(df['Age'], df['release_clause'])
plt.show()