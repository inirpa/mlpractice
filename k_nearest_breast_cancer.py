#k nearest neighbor application for breast cancer dataset
# ogrinal code reference : https://www.youtube.com/watch?v=3XPhmnf96s0&index=18&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
#orginal author : sentdex

import warnings
import pandas as pd
import numpy as np
import random
from collections import Counter

def k_nearest_neighbors(data, predict, k=4):
    if len(data) >= k:
        warnings.warn('K value improper')
    distances = []
    votes = []

    for group in data:
        for cordinates in data[group]:
            ed = np.linalg.norm(np.array(cordinates)-np.array(predict))
            distances.append([ed, group])
    distances = sorted(distances)

    for selected in range(0,k):
        votes.append(distances[selected][1])    

    predicted_group = Counter(votes).most_common(1)[0][0]
    return predicted_group

data_set = pd.read_csv("breast-cancer-wisconsin.data.txt")
data_set.drop(['id'], 1, inplace=True)
data_set.replace('?', -99999, inplace=True)

full_data = data_set.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0

for group in test_set:
    for data in test_set[group]:
        predicted_group = k_nearest_neighbors(train_set, data, k=5)
        if group == predicted_group:
            correct += 1

print('accuracy : ', correct/len(test_data))