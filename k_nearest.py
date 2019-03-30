import warnings
from euclidean_numpy import euclidean_distance_numpy
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

def k_nearest_neighbors(data, predict, k=4):
	if len(data) >= k:
		warnings.warn('K value improper')
	distances = []
	votes = []

	for group in data:
		for cordinates in data[group]:
			ed = euclidean_distance_numpy(cordinates, predict)
			distances.append([ed, group])
	distances = sorted(distances)

	for selected in range(0,k):
		votes.append(distances[selected][1])
	print(votes)

	predicted_group = Counter(votes).most_common(1)[0][0]
	print(predicted_group)
	
	for ds in data:
		for points in data[ds]:
			plt.scatter(points[0], points[1], color=ds)

	plt.scatter(predict[0], predict[1], color=predicted_group)
	plt.text(predict[0], predict[1], 'predicted')
	plt.show()

data_set = {
	'g' : [
		[3,5], [3,7], [5,8], [2.5, 4]
		],
	'r':[
		[1.5,5], [2,6], [3,9], [0,0]
		]
}
predict = [2,3]
k_nearest_neighbors(data_set, predict, k=4)