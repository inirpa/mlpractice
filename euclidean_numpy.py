import numpy as np

def euclidean_distance_numpy(cordinate_first, cordinate_second):
    total_points = len(cordinate_first)
    if(total_points != 2):
        ed = "Euclidean distance needs 2 points for calculation"
        return ed
    
    # ed = np.linalg.norm(np.array(point1) -  np.array(point2))
    ed = np.linalg.norm(np.array(cordinate_first) - np.array(cordinate_second))
    return ed