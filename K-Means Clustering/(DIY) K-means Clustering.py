import numpy as np
import matplotlib as plt
import pandas as pd


##Loading and Pre-Processing

players = pd.read_csv("players_22.csv")
features = ["overall", "potential", "value_eur","wage_eur", "age"]
players = players.dropna(subset=features) #Removes players with missing values
data = players[features].copy()

data = (data-data.min()) / (data.max()-data.min()) *9 +1 #Has to be a scale without zeros and negatives i.e. 1-10, can prob improve this by using shifted z-score



#Initialize Centroids
def random_centroids(data,k):
    centroids = []
    for i in range(k):
        centroid = data.apply(lambda x: float(x.sample())) #random point with random value from each column
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)


#Grouping points in appropriate Cluster
def distance(data, centroids):
    clusters = {0:[], 1:[], 2:[]}
    for i in range(len(data)):  #cycle through players
        distances={}
        for j in range(len(centroids)): #Creates dictionary with distance to each cluster point
            d= np.array(data.iloc[i]) - np.array(centroids[j])
            distances.update({j:np.linalg.norm(d)})
   
        clusters[min(distances, key = distances.get)].append(i) #Selects closest cluster point
    
    print(clusters)

centroids = random_centroids(data,3)
centroids = np.array(centroids).T

distance(data,centroids)
