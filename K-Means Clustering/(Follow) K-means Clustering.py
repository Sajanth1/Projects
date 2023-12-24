import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from IPython.display import clear_output


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
def get_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data- x)**2).sum(axis=1))) #Distance from each player to each cluster
    return distances.idxmin(axis=1)


#Checkpoint
# centroids = random_centroids(data, 3)
# labels = get_labels(data, centroids)
# labels



#Update Centroids based on Cluster
def new_centroids(data, labels, k):
    return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T #Geometric mean of each player in each cluster


#Visualization

def plot_clusters(data, labels, centroids, iteration):
    #Placing on 2D graph
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    
    #Plotting graph
    clear_output(wait=True)
    plt.title(f'Iteration {iteration}')
    plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=labels)
    plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1])
    plt.show()



##From random seed, stabilizes to a set of clusters
    
def train(data, k):

    max_iterations = 100

    centroids = random_centroids(data, k)
    old_centroids = pd.DataFrame()

    iteration=1
    while iteration < max_iterations and not centroids.equals(old_centroids):
        old_centroids = centroids
        
        labels = get_labels(data, centroids)
        centroids = new_centroids(data, labels, k)
        plot_clusters(data, labels, centroids, iteration)
        iteration += 1

    vartot = data.groupby(labels).apply(lambda x: np.var(x)).mean(axis=1).mean(axis=0)

    return centroids, vartot


##Tries multiple random seeds, to get best one

def best_cluster(data, k, epochs):
    scores= {}
    for i in range(epochs):
        centroids, vartot= train(data, k)
        scores.update({vartot: centroids})

    return scores[min(scores)], min(scores)


#Graphing elbow plot (variance w.r.t k)

k=1
kmax = 8
epochs = 5

history=[]
while k <= kmax:
    clusters, variance = best_cluster(data, k, epochs)
    history.append(variance)
    k+= 1


x=[i for i in range(1,kmax+1)]
y= history

plt.plot(x,y)
plt.show()



#Checking out patterns: players[labels==2][["short_name"]+features]