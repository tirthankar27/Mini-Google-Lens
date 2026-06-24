import numpy as np
from sklearn.neighbors import NearestNeighbors

embeddings = np.load("embeddings.npy")

print("loaded")

nn_model = NearestNeighbors(
    n_neighbors=6,
    metric="euclidean"
)

nn_model.fit(embeddings)

print("fitted")

query = embeddings[0].reshape(1, -1)

distances, indices = nn_model.kneighbors(query)

print("worked")
print(indices)