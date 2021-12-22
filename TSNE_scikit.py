import numpy as np
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#palette = sns.color_palette("bright", 10)

X, y = load_digits(return_X_y=True)
MACHINE_EPSILON = np.finfo(np.double).eps
n_components = 2
perplexity = 30
tsne = TSNE()
X_embedded = tsne.fit_transform(X)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c="black")
plt.title(f't-distributed Stochastic Neighbor Embedding')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
