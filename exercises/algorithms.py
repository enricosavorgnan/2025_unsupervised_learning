from argparse import ArgumentError

import numpy as np
import time
import plotly.express as px

from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------- #
# ------------------------ PCA ----------------------- #
# ---------------------------------------------------- #

def my_pca(x, dim=None):
    # compute the covariance matrix
    cov_matrix = np.cov(x, rowvar=False)        # with rowvar=True, it assumes X[features, samples]

    # find eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)

    # sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvals)[::-1]
    sorted_eigenvals = eigenvals[sorted_indices]
    sorted_eigenvecs = eigenvecs[:, sorted_indices]

    if dim is None:
        px.scatter(x=np.arange(1, len(sorted_eigenvals)+1), y=np.log(sorted_eigenvals), title="Eigenvalues", labels={"x": "Index", "y": "Eigenvalue"}, width=800, height=800).show()

        time.sleep(2)
        try:
            dim = int(float(input("Enter the number of dimensions to reduce to: ")))
        except ValueError:
            dim = 2

    # select the top d eigenvectors
    selected_eigenvecs =  sorted_eigenvecs[:, :dim]

    # project into the d dimension
    x_pca = x @ selected_eigenvecs

    return x_pca, dim, selected_eigenvecs



# ---------------------------------------------------- #
# ---------------------- ISOMAP ---------------------- #
# ---------------------------------------------------- #

def run_floyd_warshall(graph, data):
    n = graph.shape[0]
    dist = np.full((n, n), np.inf)

    for i in range(n):
        for j in range(n):
            if i == j:
                    dist[i, j] = 0

            if graph[i, j] == 1:
                dist[i, j] = np.linalg.norm(data.iloc[i].values - data.iloc[j].values)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    return dist



def my_isomap(data, dim=None, k=10):
    nn = NearestNeighbors(n_neighbors=k).fit(data)
    nn_graph = nn.kneighbors_graph(data).toarray()
    print(nn_graph)

    print(1)

    dist = run_floyd_warshall(nn_graph, data)

    print(2)

    D2 = dist ** 2
    n = D2.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * H @ D2 @ H


    print(3)

    eigenvals, eigenvecs = np.linalg.eig(G)
    eigenvals = np.real(eigenvals)

    # sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvals)[::-1]
    sorted_eigenvals = eigenvals[sorted_indices]
    sorted_eigenvecs = eigenvecs[:, sorted_indices]

    if dim is None:
        px.scatter(x=np.arange(1, len(sorted_eigenvals)+1), y=sorted_eigenvals, title="Eigenvalues", labels={"x": "Index", "y": "Eigenvalue"}, width=800, height=800).show()

        time.sleep(2)
        dim = int(float(input("Enter the number of dimensions to reduce to: ")))

    # select the top d eigenvectors
    selected_eigenvecs =  sorted_eigenvecs[:, :dim]

    # project into the d dimension
    data_iso = np.real(selected_eigenvecs * np.sqrt(sorted_eigenvals[:dim]))

    return data_iso, dim, selected_eigenvecs



# ---------------------------------------------------- #
# -------------------- Kernel-PCA -------------------- #
# ---------------------------------------------------- #

def kernel_matrix(x, kernel_type, kernel_param=None):
    x = x.values
    if kernel_type not in ['rbf', 'poly_1', 'poly_2']:
        raise TypeError
    if kernel_type == 'rbf':
        print(1)
        if not kernel_param:
            raise ArgumentError
        kernel = [[np.exp(-.5*( np.linalg.norm(i - j)/ kernel_param )**2 ) for i in x] for j in x]

    if kernel_type == 'poly_1':
        if not kernel_param:
            raise ArgumentError
        kernel = [[(1 + i@j.T)**kernel_param for i in x] for j in x]

    if kernel_type == 'poly_2':
        if not kernel_param:
            raise ArgumentError
        kernel = [[(i@j.T)**kernel_param for i in x] for j in x]
    print(2)
    kernel = np.array(kernel)
    print(3)
    return kernel


def my_kpca(x, dim=None, kernel_type='rbf', kernel_param=None):
    # calculating the kernel matrix
    print(0)
    K = kernel_matrix(x, kernel_type, kernel_param)
    print(4)
    # Double center K
    G = K - np.mean(K, axis=0, keepdims=True) - np.mean(K, axis=1, keepdims=True) - np.mean(K)

    print(5)
    eigenvals, eigenvecs = np.linalg.eig(G)
    eigenvals = np.real(eigenvals)

    sorted_indices = np.argsort(eigenvals)[::-1]
    sorted_eigenvals = eigenvals[sorted_indices]
    sorted_eigenvecs = eigenvecs[:, sorted_indices]

    if dim is None:
        px.scatter(x=np.arange(1, len(sorted_eigenvals)+1), y=sorted_eigenvals, title="Eigenvalues", labels={"x": "Index", "y": "Eigenvalue"}, width=800, height=800).show()

        time.sleep(2)
        dim = int(float(input("Enter the number of dimensions to reduce to: ")))

    print(6)
    # select the top d eigenvectors
    selected_eigenvecs =  sorted_eigenvecs[:, :dim]

    # project into the d dimension
    x_kpca = np.real(selected_eigenvecs * np.sqrt(sorted_eigenvals[:dim]))

    return x_kpca, dim, selected_eigenvecs


# ---------------------------------------------------- #
# ---------------------. Two-NN ---------------------- #
# ---------------------------------------------------- #

def my_two_nn(data):
    n = data.shape[0]

    neigh = NearestNeighbors(n_neighbors=3).fit(data)
    distances, indices = neigh.kneighbors(data)
    print(indices.shape)
    print(indices[0])
    mu = distances[:, 2] / (distances[:, 1] + 1e-8)
    d = n / np.sum(np.log(mu))

    return d


# ---------------------------------------------------- #
# ----------.Histogram Density Estimation ------------ #
# ---------------------------------------------------- #

def my_histogram_density_estimation(data, h=None):
    x0 = np.min(data)
    x1 = np.max(data)
    n = data.shape[0]
    if h is None:
        iqr = np.quantile(data, 0.75) - np.quantile(data, 0.25)
        h = 2 * iqr / np.cbrt(n)

    n_bins = int(np.ceil((x1 - x0) / h))

    print(n, h, x0, x1, n_bins)
    counts = np.array([0]*n_bins)

    for i in range(n_bins):
        for data_point in data:
            if x0+i*h <= data_point < x0+ (i + 1)*h:
                counts[i] += 1
    print(counts)
    probs = counts/n
    return probs, x0, x1, h


# ---------------------------------------------------- #
# -----------..Kernel Density Estimation ------------- #
# ---------------------------------------------------- #

def my_kernel_density_estimation(data, h=None, grid_points=100):
    n = data.shape[0]
    data = np.sort(data)
    if h is None:
        iqr = np.quantile(data, 0.75) - np.quantile(data, 0.25)
        sd = np.sqrt(np.var(data))
        h = 0.9 * np.minimum(sd, iqr/1.34)* n**(-.2)

    print(n, h)
    # Create grid for evaluation
    x_grid = np.linspace(np.min(data)-.5*np.sqrt(np.var(data)), np.max(data)+.5*np.sqrt(np.var(data)), grid_points)

    probs = np.array([0.]*len(x_grid))
    for i, x in enumerate(x_grid):
        probs[i] = np.sum(np.exp(-.5*((x - data)/h)**2)) / n / h / np.sqrt(2 * np.pi)

    return probs, h, x_grid
