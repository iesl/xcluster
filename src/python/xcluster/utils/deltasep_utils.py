"""
Utilities for creating delta separated data sets.
"""
import numpy as np


def gen_k_centers(k, dim):
    """Create a k cluster data set with required separation.

    For the purposes of validating a proof, generate each cluster center such
    that it is at least 4 * delta away from any other cluster for some value of
    delta > 0.

    Args:
        k - the number of clusters.
        dim - (optional) the dimension of the points.

    Returns:
        A list of 2 cluster centers and a value of delta such that the clusters
        centers are 4 * delta away form each other.
    """
    delta = abs(np.random.normal(0.0, 5.0))
    eps = 0.001
    centers = []
    for i in range(k):
        c = np.random.multivariate_normal(np.zeros(dim), np.identity(dim))
        if len(centers):
            c1 = centers[0]
            x = np.random.multivariate_normal(c1, np.identity(c1.size)) - c1
            direction = x / np.linalg.norm(x)
            centers.append(c1 + 2.0 * i * delta * direction + eps)
        else:
            centers.append(c)
    return centers, delta



def create_dataset(dims, size, num_clusters=20):
    """Create a delta separated data set.

    Generate a set of centers for the clusters and from each center draw size
    number of points that constitute the points in that cluster.  Then return
    a dataset of all points.

    Args:
        dims - (int) the dimention of all data points.
        size - (int) the number of points to generate for each cluster.
        num_clusters - (int) the number of clusters.
    """
    clusters, delta = gen_k_centers(num_clusters, dims)
    return _create_constrained_dataset(clusters, delta, size)


def _create_constrained_dataset(centers, delta, size):
    """Create a delta-separated dataset.

    For each of the centers draw size number of points. No two points may be
    farther than delta away form each other. Thus, to generate each point,
    choosea random direction and random distance from the center (of up to 0.5
    delta).

    Args:
      centers - a list of cluster centers.
      delta - the maximum distance between two points in the same cluster.
      size - the number of points to draw per cluster.

    Returns:
      A list of points that represents the dataset.
    """
    dataset = []
    count = 0
    for i, c in enumerate(centers):
        for j in range(size):
            x = np.random.multivariate_normal(c, np.identity(np.size(c))) - c
            direction = x / np.linalg.norm(x)
            magnitude = np.random.uniform(0.0, 0.5 * delta)
            # magnitude = np.random.uniform(0.0, delta) # NOT DEL-SEPARATED
            dataset.append((c + magnitude * direction, i, count))
            count += 1
    return dataset


def gen_4_normal():
    """Create 4 cluster centers.

    Create gaussians centered at (1,1), (1,-1), (-1,-1) and (-1,1).  Each has
    standard covariance.

    Args:
        None

    Returns:
        A list of the four cluster centers.
    """
    return [mn(mean=np.array([1.0, 1.0]),
               cov=np.array([[1.0, 0.0], [0.0, 1.0]])),
            mn(mean=np.array([1.0, -1.0]),
               cov=np.array([[1.0, 0.0], [0.0, 1.0]])),
            mn(mean=np.array([-1.0, -1.0]),
               cov=np.array([[1.0, 0.0], [0.0, 1.0]])),
            mn(mean=np.array([-1.0, 1.0]),
               cov=np.array([[1.0, 0.0], [0.0, 1.0]]))]


def _4_normal_spread():
  """Create 4 cluster centers.

  Create gaussians centered at (10,10), (10,-10), (-10,-10) and (-10,10).
  Each has standard covariance.

  Args:
    None

  Returns:
    A list of the four cluster centers.
  """
  return [mn(mean=np.array([10.0, 10.0]),
             cov=np.array([[1.0, 0.0], [0.0, 1.0]])),
          mn(mean=np.array([10.0, -10.0]),
             cov=np.array([[1.0, 0.0], [0.0, 1.0]])),
          mn(mean=np.array([-10.0, -10.0]),
             cov=np.array([[1.0, 0.0], [0.0, 1.0]])),
          mn(mean=np.array([-10.0, 10.0]),
             cov=np.array([[1.0, 0.0], [0.0, 1.0]]))]


def _5x5_grid_clusters():
  """Create a 5x5 grid of cluster centers.

  Create 25 cluster centers on the grid I^{[0, 4] x [0,4]}.  Each center is a
  gaussian with standard covariance

  Args:
    None

  Returns:
    A list of cluster centers.
  """
  return [mn(mean=np.array([i, j]), cov=np.array([[1.0, 0.0],
                                                  [0.0, 1.0]]))
          for i in range(5)
          for j in range(5)]


def _5x5_grid_clusters_spread():
  """Create a 5x5 grid of cluster centers.

  Create 25 cluster centers on the grid I^{[0, 4] x [0,4]}.  Each center is a
  gaussian with standard covariance

  Args:
    None

  Returns:
    A list of cluster centers.
  """
  return [mn(mean=np.array([i * 25, j * 25]), cov=np.array([[1.0, 0.0],
                                                            [0.0, 1.0]]))
          for i in range(5)
          for j in range(5)]


def _5x5_grid_clusters_close():
  """Create a 5x5 grid of cluster centers.

  Create 25 cluster centers on the grid I^{[0, 4] x [0,4]}.  Each center is a
  gaussian with standard covariance

  Args:
    None

  Returns:
    A list of cluster centers.
  """
  return [mn(mean=np.array([i * 5, j * 5]), cov=np.array([[1.0, 0.0],
                                                          [0.0, 1.0]]))
          for i in range(5)
          for j in range(5)]


def _2x3_grid_clusters_close():
  """Create a 3x3 grid of cluster centers.

  Create 25 cluster centers on the grid I^{[0, 4] x [0,4]}.  Each center is a
  gaussian with standard covariance

  Args:
    None

  Returns:
    A list of cluster centers.
  """
  return [mn(mean=np.array([i * 5, j * 5]), cov=np.array([[1.0, 0.0],
                                                          [0.0, 1.0]]))
          for i in range(2)
          for j in range(3)]


def _2x3_grid_clusters_spread():
  """Create a 3x3 grid of cluster centers.

  Create 25 cluster centers on the grid I^{[0, 4] x [0,4]}.  Each center is a
  gaussian with standard covariance

  Args:
    None

  Returns:
    A list of cluster centers.
  """
  return [mn(mean=np.array([i * 25, j * 25]), cov=np.array([[1.0, 0.0],
                                                            [0.0, 1.0]]))
          for i in range(2)
          for j in range(3)]


def _10x10_grid_clusters_close():
  """Create a 3x3 grid of cluster centers.

  Create 25 cluster centers on the grid I^{[0, 4] x [0,4]}.  Each center is a
  gaussian with standard covariance

  Args:
    None

  Returns:
    A list of cluster centers.
  """
  return [mn(mean=np.array([i * 5, j * 5]), cov=np.array([[1.0, 0.0],
                                                          [0.0, 1.0]]))
          for i in range(10)
          for j in range(10)]


def _10x10_grid_clusters_spread():
  """Create a 3x3 grid of cluster centers.

  Create 25 cluster centers on the grid I^{[0, 4] x [0,4]}.  Each center is a
  gaussian with standard covariance

  Args:
    None

  Returns:
    A list of cluster centers.
  """
  return [mn(mean=np.array([i * 25, j * 25]), cov=np.array([[1.0, 0.0],
                                                            [0.0, 1.0]]))
          for i in range(10)
          for j in range(10)]


def _random_standard_centers(n=100):
  """Create random cluster centers.

  Create n cluster centers randomly.  Each cluster center is a draw from a
  gaussian distribution centered at (0,0) with standard covariance.

  Args:
    n - optional; the number of centers to draw (default 100).

  Returns:
    A list of cluster centers.
  """
  generator = mn(mean=np.array([0, 0]),
                 cov=np.array([[1.0, 0.0], [0.0, 1.0]]))
  return [mn(mean=pt, cov=np.array([[1.0, 0.0], [0.0, 1.0]]))
          for pt in generator.rvs(size=n)]


def _from_file(filename):
  with open(filename, 'r') as f:
    clustering = []
    for line in f:
      splits = line.split('\t')
      l, vec = int(splits[0]), np.array([float(x) for x in splits[1:]])
      clustering.append((vec, l))
  return clustering
