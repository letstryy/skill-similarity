import numpy as np
from scipy.sparse import isspmatrix, dok_matrix, csc_matrix
import sklearn.preprocessing
from utils import MessagePrinter
import pandas as pd

M = []
for r in open("finalv.csv"):
        r = r.strip().split(",")
        M.append(list(map(lambda x: str(x.strip()), r)))

matrix = np.matrix(M)
def sparse_allclose(a, b, rtol=1e-5, atol=1e-8):
    """
    Version of np.allclose for use with sparse matrices
    """
    c = np.abs(a - b) - rtol * np.abs(b)
    # noinspection PyUnresolvedReferences
    return c.max() <= atol


def normalize(matrix):
   
    return sklearn.preprocessing.normalize(matrix, norm="l1", axis=0)


def inflate(matrix, power):

    if isspmatrix(matrix):
        return normalize(matrix.power(power))

    return normalize(np.power(matrix, power))


def expand(matrix, power):

    if isspmatrix(matrix):
        return matrix ** power

    return np.linalg.matrix_power(matrix, power)


def add_self_loops(matrix, loop_value):
    shape = matrix.shape
    assert shape[0] == shape[1], "Error, matrix is not square"

    if isspmatrix(matrix):
        new_matrix = matrix.todok()
    else:
        new_matrix = matrix.copy()

    for i in range(shape[0]):
        new_matrix[i, i] = loop_value

    if isspmatrix(matrix):
        return new_matrix.tocsc()

    return new_matrix


def prune(matrix, threshold):

    if isspmatrix(matrix):
        pruned = dok_matrix(matrix.shape)
        pruned[matrix >= threshold] = matrix[matrix >= threshold]
        pruned = pruned.tocsc()
    else:
        pruned = matrix.copy()
        pruned[pruned < threshold] = 0

    return pruned


def converged(matrix1, matrix2):
 
    if isspmatrix(matrix1) or isspmatrix(matrix2):
        return sparse_allclose(matrix1, matrix2)

    return np.allclose(matrix1, matrix2)


def iterate(matrix, expansion, inflation):
  
    # Expansion
    matrix = expand(matrix, expansion)

    # Inflation
    matrix = inflate(matrix, inflation)

    return matrix


def get_clusters(matrix):

    if not isspmatrix(matrix):
        # cast to sparse so that we don't need to handle different 
        # matrix types
        matrix = csc_matrix(matrix)

    # get the attractors - non-zero elements of the matrix diagonal
    attractors = matrix.diagonal().nonzero()[0]

    # somewhere to put the clusters
    clusters = set()

    # the nodes in the same row as each attractor form a cluster
    for attractor in attractors:
        cluster = tuple(matrix.getrow(attractor).nonzero()[1].tolist())
        clusters.add(cluster)

    return sorted(list(clusters))


def run_mcl(matrix, expansion=2, inflation=2, loop_value=1,
            iterations=100, pruning_threshold=0.001, pruning_frequency=1,
            convergence_check_frequency=1, verbose=False):

    assert expansion > 1, "Invalid expansion parameter"
    assert inflation > 1, "Invalid inflation parameter"
    assert loop_value >= 0, "Invalid loop_value"
    assert iterations > 0, "Invalid number of iterations"
    assert pruning_threshold >= 0, "Invalid pruning_threshold"
    assert pruning_frequency > 0, "Invalid pruning_frequency"
    assert convergence_check_frequency > 0, "Invalid convergence_check_frequency"

    printer = MessagePrinter(verbose)

    printer.print("-" * 50)
    printer.print("MCL Parameters")
    printer.print("Expansion: {}".format(expansion))
    printer.print("Inflation: {}".format(inflation))
    if pruning_threshold > 0:
        printer.print("Pruning threshold: {}, frequency: {} iteration{}".format(
            pruning_threshold, pruning_frequency, "s" if pruning_frequency > 1 else ""))
    else:
        printer.print("No pruning")
    printer.print("Convergence check: {} iteration{}".format(
        convergence_check_frequency, "s" if convergence_check_frequency > 1 else ""))
    printer.print("Maximum iterations: {}".format(iterations))
    printer.print("{} matrix mode".format("Sparse" if isspmatrix(matrix) else "Dense"))
    printer.print("-" * 50)

    # Initialize self-loops
    if loop_value > 0:
        matrix = add_self_loops(matrix, loop_value)

    # Normalize
    matrix = normalize(matrix)

    # iterations
    for i in range(iterations):
        printer.print("Iteration {}".format(i + 1))

        # store current matrix for convergence checking
        last_mat = matrix.copy()

        # perform MCL expansion and inflation
        matrix = iterate(matrix, expansion, inflation)

        # prune
        if pruning_threshold > 0 and i % pruning_frequency == pruning_frequency - 1:
            printer.print("Pruning")
            matrix = prune(matrix, pruning_threshold)

        # Check for convergence
        if i % convergence_check_frequency == convergence_check_frequency - 1:
            printer.print("Checking for convergence")
            if converged(matrix, last_mat):
                printer.print("Converged after {} iteration{}".format(i + 1, "s" if i > 0 else ""))
                break

    printer.print("-" * 50)

    return matrix

x = pd.DataFrame(run_mcl(matrix))
x.to_csv('graph.csv',encoding='utf8')
