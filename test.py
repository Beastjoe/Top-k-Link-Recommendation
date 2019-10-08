import numpy as np
import utils
from numba import int64
from numba.typed import List
import numba_test

X = np.load("data/wiki/wiki.npy")
X_train, X_validation, X_test, test_list = utils.generate_dataset(X, 0.6)
numba_test.SGD_algorithm(X_train, 1e-4, 1e-4, 16, 1e-4, 100)
