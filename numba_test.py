import numpy as np
from numba import jit, njit, jitclass
from numba import int64, int32, float32, boolean, float64
from numba.typed import Dict, List
import utils


@jit(float64(float64[:], float64[:]), nopython=True)
def dot(A, B):
    m = A.shape[0]
    C = 0.0
    for i in range(0, m):
        C += A[i] * B[i]
    return C


@njit(float64(float64))
def sigmoid(z):
    return 1.0 / (1.0 + np.e ** (-z))


@njit(float64(int64, float64[:, :], int64, int64, float64[:, :]))
def approximate_rank_of_link(n, X_hat, row_idx, col_idx, X):
    ret = 0.0
    if X[row_idx][col_idx] == 1:
        for i in range(n):
            if X[row_idx][i] == 0:
                ret += sigmoid(X_hat[row_idx][i] - X_hat[row_idx][col_idx])
    return ret


@njit(float64[:](int64, int64, float64[:, :], float64[:, :], float64[:, :], float64[:, :], int64, int64))
def gradient_U_i(n, r, X_train, X_hat, U, V, row_idx, reg_param):
    ret = np.zeros(r)
    for j in range(n):
        sum = np.zeros(r)
        for s in range(n):
            if X_train[row_idx][j] == 1 and X_train[row_idx][s] == 0:
                sigmoid1 = sigmoid(X_hat[row_idx][s] - X_hat[row_idx][j])
                sigmoid2 = sigmoid(X_hat[row_idx][j] - X_hat[row_idx][s])
                sum += sigmoid1 * sigmoid2 * (V[:, s] - V[:, j])
        ret += 1.0 / (1.0 + approximate_rank_of_link(n, X_hat, row_idx, j, X_train)) * sum
    ret += reg_param * U[:, row_idx]
    return ret


@njit(float64[:](int64, int64, float64[:, :], float64[:, :], float64[:, :], float64[:, :], int64, int64))
def gradient_V_j(n, r, X_train, X_hat, U, V, column_idx, reg_param):
    """
    Calculate the gradient of a given row of V

    :param column_idx: the row index of V
    :param reg_param: regularization parameter of V
    :return: n*1 ndarray of the row gradient
    """
    ret = np.zeros(r)
    for i in range(n):
        sum = np.zeros(r)
        for s in range(n):
            if X_train[i][column_idx] == 1 and X_train[i][s] == 0:
                sigmoid1 = sigmoid(X_hat[i][s] - X_hat[i][column_idx])
                sigmoid2 = sigmoid(X_hat[i][column_idx] - X_hat[i][s])
                sum += sigmoid1 * sigmoid2 * -U[:, i]
        ret += 1.0 / (1.0 + approximate_rank_of_link(n, X_hat, i, column_idx, X_train)) * sum
    ret += reg_param * V[:, column_idx]
    return ret


@njit(float64[:](int64, int64, float64[:, :], float64[:, :], float64[:, :], float64[:, :], int64, int64))
def gradient_V_s(n, r, X_train, X_hat, U, V, s, reg_param):
    """
    Calculate the gradient of Vs

    :param s: row index
    :param reg_param: regularization parameter of V
    :return: n*1 ndarray of the row gradient
    """
    ret = np.zeros(r)
    for i in range(n):
        for j in range(n):
            if X_train[i][j] == 1 and X_train[i][s] == 0:
                sigmoid1 = sigmoid(X_hat[i][s] - X_hat[i][j])
                sigmoid2 = sigmoid(X_hat[i][j] - X_hat[i][s])
                factor = 1.0 / (1.0 + approximate_rank_of_link(n, X_hat, i, j, X_train))
                ret += factor * sigmoid1 * sigmoid2 * U[:, i]
    ret += reg_param * V[:, s]
    return ret


@njit(float64[:, :](float64[:, :], float64[:, :]))
def update_estimation(U, V):
    return np.transpose(U) @ V


@njit(float64(int64, float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64, float64))
def valid_error(n, X_train, X_hat, U, V, u_reg_param, v_reg_param):
    """
    Calculate the validation error.

    :return: value of objective function
    """
    ret = 0.0
    for i in range(n):
        ret += u_reg_param * dot(U[:, i], U[:, i]) / 2.0
        ret += v_reg_param * dot(V[:, i], V[:, i]) / 2.0
        for j in range(n):
            ret += np.log(1 + approximate_rank_of_link(n, X_hat, i, j, X_train))
    return ret


@njit((float64[:, :], float64, float64, int64, float64, int64))
def SGD_algorithm(X_train, u_reg_param, v_reg_param, batch_size, learning_rate, max_iter):
    n = np.shape(X_train)[0]
    r = 30
    U = np.random.rand(r, n)
    V = np.random.rand(r, n)
    X_hat = update_estimation(U, V)
    observed_link_index = []
    unknown_status_index = []
    unknown_start_index = []
    for i in range(n):
        unknown_start_index.append(len(unknown_status_index))
        for j in range(n):
            if X_train[i][j] == 1:
                observed_link_index.append(i * 100000 + j)
            else:
                unknown_status_index.append(i * 100000 + j)

    iteration = 0
    prev_error = 0.0
    curr_error = valid_error(n, X_train, X_hat, U, V, u_reg_param, v_reg_param)
    while True:
        if iteration > max_iter or np.abs(1000 - prev_error) < 1e-5:
            break
        iteration += 1
        print(iteration)
        print(curr_error)
        pos = observed_link_index[np.random.randint(len(observed_link_index))]
        i = pos // 100000
        j = pos % 100000
        # Fix user i, uniformly draw batch_size unknown status link
        if i == n - 1:
            s_list = np.random.choice(np.arange(unknown_start_index[i], len(unknown_status_index)), batch_size,
                                      replace=False)
        else:
            s_list = np.random.choice(np.arange(unknown_start_index[i], unknown_start_index[i + 1]), batch_size,
                                      replace=False)
        for i in range(len(s_list)):
            s_list[i] = unknown_status_index[s_list[i]] % 100000

        for s in s_list:
            if dot(U[:, i], V[:, j]) < dot(U[:, i], V[:, s]):
                grad_i = gradient_U_i(n, r, X_train, X_hat, U, V, i, u_reg_param)
                grad_j = gradient_V_j(n, r, X_train, X_hat, U, V, j, v_reg_param)
                grad_s = gradient_V_s(n, r, X_train, X_hat, U, V, s, v_reg_param)
                U[:, i] -= learning_rate * grad_i
                V[:, j] -= learning_rate * grad_j
                V[:, s] -= learning_rate * grad_s

        X_hat = update_estimation(U, V)
        prev_error = curr_error
        curr_error = valid_error(n, X_train, X_hat, U, V, u_reg_param, v_reg_param)

    return
