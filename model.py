import numpy as np
import utils


class Recommend_Model:
    """ Model that running our recommendation algorithm"""
    n = 0  # number of nodes
    consider_explicit = False  # take explicit feature into consideration
    r = 30  # rank of approximate matrix
    X_train = np.array([])  # n*n adjacency matrix for training
    X_test = np.array([])  # n*n adjacency matrix for test
    X_validation = np.array([])  # n*n adjacency matrix for validation
    X_hat = np.array([])  # n*n estimated score matrix
    U = np.array([])  # r*n low rank approximation
    V = np.array([])  # r*n low rank approximation
    G = np.array([])  # n*n*3 explicit feature matrix
    theta = np.array([])  # 3*1 explicit feature param
    test_list = []  # list of index of test nodes

    def __init__(self, X, train_fraction=0.1, consider_explicit=False):
        self.X_train, self.X_validation, self.X_test, self.test_list = utils.generate_dataset(X, train_fraction)
        self.n = np.shape(X)[0]
        self.consider_explicit = consider_explicit
        self.U = np.random.rand((self.r, self.n))
        self.V = np.random.rand((self.r, self.n))
        self.update_estimation()

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.e ** (-z))

    def approximate_rank_of_link(self, row_idx, col_idx, X):
        """
        Calculate approximate rank R_hat given link

        :param row_idx: row index of the entry
        :param col_idx: column index of the entry
        :param X: adjacency matrix to be calculated
        :return: value of approximate rank of the link
        """
        ret = 0.0
        if X[row_idx][col_idx] == 1:
            for i in range(self.n):
                if X[row_idx][col_idx] == 1:
                    ret += Recommend_Model.sigmoid(self.X_hat[row_idx][i] - X[row_idx][col_idx])
        return ret

    def gradient_U_i(self, row_idx, reg_param):
        """
        Calculate the gradient of a given row of U

        :param row_idx: the row index of U
        :param reg_param: regularization parameter of U
        :return: n*1 ndarray of the row gradient
        """
        ret = np.zeros(self.n)
        for j in range(self.n):
            sum = np.zeros(self.n)
            for s in range(self.n):
                if self.X_train[row_idx][j] == 1 and self.X_train[row_idx][s] == 0:
                    sigmoid1 = Recommend_Model.sigmoid(self.X_hat[row_idx][s] - self.X_hat[row_idx][j])
                    sigmoid2 = Recommend_Model.sigmoid(self.X_hat[row_idx][j] - self.X_hat[row_idx][s])
                    sum += sigmoid1 * sigmoid2 * (self.V[s] - self.V[j])
            ret += 1.0 / (1.0 + self.approximate_rank_of_link(row_idx, j, self.X_train)) * sum
        ret += reg_param * self.U[row_idx]
        return ret

    def gradient_V_j(self, column_idx, reg_param):
        """
        Calculate the gradient of a given row of V

        :param column_idx: the row index of V
        :param reg_param: regularization parameter of V
        :return: n*1 ndarray of the row gradient
        """
        ret = np.zeros(self.n)
        for i in range(self.n):
            sum = np.zeros(self.n)
            for s in range(self.n):
                if self.X_train[i][column_idx] == 1 and self.X_train[i][s] == 0:
                    sigmoid1 = Recommend_Model.sigmoid(self.X_hat[i][s] - self.X_hat[i][column_idx])
                    sigmoid2 = Recommend_Model.sigmoid(self.X_hat[i][column_idx] - self.X_hat[i][s])
                    sum += sigmoid1 * sigmoid2 * -self.U[i]
            ret += 1.0 / (1.0 + self.approximate_rank_of_link(i, column_idx, self.X_train)) * sum
        ret += reg_param * self.V[column_idx]
        return ret

    def gradient_V_s(self, s, reg_param):
        """
        Calculate the gradient of Vs

        :param s: row index
        :param reg_param: regularization parameter of V
        :return: n*1 ndarray of the row gradient
        """
        ret = np.zeros(self.n)
        for i in range(self.n):
            for j in range(self.n):
                if self.X_train[i][j] == 1 and self.X_train[i][s] == 0:
                    sigmoid1 = Recommend_Model.sigmoid(self.X_hat[i][s] - self.X_hat[i][j])
                    sigmoid2 = Recommend_Model.sigmoid(self.X_hat[i][j] - self.X_hat[i][s])
                    factor = 1.0 / (1.0 + self.approximate_rank_of_link(i, j, self.X_train))
                    ret += factor * sigmoid1 * sigmoid2 * self.U[i]
        ret += reg_param * self.V[s]
        return ret

    def gradient_theta(self, reg_param):
        """
        Calculate the gradient of theta

        :param reg_param: regularization parameter of explicit feature parameter theta
        :return: 3*1 ndarray of the gradient
        """
        ret = np.zeros(3)
        for i in range(self.n):
            for j in range(self.n):
                for s in range(self.n):
                    if self.X_train[i][j] == 1 and self.X_train[i][s] == 0:
                        factor = 1.0 / (1.0 + self.approximate_rank_of_link(i, j, self.X_train))
                        sigmoid1 = Recommend_Model.sigmoid(self.X_hat[i][s] - self.X_hat[i][j])
                        sigmoid2 = Recommend_Model.sigmoid(self.X_hat[i][j] - self.X_hat[i][s])
                        ret += factor * sigmoid1 * sigmoid2 * (self.G[i][s] - self.G[i][j])
        ret += reg_param * self.theta
        return ret

    def valid_error(self, u_reg_param, v_reg_param, theta_reg_param):
        """
        Calculate the validation error.

        :return: value of objective function
        """
        ret = 0.0
        for i in range(self.n):
            ret += u_reg_param * np.dot(self.U[i], self.U[i]) / 2.0
            ret += v_reg_param * np.dot(self.V[i], self.V[i]) / 2.0
            for j in range(self.n):
                ret += np.log(1 + self.approximate_rank_of_link(i, j, self.X_validation))
        if self.consider_explicit:
            ret += theta_reg_param * np.dot(self.theta, self.theta) / 2.0
        return ret

    def update_estimation(self):
        """
        Update the estimation matrix using low rank approximation matrix U, V
        :return:
        """
        self.X_hat = np.matmul(np.transpose(self.U), self.V)
        if self.consider_explicit:
            for i in range(self.n):
                for j in range(self.n):
                    self.X_hat[i][j] += np.dot(self.G[i][j], self.theta)

    def SGD_algorithm(self, u_reg_param, v_reg_param, theta_reg_param, batch_size, learning_rate, max_iter):
        """
        Implementation of the main algorithm.

        :param u_reg_param: the regularization parameter for U
        :param v_reg_param: the regularization parameter for V
        :param batch_size: batch size of SGD
        :param learning_rate: learning_rate for SGD
        :param max_iter: maximum iteration of SGD
        :param theta_reg_param: the regularization parameter for theta
        :return: U and V will be obtained when the function completes.
        """
        observed_link_index = []
        unknown_status_index = {}
        for i in range(self.n):
            for j in range(self.n):
                if self.X_train[i][j] == 1:
                    observed_link_index.append([i, j])
                else:
                    if i not in unknown_status_index:
                        unknown_status_index[i] = []
                    else:
                        unknown_status_index[i].append(j)

        iteration = 0
        prev_error = 0.0
        curr_error = self.valid_error(u_reg_param, v_reg_param, theta_reg_param)
        while True:
            if iteration > max_iter or np.abs(curr_error - prev_error) < 1e-5:
                break
            if iteration % 10 == 0:
                print("Iteration {}: {}".format(iteration, curr_error))
            iteration += 1

            # Randomly pick up an observed link
            i, j = np.random.choice(observed_link_index, 1, replace=False)[0]
            # Fix user i, uniformly draw batch_size unknown status link
            s_list = np.random.choice(unknown_status_index[i], batch_size, replace=False)
            for s in s_list:
                if np.dot(self.U[i], self.V[j]) < np.dot(self.U[i], self.V[s]):
                    grad_i = self.gradient_U_i(i, u_reg_param)
                    grad_j = self.gradient_V_j(j, v_reg_param)
                    grad_s = self.gradient_V_s(s, v_reg_param)
                    self.U[i] -= learning_rate * grad_i
                    self.V[j] -= learning_rate * grad_j
                    self.V[s] -= learning_rate * grad_s

            self.update_estimation()
            prev_error = curr_error
            curr_error = self.valid_error(u_reg_param, v_reg_param, theta_reg_param)

        return
