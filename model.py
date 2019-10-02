import numpy as np


class Recommend_Model:
    """ Model that running our recommendation algorithm"""
    n = 0  # number of nodes
    consider_explicit = False  # take explicit feature into consideration
    r = 30  # rank of approximate matrix
    X = np.array([])  # n*n Adjacency matrix
    X_hat = np.array([])  # n*n Estimated score matrix
    U = np.array([])  # r*n low rank approximation
    V = np.array([])  # r*n low rank approximation
    G = np.array([])  # n*n*3 explicit feature matrix
    theta = np.array([])  # 3*1 explicit feature param

    def __init__(self, X):
        self.X = X
        self.n = np.shape(X)[0]
        U = np.random.rand((self.r, self.n))
        V = np.random.rand((self.r, self.n))

    @staticmethod
    def sigmoid(z: float) -> float:
        return 1.0 / (1.0 + np.e ** (-z))

    def approximate_rank_of_link(self, row_idx: int, col_idx: int) -> float:
        """
        Calculate approximate rank R_hat given link

        :param row_idx: row index of the entry
        :param col_idx: column index of the entry
        :return: value of approximate rank of the link
        """
        ret = 0.0
        if self.X[row_idx][col_idx] == 1:
            for i in range(self.n):
                if self.X[row_idx][col_idx] == 1:
                    ret += Recommend_Model.sigmoid(self.X_hat[row_idx][i] - self.X[row_idx][col_idx])
        return ret

    def gradient_U_i(self, row_idx: int, reg_param: float) -> np.ndarray:
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
                if self.X[row_idx][j] == 1 and self.X[row_idx][s] == 0:
                    sigmoid1 = Recommend_Model.sigmoid(self.X_hat[row_idx][s] - self.X_hat[row_idx][j])
                    sigmoid2 = Recommend_Model.sigmoid(self.X_hat[row_idx][j] - self.X_hat[row_idx][s])
                    sum += sigmoid1 * sigmoid2 * (self.V[s] - self.V[j])
            ret += 1.0 / (1.0 + self.approximate_rank_of_link(row_idx, j)) * sum
        ret += reg_param * self.U[row_idx]
        return ret

    def gradient_V_j(self, column_idx: int, reg_param: float) -> np.ndarray:
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
                if self.X[i][column_idx] == 1 and self.X[i][s] == 0:
                    sigmoid1 = Recommend_Model.sigmoid(self.X_hat[i][s] - self.X_hat[i][column_idx])
                    sigmoid2 = Recommend_Model.sigmoid(self.X_hat[i][column_idx] - self.X_hat[i][s])
                    sum += sigmoid1 * sigmoid2 * -self.U[i]
            ret += 1.0 / (1.0 + self.approximate_rank_of_link(i, column_idx)) * sum
        ret += reg_param * self.V[column_idx]
        return ret

    def gradient_V_s(self, s: int, reg_param: float) -> np.ndarray:
        """
        Calculate the gradient of Vs

        :param s: row index
        :param reg_param: regularization parameter of V
        :return: n*1 ndarray of the row gradient
        """
        ret = np.zeros(self.n)
        for i in range(self.n):
            for j in range(self.n):
                if self.X[i][j] == 1 and self.X[i][s] == 0:
                    sigmoid1 = Recommend_Model.sigmoid(self.X_hat[i][s] - self.X_hat[i][j])
                    sigmoid2 = Recommend_Model.sigmoid(self.X_hat[i][j] - self.X_hat[i][s])
                    factor = 1.0 / (1.0 + self.approximate_rank_of_link(i, j))
                    ret += factor * sigmoid1 * sigmoid2 * self.U[i]
        ret += reg_param * self.V[s]
        return ret

    def gradient_theta(self, reg_param: float) -> np.ndarray:
        """

        :param reg_param: regularization parameter of explicit feature parameter theta
        :return: 3*1 ndarray of the gradient
        """
        ret = np.zeros(3)
        for i in range(self.n):
            for j in range(self.n):
                for s in range(self.n):
                    if self.X[i][j] == 1 and self.X[i][s] == 0:
                        factor = 1.0 / (1.0 + self.approximate_rank_of_link(i, j))
                        sigmoid1 = Recommend_Model.sigmoid(self.X_hat[i][s] - self.X_hat[i][j])
                        sigmoid2 = Recommend_Model.sigmoid(self.X_hat[i][j] - self.X_hat[i][s])
                        ret += factor * sigmoid1 * sigmoid2 * (self.G[i][s] - self.G[i][j])
        ret += reg_param * self.theta
        return ret

    def SGD_algorithm(self, u_reg_param, v_reg_param, theta_reg_param, step, max_iter):
        """

        :param u_reg_param: the regularization parameter for U
        :param v_reg_param: the regularization parameter for V
        :param step: step length for SGD
        :param max_iter: maximum iteration of SGD
        :param theta_reg_param: the regularization parameter for theta
        :return: U and V will be obtained when the function completes.
        """
        t = 0
        
        return