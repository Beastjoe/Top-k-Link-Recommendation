import numpy as np
import random
import sklearn.metrics


def generate_dataset(X, train_fraction, valid_fraction=0.1, test_fraction=0.3):
    """
    Generate dataset for SGD algorithm

    :param X: the original n*n adjacency matrix
    :param train_fraction: the fraction to keep the observed links in train set
    :param valid_fraction: the fraction to keep the observed links in valid set
    :param test_fraction: the fraction to keep the observed links in test set
    :return: n*n train matrix, n*n valid matrix, n*n test matrix, test list
    """
    X_train = np.zeros(np.shape(X))
    X_valid = np.zeros(np.shape(X))
    X_test = np.zeros(np.shape(X))
    test_list = []
    random.seed()

    for i in range(np.shape(X)[0]):
        count = 0
        for j in range(np.shape(X)[1]):
            if X[i][j] == 1:
                p = random.random()
                if p >= 1 - test_fraction:
                    X_test[i][j] = 1
                    count += 1
                    if count == 3:
                        test_list.append(i)  # we only test nodes with minimum degree 3
                elif p >= 1 - test_fraction - valid_fraction:
                    X_valid[i][j] = 1
                elif p <= train_fraction:
                    X_train[i][j] = 1

    return X_train, X_valid, X_test, test_list


def test(X_estimate, X_test, test_list, method="AUC", k=1):
    """
    Calculate test result using different metrics

    :param X_estimate: The Estimation given by model
    :param X_test: The ground truth
    :param method: test method ["AUC", "MAP", "precision", "recall"]
    :param k: top k links
    :return: test result
    """
    result = 0.0
    if method == "AUC":
        # return average AUC score
        for idx in test_list:
            score = sklearn.metrics.roc_auc_score(X_test[idx], X_estimate[idx])
            result += score
    elif method == "MAP":
        # return average precision
        for idx in test_list:
            score = sklearn.metrics.average_precision_score(X_test[idx], X_estimate[idx])
            result += score
    elif method == "precision":
        # return average precision@k
        for idx in test_list:
            ind = np.argpartition(X_estimate[idx], -k)[-k:]
            count = 0.0
            for i in ind:
                if X_test[idx][i] == 1:
                    count += 1.0
            result += count / k
    elif method == "recall":
        # return average recall@k
        for idx in test_list:
            ind = np.argpartition(X_estimate[idx], -k)[-k:]
            count = 0.0
            for i in ind:
                if X_test[idx][i] == 1:
                    count += 1.0
            total = 0.0
            for link in X_test[idx]:
                if link == 1:
                    total += 1.0
            result += count / total

    return result / len(test_list)
