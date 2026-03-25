import numpy as np


def coskewness(X):
    n = X.shape[0]
    means = np.mean(X, axis=0)
    #Standard deviations of each variable
    std_devs = np.std(X, axis=0, ddof=1)
    coskew = np.zeros((X.shape[1], X.shape[1], X.shape[1]))

    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            for k in range(X.shape[1]):
                coskew[i, j, k] = np.sum(
                    (X[:, i] - means[i]) * (X[:, j] - means[j]) * (X[:, k] - means[k])
                ) / n
                
                # Normalize by the product of the standard deviations
                coskew[i, j, k] /= (std_devs[i] * std_devs[j] * std_devs[k])

    return coskew
    
    
def cokurtosis(X):
    n = X.shape[0]
    means = np.mean(X, axis=0)
    std_devs = np.std(X, axis=0, ddof=1)
    cokurt = np.zeros((X.shape[1], X.shape[1], X.shape[1], X.shape[1]))

    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            for k in range(X.shape[1]):
                for l in range(X.shape[1]):
                    cokurt[i, j, k, l] = np.sum(
                        (X[:, i] - means[i]) *
                        (X[:, j] - means[j]) *
                        (X[:, k] - means[k]) *
                        (X[:, l] - means[l])
                    ) / n

                    # Normalize by the product of the standard deviations
                    cokurt[i, j, k, l] /= (std_devs[i] * std_devs[j] * std_devs[k] * std_devs[l])

    return cokurt