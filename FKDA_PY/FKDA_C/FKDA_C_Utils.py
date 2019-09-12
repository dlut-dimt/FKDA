import pandas as pd
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier

# Utility Functions


def KFDA(X, XLabel):
    d, n = X.shape
    CLabel = pd.unique(XLabel)
    c = len(CLabel)
    N = np.zeros(c)
    C = np.zeros((d, c))
    for i in range(c):
        loc = np.where(XLabel == CLabel[i])  # np.where returns a tuple (np.array,)
        loc = loc[0]
        N[i] = len(loc)
        C[:, i] = np.mean(X[:, loc], axis=1).reshape([-1])
    return C, N, CLabel, c


def KGaussian(A):
    gamma, r = A.shape
    K1 = np.zeros((r, r))
    for i in range(r):
        for j in range(i, r):
            dis = A[:, i].reshape([-1, 1]) - A[:, j].reshape([-1, 1])     # dis.shape == (gamma,)
            # K1[i, j] = np.exp(-(np.linalg.norm(dis)**2 / 1.))   # np.linalg.norm default 2-norm
            K1[i, j] = np.exp(-np.linalg.norm(dis) ** 2 / gamma)

    K = K1.T + K1 - np.eye(r)           # K.shape == (r, r)
    return K


def Gaussian(X, z):
    gamma, r = X.shape
    m, mm = z.shape
    K = np.zeros((r, mm))
    if gamma != m:
        raise ValueError('The dimension of input data is inconsistent!')
    else:
        for i in range(r):
            for j in range(mm):
                dis = X[:, i].reshape([-1, 1]) - z[:, j].reshape([-1, 1])
                # K[i, j] = np.exp(-np.linalg.norm(dis)**2 / 1.)
                K[i, j] = np.exp(-np.linalg.norm(dis) ** 2 / gamma)

    return K


def predictWrap(trainData, trainLabel, testData, testLabel):
    tic = time.time()
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(trainData, trainLabel)
    predictLabel = model.predict(testData)
    probability = model.predict_proba(testData)
    precision = np.double(np.sum(predictLabel == testLabel.T)) / len(testLabel)
    toc = time.time()
    t_p = toc - tic
    return predictLabel, precision, t_p, probability


def Inc_KFDA(K_C, N, a, C, Ytr):
    a_train_X = a['train'][0, 0]['X'][0, 0]
    a_train_y = np.reshape(a['train'][0, 0]['y'][0, 0], [-1])

    old_N = N.copy()
    old_c = len(old_N)

    new_C = C
    y = pd.unique(a_train_y)
    new_Ytr = Ytr

    for i in range(len(y)):
        loc1 = np.where(a_train_y == y[i])
        loc1 = loc1[0]
        if y[i] in Ytr:
            loc = np.where(Ytr == y[i])[0]
            N[loc] = N[loc] + len(loc1)
            new_C[:, loc] = (old_N[loc].item() * new_C[:, loc].reshape([-1, 1]) +
                             np.sum(a_train_X[:, loc1], axis=1).reshape([-1, 1])) / N[loc]
        else:
            N = np.r_[N, len(loc1)]
            new_C = np.c_[new_C, np.mean(a_train_X[:, loc1], axis=1).reshape([-1, 1])]
            new_Ytr = np.r_[new_Ytr, y[i]]

    c = len(N)

    # Update K_C

    new_K_C = np.zeros([c, c])
    new_K_C[: old_c, :old_c] = K_C
    for i in range(len(y)):
        loc1 = np.where(a_train_y == y[i])[0]
        if y[i] in Ytr:
            loc = np.where(Ytr == y[i])[0]
            for j in range(old_c):
                new_K_C[j, loc] = Gaussian(new_C[:, j].reshape([-1, 1]),
                                           new_C[:, loc].reshape([-1, 1]))
                new_K_C[loc, j] = new_K_C[j, loc]
        else:
            loc = np.where(new_Ytr == y[i])[0]
            for j in range(c):
                new_K_C[j, loc] = Gaussian(new_C[:, j].reshape([-1, 1]),
                                           new_C[:, loc].reshape([-1, 1]))
                new_K_C[loc, j] = new_K_C[j, loc]

    new_K_C_inv = np.linalg.inv(new_K_C)

    return new_K_C, new_K_C_inv, N, new_C, new_Ytr, c


def ch_Gaussian(new_C, z, K_xz, alabel, Ytr):
    gamma, r = new_C.shape
    m, mm = z.shape
    K = np.zeros((r, mm))
    a, b = K_xz.shape
    if gamma != m:
        raise ValueError('The dimension of input data is inconsistent!')
    else:
        K[:a, :b] = K_xz
        for ii in range(len(alabel)):
            i = np.where(Ytr == alabel[ii])[0][0]
            for j in range(b):
                dis = new_C[:, i].reshape([-1, 1]) - z[:, j].reshape([-1, 1])
                # K[i, j] = np.exp(-np.linalg.norm(dis)**2 / 1.)
                K[i, j] = np.exp(-np.linalg.norm(dis) ** 2 / gamma)

        for i in range(r):
            for j in range(b, mm):
                dis = new_C[:, i].reshape([-1, 1]) - z[:, j].reshape([-1, 1])
                # K[i, j] = np.exp(-np.linalg.norm(dis)**2 / 1.)
                K[i, j] = np.exp(-np.linalg.norm(dis) ** 2 / gamma)

    return K




