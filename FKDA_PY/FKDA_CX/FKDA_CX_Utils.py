import numpy as np
import pandas as pd
import scipy.io as sio
import time
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


# Params: X must be 2-dimension, XLabel is 1-dimension, CL is integer
def KFDA_CX(X, XLabel, CL):
    d, n = X.shape
    CLabel = pd.unique(XLabel)
    c = len(CLabel)
    M = np.reshape(np.zeros([1, c]), [c])
    P = np.empty([0, d])
    N = []
    for i in range(c):
        loc = np.where(XLabel == CLabel[i])[0]  # np.where returns a tuple (np.array,)
        v = X[:, loc].reshape([-1, len(loc)])
        MM = np.array([[0]*CL])
        for j in range(CL):
            MM[0, j] = len(loc)
        N = np.append(N, MM)
        kmeans = KMeans(n_clusters=CL, random_state=0).fit(v.T)
        C = kmeans.cluster_centers_  # KxP matrix
        M[i] = CL
        P = np.r_[P, C]

    P = P.T

    return P, M, N


# Params: A must be 2-dimension
def KGaussian(A):
    gamma, r = A.shape
    K1 = np.zeros((r, r))
    for i in range(r):
        for j in range(i, r):
            dis = A[:, i].reshape([-1, 1]) - A[:, j].reshape([-1, 1])     # dis.shape == (gamma,)
            K1[i, j] = np.exp(-(np.linalg.norm(dis)**2 / 1.))   # np.linalg.norm default 2-norm
            # K1[i, j] = np.exp(-np.linalg.norm(dis) ** 2 / gamma)

    K = K1.T + K1 - np.eye(r)           # K.shape == (r, r)

    return K


# Params: X, z must be 2-dimension
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
                K[i, j] = np.exp(-np.linalg.norm(dis)**2 / 1.)
                # K[i, j] = np.exp(-np.linalg.norm(dis) ** 2 / gamma)

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


def Inc_KFDA(new_C, K_c, M, N, a, Ytr, ytr, cl, cll, DATASET):
    d, n = new_C.shape
    a_train_y = np.reshape(a['train'][0, 0]['y'][0, 0], [-1])
    if DATASET == 'Caltech256' or DATASET == 'fashion_mnist':  # MNIST or Caltech256
        a_test_X = a['test'][0, 0]['X'][0, 0]
        a_train_X = a['train'][0, 0]['X'][0, 0]
    elif DATASET == 'AWA':  # AWA
        a_test_X = a['test'][0, 0]['X'][0, 0].toarray()
        a_train_X = a['train'][0, 0]['X'][0, 0].toarray()

    new_M = M.copy()
    old_N = np.reshape(N.copy(), [-1])
    old_n = len(old_N)
    old_ytr = ytr.copy()

    y = pd.unique(a_train_y)
    alable = []

    # update new_C
    for i in range(len(y)):
        # update for known classes
        loc1 = np.where(a_train_y == y[i])[0]
        new_XX = np.empty([d, 0])
        new_yy = []
        if y[i] in Ytr:
            loc = np.where(Ytr == y[i])[0]
            loc = loc[0]
            new_MM = M.copy()
            for t in range(len(loc1)):
                new_C = np.c_[new_C[:, 0: int(sum(new_MM[0:loc+1]))],
                              np.reshape(a_train_X[:, loc1[t]], [d, -1]),
                              new_C[:, int(sum(new_MM[0: loc+1]))+1-1: int(sum(new_MM))],
                            ]
                ytr = np.append(
                                np.append(ytr[0: int(sum(new_MM[0:loc+1]))], a_train_y[loc1[t]]),
                                ytr[int(sum(new_MM[0:loc+1]))+1-1: int(sum(new_MM))]
                                )

                new_XX = np.c_[new_XX, np.reshape(a_train_X[:, loc1[t]], [d, -1])]
                new_yy = np.append(new_yy, a_train_y[loc1[t]])
                new_MM[loc] = new_MM[loc] + 1

            new_MM = M.copy()
            new_XX = np.c_[new_C[:, int(sum(new_MM[0:loc-1+1]))+1-1: int(sum(new_MM[0:loc+1]))], new_XX]
            new_yy = np.append(ytr[int(sum(new_MM[0:loc-1+1]))+1-1: int(sum(new_MM[0:loc+1]))], new_yy)
            new_XX, p, o = KFDA_CX(new_XX, new_yy, cl)
            new_yy = np.reshape(np.kron(new_yy[0], np.ones([1, cl])), [-1])

            cll = int(cll)
            new_C = np.c_[
                            new_C[:, 0:int(sum(M[0:loc-1+1]))],
                            new_XX,
                            new_C[:, int(sum(M[0:loc-1+1]))+cll+cl+1-1: int(sum(M))+cll],
                            ]
            ytr = np.append(
                            np.append(ytr[0:np.sum(M[0:loc-1+1], dtype=np.int)],new_yy),
                            ytr[int(sum(M[0:loc-1+1]))+cl+cll+1-1: int(sum(M))+cll]
                            )
            alable = np.append(alable, y[i])
        else:
            # update for novel classes
            for t in range(len(loc1)):
                new_XX = np.c_[new_XX, a_train_X[:, loc1[t]]]
                new_yy = np.append(new_yy, a_train_y[loc1[t]])

            new_XX, p, o = KFDA_CX(new_XX, new_yy, cl)
            N = np.append(N, o)
            new_yy = np.reshape(np.kron(new_yy[0], np.ones([1, cl])), [-1])
            new_C = np.c_[new_C, new_XX]
            ytr = np.append(ytr, new_yy)
            new_M = np.append(new_M, cl)
            Ytr = np.append(Ytr, y[i])

    c = len(ytr)

    # update K_c
    new_K_c = np.zeros([c, c])
    new_K_c[0:old_n, 0:old_n] = K_c
    # for i in range(old_n):
    #     for j in range(old_n):
    #         new_K_c[i, j] = K_c[i, j].copy()

    for i in range(len(y)):
        if y[i] in old_ytr:
            loc1 = np.where(old_ytr == y[i])[0]
            for j in range(old_n):
                for t in range(len(loc1)):
                    new_K_c[j, loc1[t]] = Gaussian(np.reshape(new_C[:, j], [-1, 1]),
                                                   np.reshape(new_C[:, loc1[t]], [-1, 1]))
                    new_K_c[loc1[t], j] = new_K_c[j, loc1[t]]

        else:
            # update for novel classes
            loc1 =np.where(ytr == y[i])[0]
            for j in range(c):
                for t in range(len(loc1)):
                    new_K_c[j, loc1[t]] = Gaussian(np.reshape(new_C[:, j], [-1, 1]), np.reshape(new_C[:, loc1[t]], [-1, 1]))
                    new_K_c[loc1[t], j] = new_K_c[j, loc1[t]]

    M = new_M.copy()

    return new_C, new_K_c, M, N, Ytr, ytr, c


def ch_Gaussian(new_C, z, K_xz, alabel, Ytr):
    gamma, r = new_C.shape
    m, mm = z.shape
    K = np.zeros((r, mm))
    a, b = K_xz.shape
    if gamma != m:
        raise ValueError('The dimension of input data is inconsistent!')
    else:
        K[0:a, 0:b] = K_xz
        for ii in range(len(alabel)):
            loc2 = np.where(Ytr == alabel[ii])[0]
            for j in range(b):
                for t in range(len(loc2)):
                    dis = new_C[:, loc2[t]].reshape([-1, 1]) - z[:, j].reshape([-1, 1])
                    K[loc2[t], j] = np.exp(-np.linalg.norm(dis)**2 / 1.)
                    # K[loc2[t], j] = np.exp(-np.linalg.norm(dis) ** 2 / gamma)

        for i in range(r):
            for j in range(b, mm):
                dis = new_C[:, i].reshape([-1, 1]) - z[:, j].reshape([-1, 1])
                K[i, j] = np.exp(-np.linalg.norm(dis)**2 / 1.)
                # K[i, j] = np.exp(-np.linalg.norm(dis) ** 2 / gamma)

    return K