import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier


# Params: A must be 2-dimension
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
                # K[i, j] = np.exp(-np.linalg.norm(dis)**2 / 1.)
                K[i, j] = np.exp(-np.linalg.norm(dis) ** 2 / gamma)

    return K


# Params: X must be 2-dimension, XLabel is 1-dimension
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


def Inc_KFDAnew(Xte, K_xz, new_X, K, N, a, Ytr, DATASET):
    d, n = new_X.shape
    a_train_y = np.reshape(a['train'][0, 0]['y'][0, 0], [-1])
    if DATASET == 'Caltech256' or DATASET == 'fashion_mnist':  # MNIST or Caltech256
        a_test_X = a['test'][0, 0]['X'][0, 0]
        a_train_X = a['train'][0, 0]['X'][0, 0]
    elif DATASET == 'AWA':  # AWA
        a_test_X = a['test'][0, 0]['X'][0, 0].toarray()
        a_train_X = a['train'][0, 0]['X'][0, 0].toarray()

    N = np.reshape(N, [-1])
    old_N = N.copy()
    old_c = len(old_N)

    m = Xte.shape[1]
    y = pd.unique(a_train_y)
    new_Ytr = Ytr.copy()

    K_xz_pro = Gaussian(new_X, a_test_X)
    K_xz = np.c_[K_xz, K_xz_pro]

    for i in range(len(y)):
        # update for known classes
        loc1 = np.where(a_train_y == y[i])[0]
        if y[i] in Ytr:
            loc = np.where(Ytr == y[i])[0]
            loc = loc[0]
            new_X = np.c_[new_X[:, 0:int(sum(N[0:loc+1]))],
                            np.reshape(a_train_X[:, loc1], [d, -1]),
                            new_X[:, int(sum(N[0:loc+1]))+1-1: int(sum(N))],
                            ]
            K = np.r_[np.c_[K[0:int(sum(N[0:loc+1])), 0:int(sum(N[0:loc+1]))],
                                np.zeros([int(sum(N[0: loc+1])), len(loc1)]),
                                K[0:int(sum(N[0:loc+1])), int(sum(N[0:loc+1]))+1-1:int(sum(N))],
                            ],
                        np.zeros([len(loc1), int(sum(N))+len(loc1)]),
                        np.c_[K[int(sum(N[0:loc+1]))+1-1: int(sum(N)), 0:int(sum(N[0:loc+1]))],
                                np.zeros([int(sum(N))-int(sum(N[0: loc+1])), len(loc1)]),
                                K[int(sum(N[0: loc+1]))+1-1: int(sum(N)), int(sum(N[0:loc+1]))+1-1:int(sum(N))]
                            ],
                        ]
            K_xz = np.r_[K_xz[0: int(sum(N[0:loc+1])), :],
                            np.zeros([len(loc1), m]),
                            K_xz[int(sum(N[0:loc+1]))+1-1: int(sum(N)), :]
                        ]

            K_xz[int(sum(N[0:loc+1]))+1-1: int(sum(N[0:loc+1])+len(loc1)), :] = Gaussian(a_train_X[:, loc1], Xte)

            K[:, int(sum(N[0:loc+1]))+1-1: int(sum(N[0:loc+1]))+len(loc1)] = \
                Gaussian(new_X, new_X[:, int(sum(N[0:loc+1]))+1-1: int(sum(N[0:loc+1]))+len(loc1)])

            K[int(sum(N[0:loc+1]))+1-1: int(sum(N[0:loc+1]))+len(loc1), :] = \
                np.transpose(K[:, int(sum(N[0:loc+1]))+1-1: int(sum(N[0:loc+1]))+len(loc1)])

            N[loc] = N[loc] + len(loc1)
        else:
            # update for novel classes
            new_X = np.c_[new_X, a_train_X[:, loc1]]
            K = np.r_[np.c_[K, np.zeros([K.shape[0], len(loc1)])],
                        np.c_[np.zeros([len(loc1), K.shape[1] + len(loc1)])]]
            K_xz = np.r_[K_xz, Gaussian(a_train_X[:, loc1], Xte)]

            K[:, int(sum(N))+1-1: int(sum(N))+len(loc1)] = \
                Gaussian(new_X, new_X[:, int(sum(N))+1-1: int(sum(N))+len(loc1)])

            K[int(sum(N))+1-1: int(sum(N))+len(loc1), :] = \
                np.transpose(K[:, int(sum(N))+1-1: int(sum(N))+len(loc1)])

            N = np.append(N, len(loc1))
            new_Ytr = np.append(new_Ytr, y[i])

    c = len(N)

    return K_xz, new_X, K, N, new_Ytr, c


