# ===================== FKDA_C.py(Demo_Caltech256) ====================== #
# "Novelty Detection and Online Learning for Chunk Data Streams"          #
# Y. Wang, Y. Ding, X. He, X. Fan, C. Lin, F. Li, T. Wang, Z. Luo, J. Luo #
# TPAMI-2019-02-0102                                                      #
# Y. Wang [dlutwangyi@dlut.edu.cn]                                        #
# Please email me if you find bugs, or have suggestions or questions!     #
# ======================================================================= #

import numpy as np
import pandas as pd
import scipy.io as sio
import time
from FKDA_X_Utils import *

Pre = []

DATASET = 'Caltech256'

if DATASET == 'Caltech256':
    data = sio.loadmat('../Data/Caltech256/Caltech256_chunk_non.mat')
    Xtr = data['batch']['train'][0, 0]['X'][0, 0]  # [dimensions, samlpes]
    Xte = data['batch']['test'][0, 0]['X'][0, 0]

ytr = data['batch']['train'][0, 0]['y'][0, 0]  # [1, samples]
ytr = np.reshape(ytr, [-1])
yte = data['batch']['test'][0, 0]['y'][0, 0]
yte = np.reshape(yte, [-1])

d, n = Xtr.shape    # d == dimensions, n == samples
print('d:', d, '\tn:', n)

Ytr = pd.unique(ytr)
c = len(Ytr)
print('Ytr:', Ytr)
print('c:', c)


## ============================= offline =============================
# compute the centroid matrix #
# construct the kernel matrix #
# construct the kernel vector #
print('<=== Batch Stage ===>')
tic = time.time()

K = KGaussian(Xtr)  # kernel matrix
K_inv = np.linalg.pinv(K)   # pseudo-inverse matrix of kernel matrix
K_xz = Gaussian(Xtr, Xte)   # kernel vector

toc = time.time()
t_K = toc - tic

N = np.zeros([c], dtype=np.int)
for i in range(c):
    loc = np.where(ytr == Ytr[i])[0]
    N[i] = len(loc)

E = np.ones([1, int(N[0])])
for m in range(1, len(N)):
    E = np.r_[np.c_[E, np.zeros([E.shape[0], N[m]])],
              np.c_[np.zeros([1, E.shape[1]]), np.ones([1, N[m]])],
             ]

tic = time.time()

P = np.matmul(np.matmul(E, K_inv), K_xz)    # compute the projection of samples
P_c = np.eye(c)
toc = time.time()
t_batch = toc - tic

predictLabel, precision, t_p, probability = predictWrap(P_c.T, Ytr, P.T, yte)

print('Pre:', precision)
print('t_batch:', t_batch)
print('t_p:', t_p)

T_p = []
T_p = np.r_[T_p, t_p]
T_sum = []
T_sum = np.r_[T_sum, t_K]
T_sum = np.r_[T_sum, t_batch]
Pre = []
Pre = np.r_[Pre, precision]


## ============================== online ==============================
new_X = Xtr.copy()

chunk = data['Inc']  # Inc[0:len]
n_chunk = len(chunk[0])
print('n_chunk:', n_chunk)
print('<=== Chunk Stage ===>')

for i in range(n_chunk):
    print('<===== chunk %d =====>' % (i + 1))
    z = chunk[0, i]

    if DATASET == 'Caltech256':
        z_train_X = z['train'][0, 0]['X'][0, 0]
        z_test_X = z['test'][0, 0]['X'][0, 0]

    z_train_y = np.reshape(z['train'][0, 0]['y'][0, 0], [-1])
    z_test_y = np.reshape(z['test'][0, 0]['y'][0, 0], [-1])

    Xte = np.c_[Xte, z_test_X]
    yte = np.append(yte, z_test_y)

    tic = time.time()

    K_xz, new_X, K, N, Ytr, c = Inc_KFDAnew(Xte, K_xz, new_X, K, N, z, Ytr, DATASET=DATASET)
    # calculate the updated data matrix new_X, and the updated kernel matrix K etc.
    new_K_inv = np.linalg.pinv(K)
    E = np.ones([1, N[0]])
    for m in range(1, len(N)):
        E = np.r_[np.c_[E, np.zeros([E.shape[0], N[m]])],
                  np.c_[np.zeros([1, E.shape[1]]), np.ones([1, N[m]])],
                ]

    P = np.matmul(np.matmul(E, new_K_inv), K_xz)
    P_c = np.eye(c)

    toc = time.time()
    sum_t = toc - tic

    predictLabel, precision, t_p, probability = predictWrap(P_c.T, Ytr, P.T, yte)
    T_sum = np.r_[T_sum, sum_t]
    T_p = np.r_[T_p, t_p]
    Pre = np.r_[Pre, precision]

    print('Precision:', Pre)
    print('T_sum:', T_sum)
    print('T_p:', T_p)


