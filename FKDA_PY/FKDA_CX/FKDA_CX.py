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
from FKDA_CX_Utils import *

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
l = len(Ytr)
cl = 5

print('Ytr:', Ytr)
print('l:', l)
print('cl:', cl)


## ============================= offline =============================
# compute the centroid matrix #
# construct the kernel matrix #
# construct the kernel vector #
print('<=== Batch/Offline Stage ===>')
tic = time.time()

e = np.ones([1, cl])
C_m, M, N = KFDA_CX(Xtr, ytr, cl)
K_cm = KGaussian(C_m)
K_cm_inv = np.linalg.inv(K_cm)
K_cmz = Gaussian(C_m, Xte)

toc = time.time()
t_K = toc - tic

tic = time.time()

E = np.empty([0, 0])
for i in range(l):
    E = np.r_[
            np.c_[E, np.zeros([E.shape[0], cl])],
            np.c_[np.zeros([1, E.shape[1]]), e],
            ]

P = np.matmul(np.matmul(E, K_cm_inv), K_cmz)
P_c = np.eye(l)

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
ytr = np.reshape(np.kron(Ytr, np.ones([1, cl])), [-1])
new_C_m = C_m.copy()
new_K_cm = K_cm.copy()

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

    alabel = pd.unique(z_train_y)
    XXte = len(z_train_y)
    cll = int(XXte / len(alabel))

    Xte = np.c_[Xte, z_test_X]
    yte = np.append(yte, z_test_y)

    # update the centroid matrix
    # update the kernel matrix
    # update the kernel vector
    tic = time.time()

    new_C_m, new_K_cm, M, N, Ytr, ytr, l = Inc_KFDA(new_C_m, new_K_cm, M, N, z, Ytr, ytr, cl, cll, DATASET=DATASET)
    K_cmz = ch_Gaussian(new_C_m, Xte, K_cmz, alabel, ytr)
    new_K_cm_inv = np.linalg.inv(new_K_cm)

    E = np.empty([0, 0])
    l = int(l / cl)
    for ii in range(l):
        E = np.r_[np.c_[E, np.zeros([E.shape[0], cl])], np.c_[np.zeros([1, E.shape[1]]), e]]

    new_P = np.matmul(np.matmul(E, new_K_cm_inv), K_cmz)
    new_P_c = np.eye(l)

    toc = time.time()
    sum_t = toc - tic

    predictLabel, precision, t_p, probability = predictWrap(new_P_c.T, Ytr, new_P.T, yte)

    T_sum = np.r_[T_sum, sum_t]
    T_p = np.r_[T_p, t_p]
    Pre = np.r_[Pre, precision]

    print('Precision:', Pre)
    print('T_sum:', T_sum)
    print('T_p:', T_p)


