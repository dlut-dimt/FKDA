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
from FKDA_C_Utils import *


DATASET = 'Caltech256'

if DATASET == 'Caltech256':
    data = sio.loadmat('../Data/Caltech256/Caltech256_chunk_non.mat')
    X_train = data['batch']['train'][0, 0]['X'][0, 0]  # [dimensions, samlpes]
    X_test = data['batch']['test'][0, 0]['X'][0, 0]

y_train = data['batch']['train'][0, 0]['y'][0, 0]  # [1, samples]
y_train = np.reshape(y_train, [-1])
y_test = data['batch']['test'][0, 0]['y'][0, 0]
y_test = np.reshape(y_test, [-1])

d, n = X_train.shape    # d == dimensions, n == samples
print('d:', d, '\tn:', n)

Ytr = pd.unique(y_train)
c = len(Ytr)
print('Ytr:', Ytr)
print('c:', c)


## ============================= offline =============================
# compute the centroid matrix #
# construct the kernel matrix #
# construct the kernel vector #
print('<====== Batch stage ======>')
tic = time.time()

C, N, _, _ = KFDA(X_train, y_train)
K_C = KGaussian(C) # kernel matrix
K_C_inv = np.linalg.inv(K_C) # pseudo-inverse matrix of kernel matrix
K_cz = Gaussian(C, X_test) # kernel vector

toc = time.time()
t_K = toc - tic

tic = time.time()
P = np.dot(K_C_inv, K_cz)
P_c = np.eye(c)
toc = time.time()
t_batch = toc - tic
print('t_batch:', t_batch)

predictLabel, precision, t_p, probability = predictWrap(P_c.T, Ytr, P.T, y_test)

print('KNN Precision:', precision)
print('t_p:', t_p)




## ============================== online ==============================
print('\n\n<====== Chunk stage ======>')
new_X = X_train
new_XLabel = y_train
new_C = C
new_K_C = K_C

T_p = []
T_p = np.r_[T_p, t_p]
T_sum = []
T_sum = np.r_[T_sum, t_K]
T_sum = np.r_[T_sum, t_batch]

Pre = []
Pre = np.r_[Pre, precision]

chunk = data['Inc']   # Inc[0:5]
n_chunk = len(chunk[0])
print('n_chunk:', n_chunk)

for i in range(n_chunk):
    print('<===== chunk %d =====>' % (i + 1))
    z = chunk[0, i]
    if DATASET == 'Caltech256':
        z_train_X = z['train'][0, 0]['X'][0, 0]
        z_test_X = z['test'][0, 0]['X'][0, 0]

    z_train_y = np.reshape(z['train'][0, 0]['y'][0, 0], [-1])
    z_test_y = np.reshape(z['test'][0, 0]['y'][0, 0], [-1])

    alabel = pd.unique(z_train_y)

    Xte = np.c_[X_test, z_test_X]
    yte = np.r_[y_test, z_test_y]

    tic = time.time()
    new_K_C, new_K_C_inv, N, new_C, Ytr, c = Inc_KFDA(new_K_C, N, z, new_C, Ytr)
    K_cz = ch_Gaussian(new_C, Xte, K_cz, alabel, Ytr)
    P = np.dot(new_K_C_inv, K_cz)
    P_c = np.eye(c)
    toc = time.time()
    sum_t = toc - tic

    predictLabel, precision, t_p, probability = predictWrap(P_c.T, Ytr, P.T, yte)
    T_sum = np.r_[T_sum, sum_t]
    T_p = np.r_[T_p, t_p]
    Pre = np.r_[Pre, precision]
    print('KNN Acc:', precision)


print('T_sum:', T_sum)
print('T_p:', T_p)
print('KNN Precision:', Pre)



