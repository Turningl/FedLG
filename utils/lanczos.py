# -*- coding: utf-8 -*-
# @Author : liang
# @Email : zl16035056@163.com
# @File : lanczos.py

import os
import torch
import numpy as np


def Lanczos(mat, m):
    # reference: https://en.wikipedia.org/wiki/Lanczos_algorithm
    mat = mat.cpu().numpy()
    n = mat[0].shape[0]
    v0 = np.random.rand(n)
    v0 /= np.sqrt(np.dot(v0, v0))

    if n == 1:
        # handle 1X1 matrix
        T, V = np.array([mat[0]]), np.array([[1]])
        return T, V

    V = np.zeros((m, n))
    T = np.zeros((m, m))
    V[0, :] = v0

    # step 2.1 - 2.3
    w = np.sum([np.dot(col, np.dot(np.conj(col.T), V[0, :])) for col in mat], 0)
    alfa = np.dot(w, V[0, :])
    w = w - alfa * V[0, :]
    T[0, 0] = alfa

    # needs to start the iterations from indices 1
    for j in range(1, m - 1):

        beta = np.sqrt(np.dot(w, w))
        V[j, :] = w / beta

        # This performs some rediagonalization to make sure all the vectors
        # are orthogonal to eachother
        for i in range(j - 1):
            V[j, :] = V[j, :] - np.dot(np.conj(V[j, :]), V[i, :]) * V[i, :]
        V[j, :] = V[j, :] / np.linalg.norm(V[j, :])

        w = np.sum([np.dot(col, np.dot(np.conj(col.T), V[j, :])) for col in mat], 0)
        alfa = np.dot(w, V[j, :])
        w = w - alfa * V[j, :] - beta * V[j - 1, :]

        T[j, j] = alfa
        T[j - 1, j] = beta
        T[j, j - 1] = beta

    # T = np.nan_to_num(T, nan=0, posinf=0, neginf=0)

    return T, V


def lanczos_algorithm(A, k=128):
    n = A.shape[0]
    Q = np.zeros((n, k))
    T = np.zeros((k, k))

    q = np.random.rand(n)
    q /= np.linalg.norm(q)
    Q[:, 0] = q

    for j in range(1, k):
        v = A @ Q[:, j - 1]
        for i in range(j):
            T[i, j] = Q[:, i] @ v
            v -= T[i, j] * Q[:, i]

        T[j - 1, j] = np.linalg.norm(v)
        if T[j - 1, j] == 0:
            break

        Q[:, j] = v / T[j - 1, j]

    return T, Q