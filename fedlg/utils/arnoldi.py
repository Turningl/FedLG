# -*- coding: utf-8 -*-
# @Author : liang
# @File : arnoldi.py


import os
import torch
import numpy as np


def lanczos_algorithm(A, k=128):
    """
    Compute the Lanczos tridiagonalization of a Hermitian matrix.

    Args:
        A (numpy.ndarray): A Hermitian matrix.
        k (int, optional): The number of Lanczos iterations. Defaults to 128.

    Returns:
        T (numpy.ndarray): The tridiagonal matrix.
        Q (numpy.ndarray): The matrix of Lanczos vectors.
    """
    n = A.shape[0]
    Q = np.zeros((n, k))
    T = np.zeros((k, k))

    # Initialize a random starting vector and normalize it.
    q = np.random.rand(n)
    q /= np.linalg.norm(q)
    Q[:, 0] = q

    # Iterate to compute the Lanczos vectors and tridiagonal elements.
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


def Lanczos(mat, m):
    """
    Compute the Lanczos tridiagonalization of a Hermitian matrix.

    Args:
        mat (numpy.ndarray): A Hermitian matrix.
        m (int): The number of Lanczos iterations.

    Returns:
        T (numpy.ndarray): The tridiagonal matrix.
        V (numpy.ndarray): The matrix of Lanczos vectors.
    """
    # Convert the matrix to a NumPy array and move it to the CPU if necessary.
    mat = mat.cpu().numpy()
    n = mat[0].shape[0]

    # Initialize a random starting vector and normalize it.
    v0 = np.random.rand(n)
    v0 /= np.sqrt(np.dot(v0, v0))

    # Handle the special case of a 1x1 matrix.
    if n == 1:
        T, V = np.array([mat[0]]), np.array([[1]])
        return T, V

    # Initialize the matrices for Lanczos vectors and the tridiagonal matrix.
    V = np.zeros((m, n))
    T = np.zeros((m, m))
    V[0, :] = v0

    # Step 2.1 - 2.3: Compute the first Lanczos vector and the first diagonal element.
    w = np.sum([np.dot(col, np.dot(np.conj(col.T), V[0, :])) for col in mat], 0)
    alfa = np.dot(w, V[0, :])
    w = w - alfa * V[0, :]
    T[0, 0] = alfa

    # Iterate to compute the remaining Lanczos vectors and tridiagonal elements.
    for j in range(1, m - 1):
        beta = np.sqrt(np.dot(w, w))
        V[j, :] = w / beta

        # Reorthogonalize the current vector against all previous vectors.
        for i in range(j - 1):
            V[j, :] = V[j, :] - np.dot(np.conj(V[j, :]), V[i, :]) * V[i, :]
        V[j, :] = V[j, :] / np.linalg.norm(V[j, :])

        # Compute the next Lanczos vector and tridiagonal elements.
        w = np.sum([np.dot(col, np.dot(np.conj(col.T), V[j, :])) for col in mat], 0)
        alfa = np.dot(w, V[j, :])
        w = w - alfa * V[j, :] - beta * V[j - 1, :]

        T[j, j] = alfa
        T[j - 1, j] = beta
        T[j, j - 1] = beta

    return T, V
