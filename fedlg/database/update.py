# -*- coding: utf-8 -*-
# @Author : liang
# @File : update.py


import ast
import copy
import torch
import random
import numpy as np
import scipy as sp
import torch.nn as nn
from fedlg.utils.arnoldi import Lanczos


def eigen_by_lanczos(mat, lanczos_iter, proj_dims, device):
    """
    Compute the leading eigen-vector(s) of a (large, sparse) symmetric matrix
    via the Lanczos algorithm.

    Parameters
    ----------
    mat : ndarray or sparse matrix, shape (n, n)
        Real symmetric (or Hermitian) matrix whose dominant eigenspace is sought.
    lanczos_iter : int
        Number of Lanczos iterations (i.e. Krylov subspace dimension).
    proj_dims : int
        How many of the top eigenvectors to return.
    device : torch.device
        Target device for the returned tensor.

    Returns
    -------
    torch.Tensor
        A tensor of shape `(n, proj_dims)` if `proj_dims > 1`, or shape `(n,)`
        if `proj_dims == 1`, residing on `device` and having dtype `torch.float32`.
    """
    Tri_Mat, Orth_Mat = Lanczos(mat, lanczos_iter)  # getting a tridiagonal matrix T and an orthogonal matrix V

    # T_evals_, T_evecs_ = np.linalg.eig(Tri_Mat)  # calculating the eigenvalues and eigenvectors of a tridiagonal matrix
    T_evals, T_evecs = sp.sparse.linalg.eigsh(Tri_Mat, k=2, which='LM')

    idx = T_evals.argsort()[
          -1: -(proj_dims + 1): -1]  # finding the index of the largest element in the eigenvalue array T evals

    proj_eigenvecs = np.dot(Orth_Mat.T,
                            T_evecs[:,idx])  # the eigenvector corresponding to the maximum eigenvalue is projected into the new eigenspace

    if proj_eigenvecs.size >= 2:
        proj_eigenvecs = torch.from_numpy(proj_eigenvecs).to(torch.float32).to(device).squeeze()
    else:
        proj_eigenvecs = torch.from_numpy(proj_eigenvecs).to(torch.float32).to(device).squeeze(0)

    return proj_eigenvecs


def fetch_model_update(model, state_dict_key, params, updates, model_states, proj_dims, device, means, shape_vars):
    """
    Reconstruct an updated model by projecting gradient/parameter updates onto
    the leading eigenspaces of their respective layer-wise covariance matrices.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network whose parameters will be overwritten.
    state_dict_key : list[str]
        Ordered list of keys matching `model.state_dict()`, used to reconstruct
        the state dictionary after projection.
    params : dict
        Hyper-parameters; must contain at least
        - 'lanczos_iter' : int  – Krylov dimension for Lanczos.
    updates : list[torch.Tensor]
        Flattened gradient or parameter delta for every layer.
    model_states : list[np.ndarray]
        Layer-wise covariance-like matrices (or their low-rank factors) on which
        the Lanczos iterations are performed.
    proj_dims : int
        Number of top eigenvectors to retain per layer.
    device : torch.device
        Target device for all intermediate and final tensors.
    means : list[torch.Tensor]
        Per-layer mean vectors used for centering before projection.
    shape_vars : list[torch.Size]
        Original shapes of each parameter tensor; used to reshape the projected
        update back to the layer’s dimensions.

    Returns
    -------
    tuple (model, proj_updates)
        model : torch.nn.Module
            The network with updated parameters (set to evaluation mode).
        proj_updates : list[torch.Tensor]
            The projected update tensors, one per layer, already reshaped.
    """
    # Flatten the updates
    updates = [update.flatten() for update in updates]
    proj_updates = [0] * len(model_states)

    # Execute Lanczos algorithm and project updates
    for i in range(len(model_states)):
        proj_eigenvecs = eigen_by_lanczos(mat=model_states[i].T,
                                            lanczos_iter=params['lanczos_iter'],
                                            proj_dims=proj_dims,
                                            device=device)

        proj_updates[i] = (torch.mul(proj_eigenvecs, torch.dot(proj_eigenvecs.T, (updates[i] - means[i]))) +
                           means[i]).reshape(shape_vars[i])

    # Load the updated model state dictionary
    model = model.eval()
    model.load_state_dict(dict(zip(state_dict_key, proj_updates)))

    return model, proj_updates
