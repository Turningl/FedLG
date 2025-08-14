# -*- coding: utf-8 -*-
# @Author : liang
# @File : fedlg_.py


import torch
import numpy as np
import scipy as sp
from fedlg.utils.nnutils import add_weights
from fedlg.utils.arnoldi import Lanczos


class FedLG:
    """
    Attributes:
        proj_dims (int): Number of projection dimensions.
        lanczos_iter (int): Number of Lanczos iterations.
        device (str): Device to use for computations (e.g., 'cuda' or 'cpu').
        comm_optimization (bool): Whether to use communication optimization.

        __num_open_access_databases (int): Number of open-access databases.
        __open_access_model_state (list): Internal storage for open-access model states.
        __open_access_eps (list): Internal storage for open-access epsilons.

        __num_privacy_institution_databases (int): Number of privacy-institution databases.
        __private_institutional_model_state (list): Internal storage for private-institutional model states.
        __private_institutional_eps (list): Internal storage for private-institutional epsilons.

        num_vars (int): Number of model variables (param).
        shape_vars (list): Shapes of the model variables.
        open_access_model_states (list): Open-access model states for communication optimization.
        means (list): Means for communication optimization.
    """

    def __init__(self,
                 proj_dims,
                 lanczos_iter,
                 device='cuda',
                 comm_optimization=None
                 ):
        """
        Args:
            proj_dims (int): Number of projection dimensions.
            lanczos_iter (int): Number of Lanczos iterations.
            device (str, optional): Device to use for computations. Defaults to 'cuda'.
            comm_optimization (bool, optional): Whether to use communication optimization. Defaults to None.
        """

        self.proj_dims = proj_dims
        self.lanczos_iter = lanczos_iter
        self.device = device

        self.__num_open_access_databases = 0
        self.__open_access_model_state = []
        self.__open_access_eps = []

        self.__num_privacy_institution_databases = 0
        self.__private_institutional_model_state = []
        self.__private_institutional_eps = []

        self.num_vars = None
        self.shape_vars = None

        self.comm_optimization = comm_optimization
        self.open_access_model_states = None
        self.means = None

    def aggregate(self, eps, model_state, is_open_access):
        """
        Aggregate a new model state and epsilon into the internal storage.

        Args:
            eps (float): Epsilon value for the model state.
            model_state (list of torch.Tensor): Model state to be aggregated.
            is_open_access (bool): Whether the model state is from an open-access database.
        """

        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state]

        self.num_vars = len(model_state)
        update_model_state = [state.flatten() for state in model_state]

        if is_open_access:
            self.__num_open_access_databases += 1
            self.__open_access_eps.append(eps)
            self.__open_access_model_state = add_weights(self.num_vars, update_model_state, self.__open_access_model_state)

        else:
            self.__num_privacy_institution_databases += 1
            self.__private_institutional_eps.append(eps)
            self.__private_institutional_model_state = add_weights(self.num_vars, update_model_state, self.__private_institutional_model_state)

    def __standardize(self, M):
        """
        Standardize the given matrix by subtracting the mean.

        Args:
            M (torch.Tensor): Matrix to be standardized.

        Returns:
            torch.Tensor: Standardized matrix.
            torch.Tensor: Mean of the matrix.
        """

        n, m = M.shape
        if m == 1:
            return M, torch.zeros(n, device=self.device)
        mean = torch.mul(M, torch.ones((m, 1), dtype=torch.float32, device=self.device)) / m

        return M - mean, mean.flatten()

    def __eigen_by_lanczos(self, mat):
        """
        Compute the largest eigenvalues and eigenvectors using Lanczos algorithm.

        Args:
            mat (torch.Tensor): Matrix to compute eigenvalues and eigenvectors.

        Returns:
            torch.Tensor: Eigenvectors corresponding to the largest eigenvalues.
        """

        Tri_Mat, Orth_Mat = Lanczos(mat, self.lanczos_iter)  # getting a tridiagonal matrix T and an orthogonal matrix V

        # T_evals_, T_evecs_ = np.linalg.eig(T)  # calculating the eigenvalues and eigenvectors of a tridiagonal matrix
        T_evals, T_evecs = sp.sparse.linalg.eigsh(Tri_Mat, k=2, which='LM')

        idx = T_evals.argsort()[-1: -(self.proj_dims + 1): -1]  # finding the index of the largest element in the eigenvalue array T evals

        proj_eigenvecs = np.dot(Orth_Mat.T, T_evecs[:, idx])  # the eigenvector corresponding to the maximum eigenvalue is projected into the new eigenspace

        if proj_eigenvecs.size >= 2:
            proj_eigenvecs = torch.from_numpy(proj_eigenvecs).to(torch.float32).to(self.device).squeeze()
        else:
            proj_eigenvecs = torch.from_numpy(proj_eigenvecs).to(torch.float32).to(self.device).squeeze(0)

        return proj_eigenvecs

    def __lanczos_graph_proj(self):
        """
        Perform Lanczos-based graph projection.

        Returns:
            list of torch.Tensor: Projected model state.
        """

        if len(self.__private_institutional_model_state):
            # Compute the weights for the private institutional model states based on their epsilons
            private_institutional_weights = (
                    torch.Tensor(self.__private_institutional_eps) / sum(self.__private_institutional_eps)
            ).view(self.__num_privacy_institution_databases, 1).to(self.device)

            # Compute the weights for the open-access model states based on their epsilons
            open_access_weights = (
                    torch.Tensor(self.__open_access_eps) / sum(self.__open_access_eps)
            ).view(self.__num_open_access_databases, 1).to(self.device)

            # Compute the weighted average of the private institutional model states
            mean_priv_model_state = [
                torch.sum(self.__private_institutional_model_state[i].to(self.device) * private_institutional_weights,
                          0) /
                torch.sum(private_institutional_weights)
                for i in range(self.num_vars)
            ]

            # Compute the weighted average of the open-access model states
            mean_pub_model_state = [
                torch.sum(self.__open_access_model_state[i].to(self.device) * open_access_weights, 0) /
                torch.sum(open_access_weights)
                for i in range(self.num_vars)
            ]

            # Initialize lists to store the projected private institutional model states and the final mean model states
            mean_proj_priv_model_state = [0] * self.num_vars
            mean_model_state = [0] * self.num_vars

            # Process each model variable
            for i in range(self.num_vars):
                # Standardize the open-access model state for the current variable
                open_access_model_state, mean = self.__standardize(self.__open_access_model_state[i].T)

                # Compute the eigenvectors using the Lanczos algorithm for projection
                proj_eigenvecs = self.__eigen_by_lanczos(open_access_model_state.T)

                # Project the private institutional model state onto the space spanned by the Lanczos eigenvectors
                mean_proj_priv_model_state[i] = (
                        torch.mul(proj_eigenvecs, torch.dot(proj_eigenvecs.T, (mean_priv_model_state[i] - mean))) + mean
                )

                # Compute the final mean model state by combining the projected private institutional model state
                # and the open-access model state, weighted by their respective epsilons
                mean_model_state[i] = (
                        (mean_proj_priv_model_state[i] * sum(self.__private_institutional_eps) +
                         mean_pub_model_state[i] * sum(self.__open_access_eps)) /
                        sum(self.__private_institutional_eps + self.__open_access_eps)
                ).reshape(self.shape_vars[i])

            # Return the computed mean model states
            return mean_model_state

    def __lanczos_graph_proj_communication_optimization(self, warmup):
        """
        Perform Lanczos-based graph projection with communication optimization.

        Args:
            warmup (bool): Whether this is the warm-up phase.

        Returns:
            list of torch.Tensor: Projected model state.
        """

        if len(self.__private_institutional_model_state):
            # Compute the normalized weights for the private institutional model states based on their epsilons.
            private_institutional_weights = (
                    torch.Tensor(self.__private_institutional_eps) / sum(self.__private_institutional_eps)
            ).view(self.__num_privacy_institution_databases, 1).to(self.device)

            # Compute the normalized weights for the open-access model states based on their epsilons.
            open_access_weights = (
                    torch.Tensor(self.__open_access_eps) / sum(self.__open_access_eps)
            ).view(self.__num_open_access_databases, 1).to(self.device)

            # Compute the weighted average of the private institutional model states.
            mean_priv_model_state = [
                torch.sum(self.__private_institutional_model_state[i].to(self.device) * private_institutional_weights,
                          0) / torch.sum(private_institutional_weights)
                for i in range(self.num_vars)
            ]

            # Compute the weighted average of the open-access model states.
            mean_pub_model_state = [
                torch.sum(self.__open_access_model_state[i].to(self.device) * open_access_weights, 0) /
                torch.sum(open_access_weights)
                for i in range(self.num_vars)
            ]

            # Initialize lists to store the projected private institutional model states and the final mean model states.
            mean_proj_priv_model_state = [0] * self.num_vars
            mean_model_state = [0] * self.num_vars

            # Initialize lists to store the standardized open-access model states and their means.
            open_access_model_states = []
            means = []

            if warmup:
                # If this is the warm-up phase, perform the Lanczos projection and compute the final mean model state.
                for i in range(self.num_vars):
                    # Standardize the open-access model state for the current variable.
                    open_access_model_state, mean = self.__standardize(self.__open_access_model_state[i].T)

                    # Compute the eigenvectors using the Lanczos algorithm for projection.
                    proj_eigenvecs = self.__eigen_by_lanczos(open_access_model_state.T)

                    # Project the private institutional model state onto the space spanned by the Lanczos eigenvectors.
                    mean_proj_priv_model_state[i] = torch.mul(proj_eigenvecs, torch.dot(proj_eigenvecs.T, (
                                mean_priv_model_state[i] - mean))) + mean

                    # Compute the final mean model state by combining the projected private institutional model state
                    # and the open-access model state, weighted by their respective epsilons.
                    mean_model_state[i] = (
                            (mean_proj_priv_model_state[i] * sum(self.__private_institutional_eps) +
                             mean_pub_model_state[i] * sum(self.__open_access_eps)) /
                            sum(self.__private_institutional_eps + self.__open_access_eps)
                    ).reshape(self.shape_vars[i])

                    # Store the standardized open-access model state and its mean.
                    open_access_model_states.append(open_access_model_state)
                    means.append(mean)
            else:
                # If this is not the warm-up phase, directly compute the final mean model state.
                for i in range(self.num_vars):
                    # Compute the final mean model state by combining the projected private institutional model state
                    # and the open-access model state, weighted by their respective epsilons.
                    mean_model_state[i] = (
                            (mean_proj_priv_model_state[i] * sum(self.__private_institutional_eps) +
                             mean_pub_model_state[i] * sum(self.__open_access_eps)) /
                            sum(self.__private_institutional_eps + self.__open_access_eps)
                    ).reshape(self.shape_vars[i])

                    # Standardize the open-access model state for the current variable.
                    open_access_model_state, mean = self.__standardize(self.__open_access_model_state[i].T)

                    # Store the standardized open-access model state and its mean.
                    open_access_model_states.append(open_access_model_state)
                    means.append(mean)

            # Store the standardized open-access model states and their means for future use.
            self.open_access_model_states = open_access_model_states
            self.means = means

            # Return the computed mean model states.
            return mean_model_state

    def average(self):
        """
        Compute the aggregated model updates using Lanczos-based graph projection.

        This method combines private and open-access model states to produce the final mean model state.
        It supports communication optimization to reduce the communication overhead.

        Returns:
            list of torch.Tensor: The aggregated mean model state.
        """
        # mean_updates = None
        if self.comm_optimization:
            # If communication optimization is enabled, use the optimized projection method.
            # The warmup flag is set based on whether open_access_model_states is None.
            mean_updates = self.__lanczos_graph_proj_communication_optimization(
                warmup=(self.open_access_model_states is None))
        else:
            # If communication optimization is disabled, use the standard projection method.
            mean_updates = self.__lanczos_graph_proj()

        # Reset the counters and lists for the next aggregation round.
        self.__num_open_access_databases = 0
        self.__num_privacy_institution_databases = 0

        self.__open_access_model_state = []
        self.__private_institutional_model_state = []

        self.__open_access_eps = []
        self.__private_institutional_eps = []

        return mean_updates