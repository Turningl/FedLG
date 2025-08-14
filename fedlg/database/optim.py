# -*- coding: utf-8 -*-
# @Author : liang
# @File : optim.py


import ast
import numpy as np
import configparser
import hyperopt as hy


# bayesian optimization process
def bayesian_optimization(init, model_prediction, max_step, random_seed=None):
    """
    Perform Bayesian optimization using Tree-structured Parzen Estimator (TPE).

    This method optimizes the hyperparameters defined in the `pbounds` dictionary
    using the TPE algorithm from the Hyperopt library.

    Returns:
        dict: The best hyperparameters found by the optimization process.
    """
    # Define the parameter bounds for Bayesian optimization
    pbounds = {
        'lanczos_iter': hy_parameter_setting(label='lanczos_iter'),  # Set Lanczos iteration number
    }

    # Print parameter bounds (optional)
    # print("Parameter bounds:")
    # for param, value in pbounds.items():
    #     print(f"{param}: {value}")

    print('Enhancing Model Performance Using Bayesian Optimization with TPE')

    # Set the Tree-structured Parzen Estimator (TPE) algorithm
    algo = hy.partial(hy.tpe.suggest, n_startup_jobs=init)

    rstate = None
    # Set the random state for reproducibility
    if random_seed:
        rstate = np.random.RandomState(random_seed)

    # Initialize the Trials object to store the optimization history
    trials = hy.Trials()

    # Perform Bayesian optimization
    best = hy.fmin(
        fn=model_prediction,  # Objective function to minimize
        space=pbounds,  # Parameter space to search
        algo=algo,  # Optimization algorithm
        max_evals=max_step,  # Maximum number of evaluations
        trials=trials,  # Store optimization history
        rstate=rstate  # Random state for reproducibility
    )

    return best


def hy_parameter_setting(label):
    """
    Set hyperparameters for Bayesian optimization using the provided label.

    Args:
        label (str): Label for the hyperparameter to be set.

    Returns:
        hyperopt.hp: Hyperparameter object for Bayesian optimization.
    """
    # Read the configuration file
    config = configparser.RawConfigParser()
    config.read('./param/{}_range.in'.format('optimization'))

    try:
        # Get the range for the specified label
        val = ast.literal_eval(config.get('range', label))
    except Exception as e:

        raise ImportError(f"Error reading config for {label}: {e}")

    # Set hyperopt parameter
    if type(val) is list:
        if label == 'lanczos_iter':
            # Define a choice of values for Lanczos iteration
            return hy.hp.choice(label, np.arange(val[0], val[1], 2).tolist())
        else:
            raise ImportError('Unsupported parameter. Check configuration or label name.')
    else:
        raise ImportError('Invalid data format in config file. Expected a list of two values.')