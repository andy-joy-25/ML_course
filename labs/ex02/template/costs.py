# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    # ***************************************************
    return (0.5*(1/y.shape[0])*np.matmul((y - np.matmul(tx,w)).T,(y - np.matmul(tx,w)))).item() # MSE
    #return (1/y.shape[0])*np.sum(np.abs(y - np.matmul(tx,w))) #MAE
