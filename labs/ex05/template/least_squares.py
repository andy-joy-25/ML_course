# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    w = np.linalg.solve(np.matmul(tx.T,tx),np.matmul(tx.T,y))
    
    mse = (0.5*(1/y.shape[0])*np.matmul((y - np.matmul(tx,w)).T,(y - np.matmul(tx,w)))).item()
    
    return w,mse