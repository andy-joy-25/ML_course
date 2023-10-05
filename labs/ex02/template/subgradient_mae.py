import numpy as np


def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute subgradient gradient vector for MAE
    # ***************************************************
    r = np.random.randint(-1,2) #Return a random no. from [-1,1]
    del_h = y - np.matmul(tx,w)
    del_h[del_h < 0] = -1
    del_h[del_h == 0] = r
    del_h[del_h > 0] = 1
    return -(1/y.shape[0])*np.matmul(tx.T,del_h)

