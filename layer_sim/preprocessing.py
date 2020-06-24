import torch
import numpy as np

def svd_decomposition(tensor):
    if len(tensor.shape) > 2:
        raise RuntimeError("Only 2dim tensors can be decomposed with SVD")
    if tensor.shape[0] < tensor.shape[1]:
        raise RuntimeError(f"SVD requires the number of datapoints ({tensor.shape[0]}) to be larger than the number of neurons ({tensor.shape[1]}).")
    
    vars_mean = tensor.mean(0)
    tensor -= vars_mean
    
    u, s, v = tensor.svd()
    tensor += vars_mean
    return u, s, v

def svd_reduction(tensor, var_fract_kept=.99):
    '''
    Performs an svd reduction of the given tensor keeping the singular values accounting for var_fract_kept% of the total variance (default .99)
    '''
    u, s, _ = svd_decomposition(tensor)
    var = s*s
    # cumulative proportion of variance explained for each singular value
    var_cumulative_prop = var.cumsum(0) / var.sum()

    # index in s corresponding to the last eigenvalue accounting for the var_fract_kept% of variance
    max_index_keep = len(np.where(var_cumulative_prop <= var_fract_kept)[0]) + 1

    # return the reduction of the layer
    return u[:, :max_index_keep] @ torch.diag(s[:max_index_keep])
    #return np.dot(u[:,:max_index_keep], np.diag(s[:max_index_keep]))
