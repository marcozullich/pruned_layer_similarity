import torch

def _centering(kernel):
    return kernel - kernel.mean(0, keepdims=True) - kernel.mean(1, keepdims=True) + kernel.mean()

def cka(kernel1, kernel2):
    '''
    Compute CKA between two given kernel matrices
    '''
    kernel1 = _centering(kernel1)
    kernel2 = _centering(kernel2)
    return (kernel1@kernel2).trace() / (kernel1.norm()*kernel2.norm())

def nbs(kernel1, kernel2):
    kernel1 = _centering(kernel1)
    kernel2 = _centering(kernel2)
    s, _ = (kernel1 @ kernel2).eig()
    s = s[:,0].clamp(0.).sqrt()
    return s / (kernel1.trace() * kernel2.trace()).sqrt()
