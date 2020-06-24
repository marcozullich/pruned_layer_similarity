import torch
import numpy as np
import re

from collections import OrderedDict as odict

def _build_pruning_mask(layers_dict, pruning_factor, previous_mask=None):
    '''
    Builds a pruning mask (state_dict's style) for the given layers and pruning_factor.
    Use previous_mask for iterative pruning.
    '''
    # flatten and concatenate the layers in a single distribution in order to obtain the cutoff for pruning
    # using numpy since np.where is much faster than torch.where (at least on cpu)
    layers_flat = np.concatenate([np.abs(layers_dict).flatten() for layer in layers_dict.values()])

    # repeat for mask
    if previous_mask is not None:
        prev_mask_flat = np.concatenate([np.abs(m).flatten() for m in previous_mask.values()])
        # delete from prev_mask_flat elements already pruned in the past
        layers_flat = np.delete(layers_flat, np.where(prev_mask_flat == 0))
    
    layers_flat = np.sort(layers_flat)

    cutoff_position = int(np.floor(pruning_factor) * layers_flat.shape[0])
    cutoff_value = layers_flat[cutoff_position]

    # build a submask containing 0s and 1s for all layers involved in pruning
    submask = {key: torch.from_numpy(np.where(np.abs(val.to("cpu")) > cutoff_value, 1, 0)) for key, val in layers_dict.items()}

    return submask


def lf_mask_global(net, pruning_factor=.2, previous_mask=None, layer_ids_to_prune=["conv"]):
    '''Builds a pruning mask based upon Iterative Magnitude Pruning.
    The mask is a dict mimicking the net.state_dict() with its values being
    binary structures with the same shape as the corresponding net's parameters.
    Elements with a 0 identify parameters to prune, elements with a 1 survive
    the pruning.

    Parameters:
    net -- the network to build the mask upon
    pruning_factor -- the proportion of parameters getting pruned (default 0.2)
    previous_mask -- the previous pruning mask, to use for iterative pruning in
    order to identify previously pruned parameters (default None)
    layer_ids_to_prune -- list or tuple of (sub)strings identifying the layer
    names to be pruned. Useful to preserve some layers, e.g., batchnorm, from
    being touched by pruning. Can also contain regex patterns (default ["conv"])


    Returns:
    an OrderedDict whose keys correspond to the state_dict's keys of the layers
    to be pruned; the values are binary tensors where 0 identify a parameter
    to be pruned, a 1 a parameter surviving the pruning.
    '''

    if isinstance(layer_ids_to_prune, str):
        layer_ids_to_prune = [layer_ids_to_prune]
    
    state_dict = net.state_dict()

    # isolate the parameters to be pruned in another dict
    params_to_prune = odict({name:w for name,w in state_dict.items() if any(re.match(t, name) for t in layer_ids_to_prune)})

    submask = _build_pruning_mask(params_to_prune, pruning_factor, previous_mask=previous_mask)

    return submask

def apply_mask(net, mask, gradient=False, sequential=False):
    '''Applies (in-place) a mask to the given model (net), zeroing out the
    parameters corresponding to a 0 in the mask.
    If the option gradient is specified, zeroes out also the gradient for the
    corresponding parameter (to be used during training).
    The option sequential identifies whether the net's layers are defined via
    the torch.nn.Sequential structure (distinction required due to a different
    handling of the gradient operations).
    '''

    device = next(net.parameters()).device
    state_dict = net.state_dict()

    for name, m in mask.items():
        if gradient:
            if sequential:
                # handling of gradients for sequential models is particularly painful
                # due to the difficulty in accessing the parameters' grads
                modules_names = name.split(".")

                for mod_name, par in eval("net.%s[%d].named_parameters()" % (modules_names[0], int(modules_names[1]))):
                    if mod_name == modules_names[2]:
                        par.grad *= m.to(device)
                        break
            else:
                # this workaround is required due to the absence of gradients
                # in the state_dict
                exec("model.%s.grad *= m.to(device)" % name)
        else:
            state_dict[name] *= m.to(device)