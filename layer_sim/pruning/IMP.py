import torch
import os
from ..train import train_net, test_net, accuracy_at_k
from .LF_mask import lf_mask_global, apply_mask, mask_prop_params

def imp_lrr(net, epochs, criterion, optimizer, trainloader, imp_iterations, device=None, pruning_factor=.2, iteration_restore=0, mask=None, performance=accuracy_at_k, testloader=None, save_path=None, save_pattern="IMP_checkpoint_{}.pt", layer_ids_to_prune=["conv"], **kwargs):
    '''Runs IMP with Learning Rate Rewind for imp_iterations with a pruning factor of pruning_factor.

    Parameters:
    net -- a fully-trained neural network to be pruned. Can also be a partially
        pruned NN: in case, set the iteration_restore and the mask accordingly
    epochs -- epochs for each training of IMP
    criterion -- the loss function
    optimizer -- the optimizer to use (reset after each iteration)
    device -- torch device where net and trainloader are stored (default: "cuda:0" if cuda is available, otherwise "cpu")
    pruning_factor -- percentage of parameters that will be pruned for each iteration (default 0.2)
    iteration_restore -- iteration to resume for current training.
        Set to 0 if no previous iteration was completed (default 0)
    mask -- pruning mask to apply for current iterations - use if resuming
        a partially completed IMP session (default None)
    performace -- performance function to asses the model (default accuracy@1)
    testloader -- used to test the model after each iteration of IMP (default None)
    save_path -- folder where the IMP checkpoints will be stored. If None, no
        checkpoint will be saved (default None)
    save_pattern -- a pattern w/ replacement field used for checkpoint saving
        purposes. Replacement field gets replaced by current IMP iteration
        (default IMP_checkpoint_{}.torch)
    layer_ids_to_prune -- list or tuple of (sub)strings identifying the layer
        names to be pruned. Useful to preserve some layers, e.g., batchnorm, from
        being touched by pruning. Can also contain regex patterns (default ["conv"])
    **kwargs -- additional parameters (like annealing values and sequential flag) to be passed on to the training routine


    '''

    if iteration_restore > 0 and mask is None:
        raise AttributeError("If restoring IMP from an existing checkpoint, you need to provide also the previous mask.")

    if save_path is not None:
        if os.path.isfile(save_path):
            raise TypeError(f"The given save_path ({save_path} is a file. Needs to be a dir)")
        os.makedirs(save_path, exist_ok=True)

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    for ite in range(iteration_restore, imp_iterations):
        print(f"=====> Iteration of IMP: {ite+1}/{imp_iterations}")

        # 1. Determine current mask
        mask = lf_mask_global(net, pruning_factor=pruning_factor, previous_mask=mask, layer_ids_to_prune=layer_ids_to_prune)
        print(f"Proportion of parameters in mask {mask_prop_params(mask, net)}")
        mask = {k: m.to(device).float() for k, m in mask.items()}


        # 2. Apply the mask (= zero-out low magnitude parameters)
        apply_mask(net, mask)

        # 3. Train the net passing the mask to zero-out gradient of pruned params
        train_loss, train_perf = train_net(net, epochs, criterion, optimizer, trainloader, device, performance, mask=mask, **kwargs)

        # 4. Test (if requested)
        test_loss = test_perf = None
        if testloader is not None:
            test_loss, test_perf = test_net(net, testloader, criterion, device, performance)
        
        # save state_dict with mask & stats (if requested)
        if save_path is not None:
            iteration_dict = {
                "train_loss": train_loss,
                "train_perf": train_perf,
                "test_loss": test_loss,
                "test_perf": test_perf,
                "mask": mask,
                "parameters": net.state_dict()
            }

            save_file = os.path.join(save_path, save_pattern.format(ite))
            torch.save(iteration_dict, save_file)

            print(f"==> Checkpoint for iteration {ite} dumped to {save_file}")

        

        




