import torch

def n_dataloader(dataloader):
    '''
    Return the number of instances in the dataloader
    '''
    if not isinstance(dataloader, torch.utils.data.DataLoader):
        raise TypeError("dataloader needs to be an instance of torch.utils.data.DataLoader")
    
    for i, (input_, _) in enumerate(dataloader):
        if i > 0 and i < len(dataloader) - 1:
            continue
        if i == 0:
            n_batch = input_.size(0)
            totlen = n_batch * len(dataloader)
        else:
            totlen -= (n_batch - input_.size(0))

    return totlen