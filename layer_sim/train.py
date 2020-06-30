import torch

from .pruning.LF_mask import apply_mask

def accuracy_at_k(output, target, k=1):
    '''
    Returns accuracy@k for the current output of the net vs target (ground truth)
    '''
    _, pred = output.topk(k)
    pred = pred.T
    ground_trunth = target.long().view(1,-1).expand_as(pred)
    correct = pred.eq(ground_trunth)

    return correct.int().sum().item()/len(target)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def anneal_lr(optimizer, annealing_factor):
    for param_group in optimizer.param_groups:
        param_group["lr"] /= annealing_factor
        print(f'LR annealed: previous {param_group["lr"]*annealing_factor}, current {param_group["lr"]}')
    
    return optimizer

def train_net(net, epochs, criterion, optimizer, trainloader, device=None, performance=accuracy_at_k, epochs_annealing=None, lr_annealing_factor=None,  resume_checkpoint=None, save_checkpoint=None, mask=None):
    '''Trains a specified network

    Parameters:
    net -- the neural network to train
    epochs -- the number of epochs to train for
    criterion -- the loss function to use
    optimizer -- the optimizer to use
    device -- torch device where net and trainloader are stored (default: "cuda:0" if cuda is available, otherwise "cpu")
    performance -- the performance function to assess the model during training (default: accuracy@1)
    epochs_annealing -- a list - the epochs after which the learning rate gets annealed (default: None)
    lr_annealing_factor -- the number through which the learning rate gets annealed after each epochs_annealing (default: None)
    resume_checkpoint -- a torch save containing the minimal elements for resuming training at an intermediate epoch (default: None)
    save_checkpoint -- path where the training checkpoint will be stored each epoch - note: will be overwritten each successive epoch (default: None)
    mask -- a torch structure containing the pruning mask - instrumental to operate pruning like IMP
    '''
    if (epochs_annealing is None) != (lr_annealing_factor is None):
        raise AttributeError(f"epochs_annealing and lr_annealing_factor must be either both None or they must have a value. Found {epochs_annealing} and {lr_annealing_factor}")
    
    if epochs_annealing is None:
        epochs_annealing = []
    else:
        if isinstance(epochs_annealing, int):
            epochs_annealing = [epochs_annealing]
        elif not isinstance(epochs_annealing, (list, tuple)):
            raise TypeError(f"Unsupported type for epochs_annealing. Supported {(int, list, tuple)}. Found {type(epochs_annealing)}")

    
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    net.train() #activate training mode in network

    start_epoch = 0
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        del checkpoint

    for epoch in range(start_epoch, epochs):
        losses = AverageMeter()
        perf = AverageMeter()

        for input_, target in trainloader:
            input_=input_.to(device)
            target=target.to(device)

            output = net(input_)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()

            if mask is not None: #apply mask to zero out gradient on masked-out elements
                apply_mask(net, mask, gradient=True)

            # gradient clipping to avoid explosion
            torch.nn.utils.clip_grad.clip_grad_norm_(net.parameters(), 1)

            optimizer.step()

            output = output.float()
            loss = loss.float()

            pref_val = performance(output.data, target)
            losses.update(loss.item(), input_.size(0))
            perf.update(pref_val, input_.size(0))

        print(f"===> Epoch {epoch + 1}/{epochs} ### Loss {losses.avg} ### Performance {perf.avg}")
        
        if epoch in epochs_annealing:
            optimizer = anneal_lr(optimizer, lr_annealing_factor)
        
        if save_checkpoint is not None:
            checkpoint_dict = {
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint_dict, save_checkpoint)
            print(f"Saved model checkpoint to {checkpoint_dict}")
    return losses.avg, perf.avg
    
def test_net(net, testloader, criterion, device=None, performance=accuracy_at_k):
    '''Test the network on the specified testloader

    Parameters:
    net -- the neural network to train
    criterion -- the loss function to use
    optimizer -- the optimizer to use
    device -- torch device where net and trainloader are stored (default: "cuda:0" if cuda is available, otherwise "cpu")
    performance -- the performance function to assess the model (default: accuracy@1)
    
    Returns:
    average loss for the test set
    average performance for the training set
    '''
    losses = AverageMeter()
    perf = AverageMeter()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    net = net.to(device)
    net.eval()

    for input_, target in testloader:
        input_ = input_.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = net(input_)
            loss = criterion(output, target)
        
        perf_val = performance(output.data, target)

        losses.update(loss, input_.size(0))
        perf.update(perf_val, input_.size(0))

    print(f"===> TEST ### Loss {losses.avg} ### Performance {perf.avg}")

    return losses.avg, perf.avg



        





