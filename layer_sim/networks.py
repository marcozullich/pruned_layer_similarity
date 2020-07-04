import torch
from torch import nn
from .utils import n_dataloader

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNN(nn.Module):
    '''
    A generic class for a fully-convolutional CNN composed of a features
    subnetwork (convolutions, max pooling etc) and a classifier subnetwork
    (fully-connected layers + output) interspersed with a average pooling
    module which is needed in order to flatten the convolutional layers' 
    representation so it can be fed to the f-c layers.
    An instance of this class is also able to produce an activation matrix
    of its layers, via the method extract_network_representation.
    '''
    def __init__(self, features, flat, classifier):
        '''
        Parameters:
        features -- a nn.Sequential object composing the features module
        classifier -- a nn.Sequential object composing the classifier module
        '''
        super(CNN, self).__init__()
        self.features = features
        self.flat = flat
        self.classifier = classifier

    def forward(self, x):
        x = self.features(x)
        x = self.flat(x)
        x = self.classifier(x)
        return x
    
    def _hook_layers(self, sequential, x, layer_types_to_hook):
        '''
        retains in a list the outputs of the given types of layers in response
        to x.

        Parameters:
        sequential -- a nn.Sequential object containing (one or more) nn.Modules
            to evaluate
        x -- a tensor representing an input to the net
        layer_types_to_hook -- types of layers whose activations will be stored
            in a list and returned

        Returns:
        the evaluation of x on the whole sequential model
        a list containing the outputs of the selected layers (modules)
        '''
        out = []
        for layer in sequential:
            x = layer(x)
            if isinstance(layer, layer_types_to_hook):
                out.append(x)
        
        return x, out
    
    def forward_with_hooks(self, x, layer_types_to_hook=(nn.ReLU, nn.MaxPool2d)):
        out = []
        for seq in (self.features, self.flat, self.classifier):
            x, out_seq = self._hook_layers(seq, x, layer_types_to_hook=layer_types_to_hook)
            out.extend(out_seq)

        # output layer is a special case - a nn.Linear without activation fct.
        if not isinstance(self.classifier[-1], layer_types_to_hook):
            out.append(x)

        return out
      
    
    def extract_network_representation(self, dataloader, limit_datapoints=5000, layer_types_to_hook=(nn.ReLU, nn.MaxPool2d), device=None):
        '''
        Extract the activation matrices of the layers of the network (for the 
        desired layer_types) given a dataloader. The number of datapoints of
        the representation can also be specified using limit_datapoints (defaults
        to 5000)
        '''
        self.eval()

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.to(device)
        
        evaluated_datapoints = 0
        repr_datapoints = n_dataloader(dataloader)

        if limit_datapoints is None or limit_datapoints == 0:
            limit_datapoints = repr_datapoints
        elif repr_datapoints < limit_datapoints:
            raise RuntimeError(f"The dataloader has not enough datapoints to compute the representations. Required {limit_datapoints}; found {repr_datapoints}")

        for i, (input_, _) in enumerate(dataloader):                
            if input_.size(0) + evaluated_datapoints > limit_datapoints:
                n_batch = limit_datapoints - evaluated_datapoints
            else:
                n_batch = input_.size(0)
            input_ = input_[:n_batch].to(device)
            
            repr_batch = self.forward_with_hooks(input_, layer_types_to_hook=layer_types_to_hook)

            if i == 0:
                representation = [torch.empty([limit_datapoints] + list(r.shape[1:])) for r in repr_batch]

            for j in range(len(representation)):
                representation[j][evaluated_datapoints : (evaluated_datapoints + n_batch)] = repr_batch[j]

            evaluated_datapoints += n_batch

            if evaluated_datapoints >= limit_datapoints:
                break
    
        return representation
        





class VGG_SVHN(CNN):
    
    def __init__(self, num_classes=10):
        # helper function to build convolutional layers with BN and ReLU
        def conv2d_svhn(in_channels, out_channels):
            return [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
        
        feats = []
        # build feature extractor
        for i in range(1,5):
            if i == 1:
                # first conv layer is fed images (-> in_channels=3)
                feats.extend(conv2d_svhn(3, 32))
            else:
                feats.extend(conv2d_svhn(32*(2**(i-2)), 32*(2**(i-1))))
            
            if i <= 3:
                # last conv block has only 1 conv layer
                feats.extend(conv2d_svhn(32*(2**(i-1)), 32*(2**(i-1))))
            feats.append(nn.MaxPool2d(2, 2))
        
        features = nn.Sequential(*feats)
        
        flat = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        )

        classifier = nn.Sequential(
            nn.Linear(256, num_classes)
        )

        super(VGG_SVHN, self).__init__(features, flat,  classifier)
    

class LeNet5(CNN):
    '''
    Note: specific for MNIST or any grayscale 32x32 variation.
    Variation w.r.t. the original implementation:
    * uses ReLU instead of tanh
    * 2nd conv2d's 16 channels are all connected to the previous avgpool; in original implementation, only 10 of 16 are connected
    '''
    def __init__(self, num_classes):
        features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        flat = nn.Sequential(
            Flatten()
        )
        classifier = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
        super(LeNet5, self).__init__(features, flat,  classifier)


