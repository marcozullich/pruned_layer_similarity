import torch
from torch import nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class VGG_SVHN(nn.Module):
    
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
                feats.extend(conv2d_svhn(32*(i-1), 32*i))
            
            if i <= 3:
                # last conv block has only 1 conv layer
                feats.extend(conv2d_svhn(32*i, 32*i))
            feats.append(nn.MaxPool2d(2, 2))
        
        self.features = nn.Sequential(*feats)
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
    
    def forward_with_hooks(self, x):
        '''
        Execute a forward pass, hooking all the outputs of the convolutional
        layers (after BN and ReLU), the max-pooling layers, and the output layer.
        Returns a list of tensors containing the output of such layers.
        '''
        out = []

        for layer in self.classifier:
            x = layer(x)
            if isinstance(layer, (nn.ReLU, nn.MaxPool2d)):
                out.append(x)
        
        x = self.avgpool(x)
        x = self.classifier(x)
        out.append(x)

        return out        

class LeNet5(nn.Module):
    '''
    Note: specific for MNIST or any grayscale 32x32 variation.
    Variation w.r.t. the original implementation:
    * uses ReLU instead of tanh
    * 2nd conv2d's 16 channels are all connected to the previous avgpool; in original implementation, only 10 of 16 are connected
    '''
    def __init__(self, num_classes):
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.flat = Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flat(x)
        x = self.classifier(x)

    def forward_with_hooks(self, x):
        out = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, (nn.AvgPool2d, nn.ReLU)):
                out.append(x)
        x = self.flat(x)
        for layer in self.classifier:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                out.append(x)
        out.append(x) # out layer, which is Linear but has no activation
        return out

