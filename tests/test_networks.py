import torch
import unittest
import sys
import torch_testing as tt

sys.path.append("..")
from layer_sim.networks import *
from layer_sim import datasets

class TestCNNLeNet(unittest.TestCase):
    def setUp(self):
        self.lenet = LeNet5(10)
        self.x = torch.rand((2,1,28,28))
    
    def test_output_comp(self):
        self.assertEqual(self.lenet.classifier[-1].out_features, 10)

    def test_forward(self):
        x = self.lenet(self.x)
        self.assertEqual(list(x.shape), [2,10])

    def test_hooks(self):
        self.lenet.eval()
        x, out = self.lenet._hook_layers(self.lenet.features, self.x, layer_types_to_hook=(nn.ReLU, nn.AvgPool2d))
        self.assertEqual(len(out), 4)
        self.assertEqual(list(x.shape), [2, 16, 4, 4])

        x, out = self.lenet._hook_layers(self.lenet.flat, x, layer_types_to_hook=(nn.ReLU, nn.AvgPool2d))
        self.assertEqual(len(out), 0)
        self.assertEqual(list(x.shape), [2, 256])

        x, out = self.lenet._hook_layers(self.lenet.classifier, x, layer_types_to_hook=(nn.ReLU, nn.AvgPool2d))
        self.assertEqual(len(out), 2)
        self.assertEqual(list(x.shape), [2, 10])
        

    def test_forward_with_hooks(self):
        self.lenet.eval()
        out = self.lenet.forward_with_hooks(self.x, layer_types_to_hook=(nn.ReLU, nn.AvgPool2d))
        X = self.lenet(self.x)

        self.assertEqual(len(out), 7)
        tt.assert_almost_equal(X, out[-1])

    def test_representations(self):
        self.lenet.eval()
        loader, _ = datasets.MNIST("../data", 128, train=False, num_workers=4)
        rep = self.lenet.extract_network_representation(loader, limit_datapoints=500, device="cpu", layer_types_to_hook=(nn.ReLU, nn.AvgPool2d))

        self.assertEqual(len(rep), 7)
        for r in rep:
            self.assertEqual(r.size(0), 500)
        
        # testing some specific dims
        self.assertEqual(list(rep[0].shape), [500, 6, 24, 24])
        self.assertEqual(list(rep[3].shape), [500, 16, 4, 4])
        self.assertEqual(list(rep[4].shape), [500, 120])
        self.assertEqual(list(rep[5].shape), [500, 84])

class TestVGGSVHN(unittest.TestCase):
    def setUp(self):
        self.net = VGG_SVHN(5)
        self.x = torch.rand((5, 3, 32, 32))

        class PhonyDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            
            def __getitem__(self, index):
                return self.data[index], 0

            def __len__(self):
                return len(self.data)

        # create random dataset
        dataset = PhonyDataset(torch.rand((100,3,32,32)))
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    
    def test_forward(self):
        x = self.net.forward(self.x)
        self.assertEqual(list(x.shape), [5, 5])
    
    def test_forward_with_hooks(self):
        self.net.eval()
        out = self.net.forward_with_hooks(self.x)
        self.assertEqual(len(out), 12)
        x = self.net(self.x)
        tt.assert_almost_equal(x, out[-1])

    @unittest.skip("data download too heavy")
    def test_representations(self):
        rep = self.net.extract_network_representation(self.loader, limit_datapoints=25, device="cpu")

        self.assertEqual(len(rep), 12)
        for r in rep:
            self.assertEqual(r.size(0), 25)

    def test_failure_pts(self):
        self.assertRaises(RuntimeError, self.net.extract_network_representation, self.loader, limit_datapoints=500)


        


    
    


