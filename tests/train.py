import unittest
import torch
from torchvision.models.vgg import vgg11
import sys

sys.path.append("..")
from layer_sim.train import accuracy_at_k, anneal_lr

class TestTrain(unittest.TestCase):
    def test_accuracy_k(self):
        output1 = torch.Tensor([[3,2,7],[1,5,4],[3,0,2],[5,1,0]])
        gr_truth1 = torch.Tensor([2,2,0,0])
        self.assertAlmostEqual(accuracy_at_k(output1, gr_truth1), .75)
        self.assertAlmostEqual(accuracy_at_k(output1, gr_truth1, 2), 1.0)

        output2 = torch.Tensor([[5,4,0,0,2,1,6],
                                [6,0,0,1,2,0,0],
                                [4,1,5,0,2,0,1],
                                [0,2,3,5,2,1,0],
                                [1,7,0,2,0,5,1]])
        gr_truth2 = torch.Tensor([0,1,2,3,5])
        self.assertAlmostEqual(accuracy_at_k(output2, gr_truth2), 2/5)
        self.assertAlmostEqual(accuracy_at_k(output2, gr_truth2, 2), 4/5)
        self.assertAlmostEqual(accuracy_at_k(output2, gr_truth2, 3), 4/5)
    
    def test_anneal_lr(self):
        net = vgg11()
        opt = torch.optim.SGD(net.parameters(), lr = .1)
        anneal_lr(opt, 10)
        for o in opt.param_groups:
            self.assertEqual(o["lr"], 0.01)
