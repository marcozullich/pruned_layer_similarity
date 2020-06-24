import unittest
import torch
import sys
import torch_testing as tt

from copy import deepcopy

sys.path.append("..")
from layer_sim.preprocessing import *

class testSVD(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tensor = torch.rand((50, 50))
    
    def test_svd_decomp(self):
        tensor_before = deepcopy(self.tensor)
        u, s, v = svd_decomposition(self.tensor)

        # check that the routine did not modify the tensor
        tt.assert_almost_equal(self.tensor, tensor_before)
        # check correctness of shapes
        self.assertEqual((u.shape, s.shape, v.shape), (torch.Size([50,50]), torch.Size([50]), torch.Size([50,50])))
        # check (almost) correctness of s
        s_true = torch.Tensor([4.0633e+00, 3.7529e+00, 3.6790e+00, 3.4786e+00, 3.2625e+00, 3.2387e+00,                                                 3.1560e+00, 3.1081e+00, 2.9156e+00, 2.8221e+00, 2.6983e+00, 2.6267e+00,                                                 2.4983e+00, 2.4625e+00, 2.3405e+00, 2.2782e+00, 2.2350e+00, 2.1671e+00,                                                 2.0741e+00, 2.0360e+00, 1.8878e+00, 1.8201e+00, 1.7788e+00, 1.7461e+00,                                                 1.6808e+00, 1.6425e+00, 1.5536e+00, 1.3986e+00, 1.3590e+00, 1.3023e+00,                                                 1.1921e+00, 1.1607e+00, 1.0730e+00, 9.8077e-01, 9.1926e-01, 8.4475e-01,                                                 7.8537e-01, 7.1252e-01, 6.5127e-01, 6.0697e-01, 5.1137e-01, 4.8607e-01,                                                 4.1669e-01, 3.3902e-01, 3.1693e-01, 2.5806e-01, 1.8515e-01, 1.5145e-01,                                                 1.0309e-01, 9.6590e-07])
        tt.assert_almost_equal(s, s_true, decimal=4)

    def test_svd_reduce(self):
        tensor_reduced = svd_reduction(self.tensor, var_fract_kept=.99)
        self.assertEqual(tensor_reduced.shape[0], 50)
        self.assertEqual(tensor_reduced.shape[1], 38)
        tensor_reduced = svd_reduction(self.tensor, var_fract_kept=.75)
        self.assertEqual(tensor_reduced.shape[0], 50)
        self.assertLess(tensor_reduced.shape[1], 38)
