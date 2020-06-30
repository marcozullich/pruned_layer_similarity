import torch
import unittest
import sys
import torch_testing as tt

sys.path.append("..")
from layer_sim.pruning.LF_mask import _build_pruning_mask, apply_mask, lf_mask_global, mask_prop_params
from layer_sim.networks import VGG_SVHN

class TestLFMask(unittest.TestCase):
    def test_build(self):
        layers_dict = {
            1: torch.Tensor([[-1,5,-7],[2,-3,1],[0,0,0]]),
            2: torch.Tensor([2,-3,4,-5,-1,7,2,-4,3,-2,8])
        }
        # should prune 0s and 1s (also -1s)
        mask = _build_pruning_mask(layers_dict, .3)
        
        tt.assert_equal(mask[1].int(), torch.Tensor([[0,1,1],[1,1,0],[0,0,0]]))
        tt.assert_equal(mask[2].int(), torch.Tensor([1,1,1,1,0,1,1,1,1,1,1]))
        

    def test_mask(self):
        torch.random.manual_seed(222)
        net = VGG_SVHN()

        re_layers = [r"features\.\d+\.weight", r"features\.\d+\.bias"]
        mask = lf_mask_global(net, layer_ids_to_prune=re_layers) # prune only conv weights & biases

        self.assertNotIn("classifier.0.weight", mask.keys())
        self.assertNotIn("classifier.0.bias", mask.keys())
        self.assertNotIn("features.11.num_batches_tracked", mask.keys())
        self.assertIn("features.7.weight", mask.keys())
        self.assertIn("features.22.bias", mask.keys())
        self.assertEqual(len(mask), 28)

        prop_params_mask_nonet = mask_prop_params(mask)
        self.assertAlmostEqual(prop_params_mask_nonet, 0.8, places=4)

        prop_params_mask_net = mask_prop_params(mask, net)
        self.assertAlmostEqual(prop_params_mask_net, 0.802, places=3)

        apply_mask(net, mask, sequential=True)

        submask_old = mask['features.7.weight'][0,:50]
        zero_indices_submask = torch.where(submask_old==0)
        tt.assert_equal(
            mask['features.7.weight'][0,zero_indices_submask[0], zero_indices_submask[1], zero_indices_submask[2]], 
            net.state_dict()['features.7.weight'][0, zero_indices_submask[0], zero_indices_submask[1], zero_indices_submask[2]]
        )

        mask = lf_mask_global(net, layer_ids_to_prune=re_layers, previous_mask=mask)
        # submask_new = mask['features.7.weight'][0,:50]
        # # test that all previous zeros are zeros in the new mask
        # tt.assert_equal(
        #     submask_new[zero_indices_submask[0], zero_indices_submask[1], zero_indices_submask[2]],
        #     submask_old[zero_indices_submask[0], zero_indices_submask[1], zero_indices_submask[2]]
        # )

        prop_params_mask_nonet = mask_prop_params(mask)
        self.assertAlmostEqual(prop_params_mask_nonet, 0.64, places=4)




        
