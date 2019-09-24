import torch
import torch.nn as nn

import unittest

import utils.utils as utils

from predicting_performance.data_point_models.custom_layers import View

from pdb import set_trace as st

class TestHardSampler(unittest.TestCase):

    def test_imports_through_packages(self):
        helloworld = utils.HelloWorld()
        self.assertTrue( helloworld == 'HelloWorld')

    def test_view_layer(self):
        '''
        Tests that the view layer works
        '''
        ##
        batch_size = 1
        CHW = (3, 32, 32)
        out = torch.randn(batch_size,*CHW)
        print(f'out.size()')
        ##
        conv2d_shape = (-1, 8, 8)
        view = View(shape=(batch_size,*conv2d_shape))
        ##
        out = view(out)
        print(f'out.size()')


if __name__ == '__main__':
    unittest.main()
