import torch.nn as nn

from predicting_performance.data_point_models.custom_layers import Flatten

from collections import OrderedDict

from pdb import set_trace as st

def get_debug_models():
    '''
    Debugging data set for Cifar10.
    :return list: a list of debugging models for cifar10
    '''
    ## mdl1
    mdl1 = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(in_channels=3,out_channels=4,kernel_size=3)),
        ('relu1', nn.ReLU()),
        ('conv2', nn.Conv2d(in_channels=4,out_channels=2,kernel_size=3)),
        ('relu2', nn.ReLU()),
        ('Flatten', Flatten()),
        ('fc', nn.Linear(in_features=28*28*2,out_features=10) )
    ]))
    ## mdl2
    mdl2 = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(in_channels=3,out_channels=1,kernel_size=3)),
        ('relu1', nn.ReLU()),
        ('Flatten', Flatten()),
        ('fc', nn.Linear(in_features=30*30*1,out_features=10) )
    ]))
    ## mdl3
    mdl3 = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(in_channels=3,out_channels=2,kernel_size=5)),
        ('relu1', nn.ReLU()),
        ('Flatten', Flatten()),
        ('fc', nn.Linear(in_features=28*28*2,out_features=10) )
    ]))
    ## mdl4
    mdl4 = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(in_channels=3,out_channels=3,kernel_size=7)),
        ('relu1', nn.ReLU()),
        ('Flatten', Flatten()),
        ('fc', nn.Linear(in_features=26*26*3,out_features=10) )
    ]))
    ## mdl5, from cifar10 tutorial
    mdl5 = nn.Sequential(OrderedDict([
        ('pool1', nn.MaxPool2d(2, 2)),
        ('relu1', nn.ReLU()),
        ('conv1', nn.Conv2d(3, 6, 5)),
        ('pool1', nn.MaxPool2d(2, 2)),
        ('relu2', nn.ReLU()),
        ('conv2', nn.Conv2d(6, 16, 5)),
        ('relu2', nn.ReLU()),
        ('Flatten', Flatten()),
        ('fc1', nn.Linear(1024, 120)), # figure out equation properly
        ('relu4', nn.ReLU()),
        ('fc2', nn.Linear(120, 84)),
        ('relu5', nn.ReLU()),
        ('fc3', nn.Linear(84, 10))
    ]))
    return [mdl1, mdl2, mdl3, mdl4, mdl5]
