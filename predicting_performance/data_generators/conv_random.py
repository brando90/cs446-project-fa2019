import torch
import torchvision
import torch.nn as nn

import numpy as np

import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import models

from collections import OrderedDict
import random
import os
import yaml
from yaml import Loader, Dumper
import time

import math
from math import inf

from pathlib import Path

import predicting_performance.metrics as metrics

from predicting_performance.stats_collector import StatsCollector
from predicting_performance.trainer import Trainer

from utils.utils import make_and_check_dir, timeSince, report_times

from predicting_performance.data_generators.debug_model_gen import save_model_info

from predicting_performance.data_loader_cifar import get_cifar10_for_data_point_mdl_gen
from predicting_performance.data_point_models.custom_layers import Flatten

from pdb import set_trace as st

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Gene_data(nn.Module):
    def __init__(self, para_min = 40000, min_filter = 26, min_fc = 32, max_filter = 32, max_fc = 256,min_num_layers=5, max_num_layers=20, max_para_times = 50, default_init_w_algor = True):
        super().__init__()
        '''
        This class is to generate a almost total random model.
        self.min_filter: minimum number of filter size for convolution layers
        self.min_fc: minimum number of filter size for fully-connected layer
        self.max_filte: maximum number of filter size for convolution layers
        self.max_fc: maximym number of filter size for fully-connected layer
        self.in_channels: the number of in_channels of the layers, the first one is always default as 3.
        self.out_channels: number of out_channels,
                            randomly chosen from the range of (min_fiter_size, max_filter_size) for convolution layers
                            randomly chosen from the range of (min_filter, min_fc) for fully-connected layer
        self.kernel_choice: the choice of kernel size for maxpool layer and convolution layers
        self.H = 32:height of picture, the deault initial one is 32 for cifar 10.It will change in conv and pool layer.
        self.W = 32:height of picture, the deault initial one is 32 for cifar 10.It will change in conv and pool layer.
                    always the same with self.H in square pictures.
        self.para_min: minimum number of parameters. Set a limit of the number of parameter in the range of (para_min, para_max).
        self.para_max : maximum number of parameter
        self.max_para_times: self.para_max = self.max_para_times * self.para_min
        self.min_num_layers: minimum number of layers before make10layer
        self.max_num_layers: maximum number of layers before make10layer
        self.draft_order: a list that has the important architect of the model.
        self.architecture_order: a list has the specific details of the architect of the model:
                            such as 1d or 2d, which make10layer to use, where to put the flatten layer
        self.first_linear: if True, there is no linear layer yet, the conv, batchnorm and pool layer still use 2d forma;
                            if False, put a faltten layer before the first linear layer and use 1d format after that.
        self.randomchoice_all: choices of all kinds of layers
        self.radomchoice_no_act: choices of layers except activity layers
        self.randomchoice_act: choices of activity layer
        self.randomchoice_keepsize: layers to be put after make10layer,which would not change the out_channel ==10.
        self.init_choice: choices of init_weight_algorithm.
        self.init_algorithm_list: a list to store the algorithm of init_weight_algorithm.
        self.init_hyerparam_list: a list to store the parameter of init_weight_algorithm.
        '''
        self.min_filter = min_filter
        self.min_fc = min_fc
        self.max_filter = max_filter
        self.max_fc = max_fc
        self.in_channels = 3
        self.out_channels = random.randint(min_filter, max_filter)
        self.kernel_choice = [1,3,5,7,9,11,13]
        self.H = 32
        self.W = 32
        self.para_min = para_min
        self.para_max = para_min * max_para_times
        self.max_para_times = max_para_times
        self.min_num_layers = min_num_layers
        self.max_num_layers = max_num_layers
        self.draft_order = []
        self.architecture_order = []
        self.first_linear = True
        self.randomchoice_all =['conv', 'linear', 'maxpool', 'batch','relu','selu','tanh','leaky_relu','softmax','dropout']
        self.radomchoice_no_act = ['conv', 'linear', 'maxpool', 'batch', 'softmax','dropout']
        self.randomchoice_act = ['relu','selu','tanh','leaky_relu']
        self.randomchoice_keepsize = ['softmax','dropout','batch', 'activity']
        self.init_choice = ['default','xavier_uniform','xavier_normal','He_uniform','He_normal']
        self.init_algorithm_list = []
        self.init_hyerparam_list = []
        self.default_init_w_algor = default_init_w_algor

    def create_draft_order(self):
        '''
        Create a list draft_order:
        1.randomly choose the layers from self.randomchoice_all
        2.make sure no two of the same activity layers in a row.
        3.append a make10layer to ensure the out_channel is 10.
        4.randomly append 0 to 3 layers form self.randomchoice_keepsize which keep the number of self.out_channels.

        '''
        #gets a random number for the number of layers
        num_layers = random.randint(self.min_num_layers,self.max_num_layers)
        print(f"num layers: {num_layers}")

        self.draft_order.append('conv')

        i = 0
        while i <= num_layers:
            self.draft_order.append(random.choice(self.randomchoice_all))
            i = i + 1

        #fix the situation where have two consecutive activity layers
        for i in range(len(self.draft_order) - 1):
            if self.draft_order[i] in self.randomchoice_act and self.draft_order[i + 1] == self.draft_order[i]:
                self.draft_order[i + 1] = random.choice(self.radomchoice_no_act)

        self.draft_order.append('make10layer')
        nb_layers_add = random.choice([0,1,2])
        i = 0
        while i < nb_layers_add:
            toappend = random.choice(self.randomchoice_keepsize)
            if toappend == 'activity':
                self.draft_order.append(random.choice(self.randomchoice_act))
            else:
                self.draft_order.append(toappend)
            i = i + 1
        print(self.draft_order)

    def create_architecture_order(self):
        '''
        Base on the self.draft_order,
        we will have the actual and detail architecture:
        The first linear layer will have a faltten layer before it.
        The conv, maxpool, batchnorm layer after faltten layer will convert from 2d to 1d.
        '''
        for block in self.draft_order:
            if block == 'conv':
                if self.first_linear == True:
                    self.architecture_order.append('conv2d')
                else:
                    self.architecture_order.append('conv1d')
            if block == 'linear':
                if self.first_linear:
                    self.architecture_order.append('Flatten')
                    self.architecture_order.append('linear')
                    self.first_linear = False
                else:
                    self.architecture_order.append('linear')
            if block == 'batch':
                if self.first_linear:
                    self.architecture_order.append('batchnorm')
                else:
                    self.architecture_order.append('batchnorm1D')
            if block == 'softmax':
                self.architecture_order.append('softmax')
            if block == 'maxpool':
                if self.first_linear:
                    self.architecture_order.append('maxpool2d')
                else:
                    self.architecture_order.append('maxpool1d')
            if block == 'relu':
                self.architecture_order.append('relu')
            if block == 'selu':
                self.architecture_order.append('selu')
            if block == 'tanh':
                self.architecture_order.append('tanh')
            if block == 'leaky_relu':
                self.architecture_order.append('leaky_relu')
            if block == 'dropout':
                if self.first_linear:
                    self.architecture_order.append('dropout2d')
                else:
                    self.architecture_order.append('dropout1d')
            if block == 'make10layer':
                self.architecture_order.append('make10layer')

        # print(self.architecture_order)

    def apply_init_algor(self,layer):
        '''
        If we want the init_weight_algorithm not the default one.
        call this function will choose a random init_weight_algorithm.
        Also, information of algorithm and parameters will be store in self.init_algorithm_list and self.init_hyerparam_list
        '''
        init_use = random.choice(self.init_choice)
        self.init_algorithm_list.append(init_use)
        if init_use == 'xavier_uniform':
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
            self.init_hyerparam_list.append(str("gain=calculate_gain('relu')"))
        elif init_use =='xavier_normal':
            nn.init.xavier_normal_(layer.weight)
            self.init_hyerparam_list.append(torch.__version__)
        elif init_use =='He_uniform':
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            self.init_hyerparam_list.append(str("mode='fan_in', nonlinearity='relu'"))
        elif init_use =='He_normal':
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            self.init_hyerparam_list.append(str("mode='fan_out', nonlinearity='relu'"))
        else:
            self.init_hyerparam_list.append(torch.__version__)

    def create_mdl(self):
        '''
        Based on self.architecture_order,
        Build the read mdl.
        '''
        model_architecture = OrderedDict()
        i = 0
        for layer in self.architecture_order:
            if layer == 'conv2d':
                kernel_size = random.choice(self.kernel_choice)
                while kernel_size >= self.in_channels:
                    kernel_size= 1
                model_architecture[f'conv{i}'] = nn.Conv2d(self.in_channels,self.out_channels,kernel_size)
                conv2 = nn.Conv2d(self.in_channels,self.out_channels,kernel_size)
                if self.default_init_w_algor == False:
                    self.apply_init_algor(conv2)
                else:
                    self.init_algorithm_list.append('default')
                    self.init_hyerparam_list.append(torch.__version__)
                self.in_channels = self.out_channels
                self.out_channels = random.randint(self.min_filter, self.max_filter)
                self.H = self.H - kernel_size + 1
            elif layer == 'conv1d':
                kernel_size = random.choice(self.kernel_choice)
                while kernel_size >= self.in_channels:
                    kernel_size= 1
                model_architecture[f'conv{i}'] = nn.Conv1d(self.in_channels,self.out_channels,kernel_size)
                conv1 = nn.Conv1d(self.in_channels,self.out_channels,kernel_size)
                if self.default_init_w_algor == False:
                    self.apply_init_algor(conv1)
                else:
                    self.init_algorithm_list.append('default')
                    self.init_hyerparam_list.append(torch.__version__)
                self.in_channels = self.out_channels
                self.out_channels = random.randint(self.min_filter, self.max_filter)
                self.H = self.H - kernel_size + 1
            elif layer == 'linear':
                model_architecture[f'linear{i}'] = nn.Linear(self.in_channels,self.out_channels)
                lin = nn.Linear(self.in_channels,self.out_channels)
                if self.default_init_w_algor == False:
                    self.apply_init_algor(lin)
                else:
                    self.init_algorithm_list.append('default')
                    self.init_hyerparam_list.append(torch.__version__)
                self.in_channels = self.out_channels
                self.out_channels = random.randint(self.min_fc, self.max_fc)
            elif layer == 'make10layer':
                self.out_channels = 10
                make10layer = random.choice(['linear', 'conv'])
                if make10layer == 'linear':
                    model_architecture[f'linear{i}'] = nn.Linear(self.in_channels,self.out_channels)
                    lin = nn.Linear(self.in_channels,self.out_channels)
                    if self.default_init_w_algor == False:
                        self.apply_init_algor(lin)
                    else:
                        self.init_algorithm_list.append('default')
                        self.init_hyerparam_list.append(torch.__version__)
                elif make10layer == 'conv':
                    kernel_size = random.choice(self.kernel_choice)
                    while kernel_size >= self.in_channels:
                        kernel_size= 1
                    if self.first_linear == True:
                        model_architecture[f'conv{i}'] = nn.Conv2d(self.in_channels,self.out_channels,kernel_size)
                        conv = nn.Conv2d(self.in_channels,self.out_channels,kernel_size)
                    else:
                        model_architecture[f'conv{i}'] = nn.Conv1d(self.in_channels,self.out_channels,kernel_size)
                        conv = nn.Conv1d(self.in_channels,self.out_channels,kernel_size)
                    if self.default_init_w_algor == False:
                        self.apply_init_algor(conv)
                    else:
                        self.init_algorithm_list.append('default')
                        self.init_hyerparam_list.append(torch.__version__)
                self.in_channels = self.out_channels

            elif layer == 'maxpool2d':
                kernel_size = random.choice(self.kernel_choice)
                while kernel_size >= self.in_channels:
                    kernel_size =1
                model_architecture[f'maxpool{i}'] = nn.MaxPool2d(kernel_size,kernel_size)
                self.H = int((self.H - kernel_size) /kernel_size + 1)
                maxpool = nn.MaxPool2d(kernel_size,kernel_size)
                print(maxpool)
            elif layer == 'maxpool1d':
                kernel_size = random.choice(self.kernel_choice)
                while kernel_size >= self.in_channels:
                    kernel_size =1
                model_architecture[f'maxpool{i}'] = nn.MaxPool1d(kernel_size,kernel_size)
                self.H = int((self.H - kernel_size) /kernel_size + 1)
                maxpool = nn.MaxPool2d(kernel_size,kernel_size)
                print(maxpool)
            elif layer == 'batchnorm':
                model_architecture[f'batchnorm{i}'] = nn.BatchNorm2d(self.in_channels)
            elif layer == 'batchnorm1D':
                model_architecture[f'batchnorm1D{i}'] = nn.BatchNorm1d(self.in_channels)
            elif layer == 'relu':
                model_architecture[f'relu{i}'] = nn.ReLU()
            elif layer == 'selu':
                model_architecture[f'selu{i}'] = nn.SELU()
            elif layer == 'tanh':
                model_architecture[f'tanh{i}'] = nn.Tanh()
            elif layer == 'leaky_relu':
                negative_slope = random.random()
                model_architecture[f'leaky_relu{i}'] = nn.LeakyReLU(negative_slope)
            elif layer == 'softmax':
                model_architecture[f'softmax{i}'] = nn.Softmax()
            elif layer == 'Flatten':
                model_architecture[f'flatten{i}'] = Flatten()
                self.in_channels = self.H * self.H * self.in_channels
            elif layer == 'dropout2d':
                p = random.random()
                model_architecture[f'dropout{i}'] = nn.Dropout2d(p)
            elif layer == 'dropout1d':
                p = random.random()
                model_architecture[f'dropout{i}'] = nn.Dropout(p)

            # print(in_channels, self.out_channels)
            i += 1
        mdl = nn.Sequential(model_architecture).to(device) ## Model!

        return mdl,self.init_algorithm_list,self.init_hyerparam_list

def count_nb_params(net):
    '''
    count the number of number_parameters
    '''
    count = 0
    for p in net.parameters():
        count += p.data.nelement()
    return count

def local_test_train(i, data_path):
    '''
    Generates a debugging dataset to cifar10
    This is for use in local CPU to test if the code is working or not.
    Try-catch structure can be applied.
    '''
    success_nb = 0
    while success_nb < 1:
        try:
            gene = Gene_data()
            number_parameters = 0
            while number_parameters < gene.para_min or number_parameters > gene.para_max:
                gene.create_draft_order()
                gene.create_architecture_order()
                mdl, init_algorithm_list,init_hyerparam_list= gene.create_mdl()
                number_parameters = count_nb_params(mdl)
                print(init_algorithm_list)
                print(init_hyerparam_list)
                print(number_parameters)

            trainloader, valloader, testloader = get_cifar10_for_data_point_mdl_gen()
            print(mdl)
            ##
            start = time.time()
            mdl = mdl.to(device)
            epochs = 1
            optimizer = torch.optim.Adam(mdl.parameters())
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1.0)
            criterion = nn.CrossEntropyLoss()
            error_criterion = metrics.error_criterion
            init_params = list(mdl.parameters())
            stats_collector = StatsCollector()
            iterations, train_iterations = 1, 1 # CAREFUL it might be returining 1, for real production we want it to be greater than 1
            trainer = Trainer(trainloader,valloader, testloader, optimizer, scheduler, criterion, error_criterion, stats_collector, device)
            train_loss, train_error, val_loss, val_error, test_loss, test_error = trainer.train_and_track_stats(mdl, epochs, iterations=iterations, train_iterations=train_iterations)
            final_params = list(mdl.parameters())
            ## save data point
            how_long, hours = timeSince(start)
            print(f'hours = {hours}')
            print(f'{how_long}')
            mdl_name = f'debug_{i}'
            other_data = trainer.stats_collector.get_stats_dict({'error_criterion':error_criterion.__name__})
            batch_size_train = trainloader.batch_size
            batch_size_test = testloader.batch_size
            batch_size_val = valloader.batch_size
            save_model_info(data_path, mdl, init_params, final_params,
                train_loss, train_error, val_loss, val_error, test_loss, test_error,
                optimizer, epochs, criterion, error_criterion,
                hours, mdl_name, init_algorithm_list, init_hyerparam_list,
                batch_size_train, batch_size_val, batch_size_test,
                number_parameters,
                scheduler, other_data)
            success_nb =success_nb + 1
            print('Success')
        except Exception as e:
            print('FAIL')
            print(e)

def main(i,gene,data_path,epochs,mdl_name):
    '''
    The main train fucntion to be used on GPU.
    Have a try-catch structure.
    i: each call will run i times and save i data points.
    gene: an object of Gene_data. Can change the default parameters in this fucntion, such as
     (number para_min = 40000, min_filter = 26, min_fc = 32, max_filter = 32, max_fc = 256,min_num_layers=5, max_num_layers=20)
     data
    data_path: the root of where to save the results of training.
    epochs: number of epochs to train.
    mdl_name: change the mdl name.
    '''
    # get model type
    success_nb = 0
    while success_nb < 1:
        try:
            # gene =Gene_data()
            number_parameters = 0
            while number_parameters < gene.para_min or number_parameters > gene.para_max:
                gene.create_draft_order()
                gene.create_architecture_order()
                mdl, init_algorithm_list,init_hyerparam_list= gene.create_mdl()
                number_parameters = count_nb_params(mdl)
                print(number_parameters)

            trainloader, valloader, testloader = get_cifar10_for_data_point_mdl_gen()
            ## create directory to save models
            make_and_check_dir(data_path)
            ## start creating models and its variations
            start = time.time()
            ## generate mdl data point
            mdl = mdl.to(device)
            optimizer = torch.optim.Adam(mdl.parameters())
            criterion = nn.CrossEntropyLoss()
            error_criterion = metrics.error_criterion
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1.0)
            stats_collector = StatsCollector()
            trainer = Trainer(trainloader,valloader, testloader, optimizer, scheduler, criterion, error_criterion, stats_collector, device)
            init_params = list(mdl.parameters())
            train_loss, train_error, val_loss, val_error, test_loss, test_error = trainer.train_and_track_stats(mdl, epochs)
            final_params = list(mdl.parameters())
            ## save data point
            how_long, seconds, minutes, hours = report_times(start)
            print(f'hours = {hours}')
            print(how_long)
            # mdl_name = f'tower_mdl_{i}'
            other_data = trainer.stats_collector.get_stats_dict({'error_criterion':error_criterion.__name__})
            batch_size_train = trainloader.batch_size
            batch_size_test = testloader.batch_size
            batch_size_val = valloader.batch_size
            save_model_info(data_path, mdl, init_params, final_params,
                train_loss, train_error, val_loss, val_error, test_loss, test_error,
                optimizer, epochs, criterion, error_criterion,
                hours, mdl_name, init_algorithm_list, init_hyerparam_list,
                batch_size_train, batch_size_val, batch_size_test,
                number_parameters,
                scheduler, other_data)
            success_nb =success_nb + 1
            print('Success')
        except Exception as e:
            print('FAIL')
            print(e)


if __name__ == '__main__':
    '''
    change those parameter before train:
    i: each call will run i times and save i data points.
    gene: an object of Gene_data. Can change the default parameters in this fucntion, such as
     (number para_min = 40000, min_filter = 26, min_fc = 32, max_filter = 32, max_fc = 256,min_num_layers=5, max_num_layers=20)
     data
    data_path: the root of where to save the results of training.
    epochs: number of epochs to train.
    mdl_name: change the mdl name.
    '''
    # gene =Gene_data()
    # data_path = '~/predicting_generalization/automl/data/'
    # data_path = Path(data_path).expanduser()
    # for i in range(1000):
    #     epochs = random.randint(400,600)
    #     mdl_name = f'conv_random_{i}'
    #     main(i,gene,data_path,epochs,mdl_name)
    # print('done \a')

    #if want to test one local one:
    data_path = '~/predicting_generalization/automl/data/xiao_test_test'
    data_path = Path(data_path).expanduser()
    for i in range(1):
        local_test_train(i,data_path)
    print('done \a')
