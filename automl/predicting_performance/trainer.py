'''
based on code: https://github.com/brando90/overparam_experiments/blob/master/pytorch_experiments/new_training_algorithms.py

'''
import time
import numpy as np
import torch

from torch.autograd import Variable

from math import inf
import os

from predicting_performance.stats_collector import StatsCollector

from pdb import set_trace as st

def get_function_evaluation_from_name(name):
    if name == 'evalaute_running_mdl_data_set':
        evalaute_mdl_data_set = evalaute_running_mdl_data_set
    elif name == 'evalaute_mdl_on_full_data_set':
        evalaute_mdl_data_set = evalaute_mdl_on_full_data_set
    else:
        return None
    return evalaute_mdl_data_set

def evalaute_running_mdl_data_set(loss, error, net, dataloader, device, iterations=inf):
    '''
    Evaluate the approx (batch) error of the model under some loss and error with a specific data set.
    The batch error is an approximation of the train error (empirical error), so it computes average batch size error
    over all the batches of a specific size. Specifically it computes:
    avg_L = 1/N_B sum^{N_B}_{i=1} (1/B sum_{j \in B_i} l_j )
    which is the average batch loss over N_B = ceiling(# data points/ B ).

    The argument iterations determines the # of batches to compute the error on.
    It's inclusive  e.g. If its 2 then it computes the average error on 2 batches.
    If iterations=0 then we compute the error on zero batches which gives an error.
    Thus, we this number must be 0>1 or it will return an error.

    Note: this method is approximate (TODO: how so and why)

    :param int iterations: the # of times to sample a batch from the dataloader. note: ceil(N/batch_size) = total number of iterations to do 1 epoch.
    '''
    if iterations <= 0:
        raise ValueError(f'argument iterations must be strictly greater than 0, but values was {iterations}')
    running_loss,running_error = 0,0
    with torch.no_grad():
        for i,(inputs,targets) in enumerate(dataloader):
            if i >= iterations:
                break
            if type(inputs) is torch.Tensor:
                inputs,targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            running_loss += loss(outputs,targets).item()
            running_error += error(outputs,targets).item()
    return running_loss/(i+1),running_error/(i+1)

def evalaute_mdl_on_full_data_set(loss, error, net, dataloader, device, iterations=inf):
    '''
    Evaluate the error of the model under some loss and error with a specific data set, but use the full data set.
    The argument iterations determines the # of batches to compute the error on.
    It's inclusive  e.g. If its 2 then it computes the average error on 2 batches.
    If iterations=0 then we compute the error on zero batches which gives an error.
    Thus, we this number must be 0>1 or it will return an error.

    Note: this method computes exact error on dataloader.

    :param int iterations: the # of times to sample a batch from the dataloader. note: ceil(N/batch_size) = total number of iterations to do 1 epoch.
    '''
    if iterations <= 0:
        raise ValueError(f'argument iterations must be strictly greater than 0, but values was {iterations}')
    N = len(dataloader.dataset)
    avg_loss,avg_error = 0,0
    with torch.no_grad():
        for i,(inputs,targets) in enumerate(dataloader):
            batch_size = targets.size()[0]
            if i >= iterations:
                n_total = batch_size*i
                avg_loss = (N/batch_size)*avg_loss
                avg_error = (N/batch_size)*avg_loss
                return avg_loss/n_total,avg_error/n_total
            if type(inputs) is torch.Tensor:
                inputs,targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            avg_loss += (batch_size/N)*loss(outputs,targets).item()
            avg_error += (batch_size/N)*error(outputs,targets).item()
    return avg_loss,avg_error

class Trainer:

    def __init__(self, trainloader, valloader, testloader, optimizer, scheduler, criterion,
                    error_criterion, stats_collector, device,
                    evalaute_mdl_data_set='evalaute_running_mdl_data_set'):
        '''
        '''
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.error_criterion = error_criterion
        self.stats_collector = stats_collector
        self.device = device
        ''' how to evaluate the model as we train '''
        self.evalaute_mdl_data_set = get_function_evaluation_from_name(evalaute_mdl_data_set)
        if evalaute_mdl_data_set is None:
            raise ValueError(f'Data set function evaluator evalaute_mdl_data_set={evalaute_mdl_data_set} is not defined.')

    def train_and_track_stats(self, net, nb_epochs, iterations=4, train_iterations=inf, target_train_loss=inf, precision=0.01, reset_stats_collector=True):
        '''
        Train net with nb_epochs and 1 epoch only # iterations = iterations
        Note, the stats collector is restarted to empty every time this function is called.
        Note the return value is just meant for printing, debugging purposes or saving to yaml files.
        :param torch.model net: pytorch model to be trained.
        :param int nb_epochs: number of epochs to train a data set (epoch=whole data set once)
        :param int iterations: number of batches it uses to evaluate how the model is doing (and storing the value according to this) (1 iteration = 1 evaluation on 1 batch. iteration = nb_epochs means we see whole data set)
        :param int train_iterations: number of batches it uses to train the model per epoch (1 iteration = 1 evaluation on 1 batch. iteration = ceil(N_train/batch_size) means we see whole data set per epoch) set this low to train really quickly.
        :param int target_train_loss: the target loss to halt the model when (approximately) reached according to precision param
        :param float precision: the precision/closeness our model's loss should be to the target loss
        :param boolean reset_stats_collector: weather to resart the stats collector (stats collector collets errors stats etc during training)
        :return train_loss_epoch: loss/error when halted
        :return train_error_epoch: loss/error when halted
        :return test_loss_epoch: loss/error when halted
        :return test_error_epoch: loss/error when halted
        '''
        if train_iterations <= 0:
            raise ValueError(f'argument train_iterations must be strictly greater than 0, but values was {train_iterations}')
        if reset_stats_collector:
            self.stats_collector.reset()
        ''' Add stats before training '''
        train_loss_epoch, train_error_epoch = self.evalaute_mdl_data_set(self.criterion, self.error_criterion, net, self.trainloader, self.device, iterations)
        val_loss_epoch, val_error_epoch = self.evalaute_mdl_data_set(self.criterion, self.error_criterion, net, self.valloader, self.device, iterations)
        test_loss_epoch, test_error_epoch = self.evalaute_mdl_data_set(self.criterion, self.error_criterion, net, self.testloader, self.device, iterations)
        self.stats_collector.append_losses_errors_accs(train_loss_epoch, train_error_epoch, val_loss_epoch, val_error_epoch, test_loss_epoch, test_error_epoch)
        print(f'(train_loss: {train_loss_epoch}, train error: {train_error_epoch}) , (test loss: {test_loss_epoch}, test error: {test_error_epoch})')
        ''' Start training '''
        print('about to start training')
        for epoch in range(nb_epochs):  # loop over the dataset multiple times
            self.scheduler.step()
            net.train()
            running_train_loss,running_train_error = 0.0, 0.0
            for i,(inputs,targets) in enumerate(self.trainloader, 0): # complete one epoch i.e. one complete sweep over data set
                #print(f'i , train_iterations = {i , train_iterations}')
                if i >= train_iterations:
                    #print(f'BREAK, i, train_iterations = {i, train_iterations}')
                    break
                ''' zero the parameter gradients '''
                self.optimizer.zero_grad()
                ''' train step = forward + backward + optimize '''
                if type(inputs) is torch.Tensor:
                    inputs,targets = inputs.to(self.device),targets.to(self.device)
                outputs = net(inputs)
                loss = self.criterion(outputs,targets)
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item()
                running_train_error += self.error_criterion(outputs,targets)
                ''' print error first iteration'''
                #if i == 0 and epoch == 0: # print on the first iteration
                #    print(data_train[0].data)
            #print(f'i={i}')
            ''' End of Epoch: evaluate nets on data '''
            net.eval()
            if self.evalaute_mdl_data_set.__name__ == 'evalaute_running_mdl_data_set':
                train_loss_epoch, train_error_epoch = running_train_loss/(i+1), running_train_error/(i+1)
            else:
                train_loss_epoch, train_error_epoch = self.evalaute_mdl_data_set(self.criterion, self.error_criterion, net, self.trainloader, self.device, iterations)
            val_loss_epoch, val_error_epoch = self.evalaute_mdl_data_set(self.criterion, self.error_criterion, net, self.valloader, self.device, iterations)
            test_loss_epoch, test_error_epoch = self.evalaute_mdl_data_set(self.criterion, self.error_criterion, net, self.testloader, self.device, iterations)
            ''' collect results at the end of epoch'''
            self.stats_collector.append_losses_errors_accs(train_loss_epoch, train_error_epoch, val_loss_epoch, val_error_epoch, test_loss_epoch, test_error_epoch)
            #print(f'[e={epoch}, it={i+1}], (train_loss: {train_loss_epoch}, train error: {train_error_epoch}) , (test loss: {test_loss_epoch}, test error: {test_error_epoch})')
            print(f'[e={epoch},it={i+1}], train_loss: {train_loss_epoch}, train error: {train_error_epoch}, test loss: {test_loss_epoch}, test error: {test_error_epoch}')
            ''' check target loss '''
            if abs(train_loss_epoch - target_train_loss) < precision:
                return train_loss_epoch, train_error_epoch, val_loss_epoch, val_error_epoch, test_loss_epoch, test_error_epoch
        return train_loss_epoch, train_error_epoch, val_loss_epoch, val_error_epoch, test_loss_epoch, test_error_epoch
