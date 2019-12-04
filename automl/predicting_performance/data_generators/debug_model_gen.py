import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

import numpy as np

from collections import OrderedDict

import time
import random
import os
import yaml
from yaml import Loader, Dumper
import matplotlib.pyplot as plt

from pathlib import Path

import subprocess

from pdb import set_trace as st
# from torchsummary import summary

from predicting_performance import metrics

from predicting_performance.stats_collector import StatsCollector
from predicting_performance.trainer import Trainer
from predicting_performance.data_loader_cifar import get_cifar10_for_data_point_mdl_gen

from utils.utils import count_nb_params
from utils.utils import make_and_check_dir, timeSince

from predicting_performance.data_point_models.debug_models import get_debug_models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_model_info(data_path, mdl, init_params, final_params,
    train_loss, train_error, val_loss, val_error, test_loss, test_error,
    optimizer, epochs, criterion, error_criterion,
    hours, mdl_name, init_algorithm_list, init_hyerparam_list,
    batch_size_train, batch_size_val, batch_size_test,
    number_parameters,
    scheduler=None, other_data=None):
    """
    Saves model info to yaml and numpy files to be used later.
    Tips:
        - you can save loss = error if criterion = error_criterion.
        - if you have no val set set val_loss and val_error = -1
    :param str data_path: path to the main dolfer of the data set e.g. '../main_full_auto_ml/data/automl_dataset_debug/'
    :param torch.nn.Sequential mdl: model generated from model data generator
    :param torch.tensor init_params: initial weights for the model
    :param torch.tensor final_params: final weights for the model
    :param float train_error: training error for the model
    :param float test_error: test error for the model
    :param torch.optim optimizer: optimizer used in the model
    :param int epochs: number of epochs the model was trained for
    :param torch.nn loss: the loss function used in the model
    :param str mdl_name: name of the model to save
    TODO: change so that the path to where things should be save is part of the class data-gen rather than here
    """
    if scheduler is None:
        milestones, scheduler_gamma, last_epoch = [], 1.0, -1
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=scheduler_gamma, last_epoch=last_epoch)
    milestones, scheduler_gamma, last_epoch = list(scheduler.milestones), scheduler.gamma, scheduler.last_epoch
    print(f'milestones, scheduler_gamma, last_epoch = {milestones, scheduler_gamma, last_epoch}')
    ##format mdl, loss function and optimizer to be put into the metadata file
    meta_data = f"arch_and_hp: '{str(mdl)}'"
    optimizer = f"optimizer: '{str(optimizer)}'"
    scheduler = f'scheduler: {str(scheduler)}'
    milestones = f'milestones: {str(milestones)}'
    scheduler_gamma = f'scheduler_gamma: {str(scheduler_gamma)}'
    last_epoch = f'last_epoch: {str(last_epoch)}'
    epochs = f"epochs: {epochs}"
    criteron = f"criterion: '{str(criterion)}'"
    error_criterion = f"error_criterion: {error_criterion}"
    train_loss = f'train_loss: {str(float(train_loss))}'
    train_error = f'train_error: {str(float(train_error))}'
    val_loss = f'val_loss: {str(float(val_loss))}'
    val_error = f'val_error: {str(float(val_error))}'
    test_loss = f'test_loss: {str(float(test_loss))}'
    test_error = f'test_error: {str(float(test_error))}'
    number_parameters = f'number_parameters: {str(number_parameters)}'
    init_algorithm_list = f'init_algorithm_list: {init_algorithm_list}'
    init_hyerparam_list = f'init_hyerparam_list: {init_hyerparam_list}'
    batch_size_train = f'batch_size_train: {batch_size_train}'
    batch_size_test = f'batch_size_test: {batch_size_test}'
    batch_size_val = f'batch_size_val: {batch_size_val}'
    #save mdl path
    mdl_path = f'mdl_{mdl_name}'
    process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    git_head_hash = process.communicate()[0].strip()
    git_head_hash = f'git_head_hash: {git_head_hash}'
    folder_path = os.path.join(data_path, mdl_path)
    #meta_data_path = os.path.join(folder_path,f'meta_data_{mdl_name}.yml')
    meta_data_path = os.path.join(folder_path,f'meta_data.yml')
    print(f'meta_data_path = {meta_data_path}')
    make_and_check_dir(folder_path)
    explicit_start = False
    default_flow_style = False
    with open(meta_data_path, "w") as conffile:
        #writes meta_data info to the yaml file
        yaml.dump(yaml.load(meta_data, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(optimizer, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(scheduler, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(milestones, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(scheduler_gamma, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(last_epoch, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        print(f'---> milestones, scheduler_gamma, last_epoch = {milestones, scheduler_gamma, last_epoch}')
        yaml.dump(yaml.load(epochs, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(criteron, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(train_loss, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(train_error, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(val_loss, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(val_error, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(test_loss, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(test_error, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(number_parameters, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(init_algorithm_list, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(init_hyerparam_list, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        seconds = hours*60.0*60.0
        minutes = hours*60.0
        days = hours/24.0
        seconds = f'seconds: {seconds}'
        minutes = f'minutes: {minutes}'
        hours = f'hours: {hours}'
        days = f'days: {days}'
        yaml.dump(yaml.load(str(seconds), Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(str(minutes), Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(str(hours), Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(str(days), Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(batch_size_train, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(batch_size_test, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(batch_size_val, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(git_head_hash, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
    ##
    #other_data_path = os.path.join(folder_path,f'other_data_{mdl_name}.yml')
    other_data_path = os.path.join(folder_path,f'other_data.yml')
    if other_data is not None:
        with open(other_data_path, "w") as otherfile:
            yaml.dump(other_data, otherfile, explicit_start=explicit_start, default_flow_style=default_flow_style)
    #writes tensors to numpy file.
    #meta_data_path = os.path.join(folder_path,f'tensors_{mdl_name}')
    param_stats_data_path = os.path.join(folder_path,f'param_stats.yml')
    # save in numpy array format (if not done this it willl save if as pytorch)
    ## init_params = [ param.data.cpu().numpy() for param in init_params]
    ## final_params = [ param.data.cpu().numpy() for param in final_params]
    init_params_mu = [ param.data.mean().item() for param in init_params]
    final_params_mu = [ param.data.mean().item() for param in final_params]
    init_params_std = [ param.data.std().item() for param in init_params]
    final_params_std = [ param.data.std().item() for param in final_params]
    init_params_l2 = [ param.data.norm(2).item() for param in init_params]
    final_params_l2 = [ param.data.norm(2).item() for param in final_params]
    with open(param_stats_data_path, "w") as otherfile:
        yaml.dump(yaml.load(f'init_params_mu: {init_params_mu}', Loader=Loader), otherfile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(f'final_params_mu: {final_params_mu}', Loader=Loader), otherfile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(f'init_params_std: {init_params_std}', Loader=Loader), otherfile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(f'final_params_std: {final_params_std}', Loader=Loader), otherfile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(f'init_params_l2: {init_params_l2}', Loader=Loader), otherfile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(f'final_params_l2: {final_params_l2}', Loader=Loader), otherfile, explicit_start=explicit_start, default_flow_style=default_flow_style)
    #np.savez(meta_data_path, init_params=init_params, final_params=final_params, train_loss=train_loss, train_error=train_error, test_loss=test_loss, test_error=test_error)

def save_model_info_lstm(data_path,data_path_save,init_params, final_params,
    train_loss, train_error, val_loss, val_error, test_loss, test_error,epochs,optimizer,
    hours,batch_size_train, batch_size_val, batch_size_test,scheduler=None,other_data = None):
    """
    save_model_info(data_path, mdl, init_params, final_params,
        train_loss, train_error, val_loss, val_error, test_loss, test_error,
        optimizer, epochs, criterion, error_criterion,
        hours, mdl_name, init_algorithm_list, init_hyerparam_list,
        batch_size_train, batch_size_val, batch_size_test,
        number_parameters,
        scheduler=None, other_data=None):
    """
    if scheduler is None:
        milestones, scheduler_gamma, last_epoch = [], 1.0, -1
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=scheduler_gamma, last_epoch=last_epoch)
    milestones, scheduler_gamma, last_epoch = list(scheduler.milestones), scheduler.gamma, scheduler.last_epoch
    print(f'milestones, scheduler_gamma, last_epoch = {milestones, scheduler_gamma, last_epoch}')

    scheduler = f'scheduler: {str(scheduler)}'
    milestones = f'milestones: {str(milestones)}'
    scheduler_gamma = f'scheduler_gamma: {str(scheduler_gamma)}'
    last_epoch = f'last_epoch: {str(last_epoch)}'
    epochs = f"epochs: {epochs}"
    train_loss = f'train_loss: {str(float(train_loss))}'
    train_error = f'train_error: {str(float(train_error))}'
    val_loss = f'val_loss: {str(float(val_loss))}'
    val_error = f'val_error: {str(float(val_error))}'
    test_loss = f'test_loss: {str(float(test_loss))}'
    test_error = f'test_error: {str(float(test_error))}'
    batch_size_train = f'batch_size_train: {batch_size_train}'
    batch_size_test = f'batch_size_test: {batch_size_test}'
    batch_size_val = f'batch_size_val: {batch_size_val}'
    #save mdl path
    process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    git_head_hash = process.communicate()[0].strip()
    git_head_hash = f'git_head_hash: {git_head_hash}'
    folder_path = os.path.join(data_path)
    #meta_data_path = os.path.join(folder_path,f'meta_data_{mdl_name}.yml')
    meta_data_path = os.path.join(folder_path,f'meta_data.yml')
    print(f'meta_data_path = {meta_data_path}')
    make_and_check_dir(folder_path)
    explicit_start = False
    default_flow_style = False
    with open(meta_data_path, "w") as conffile:
        #writes meta_data info to the yaml file
        # yaml.dump(yaml.load(meta_data, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        # yaml.dump(yaml.load(optimizer, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        # yaml.dump(yaml.load(scheduler, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        # yaml.dump(yaml.load(milestones, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        # yaml.dump(yaml.load(scheduler_gamma, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        # yaml.dump(yaml.load(last_epoch, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        # print(f'---> milestones, scheduler_gamma, last_epoch = {milestones, scheduler_gamma, last_epoch}')
        yaml.dump(yaml.load(epochs, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        # yaml.dump(yaml.load(criteron, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(train_loss, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(train_error, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(val_loss, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(val_error, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(test_loss, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(test_error, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)

        # minutes = hours*60.0
        # days = hours/24.0
        # seconds = f'seconds: {seconds}'
        # minutes = f'minutes: {minutes}'
        # hours = f'hours: {hours}'
        # days = f'days: {days}'
        # yaml.dump(yaml.load(str(seconds), Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        # yaml.dump(yaml.load(str(minutes), Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        # yaml.dump(yaml.load(str(hours), Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        # yaml.dump(yaml.load(str(days), Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(batch_size_train, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(batch_size_test, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(batch_size_val, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(git_head_hash, Loader=Loader), conffile, explicit_start=explicit_start, default_flow_style=default_flow_style)
    other_data_path = os.path.join(folder_path,f'other_data.yml')
    if other_data is not None:
        with open(other_data_path, "w") as otherfile:
            yaml.dump(other_data, otherfile, explicit_start=explicit_start, default_flow_style=default_flow_style)
    param_stats_data_path = os.path.join(folder_path,f'param_stats.yml')
    # save in numpy array format (if not done this it willl save if as pytorch)
    ## init_params = [ param.data.cpu().numpy() for param in init_params]
    ## final_params = [ param.data.cpu().numpy() for param in final_params]
    init_params_mu = [ param.data.mean().item() for param in init_params]
    final_params_mu = [ param.data.mean().item() for param in final_params]
    init_params_std = [ param.data.std().item() for param in init_params]
    final_params_std = [ param.data.std().item() for param in final_params]
    init_params_l2 = [ param.data.norm(2).item() for param in init_params]
    final_params_l2 = [ param.data.norm(2).item() for param in final_params]
    with open(param_stats_data_path, "w") as otherfile:
        yaml.dump(yaml.load(f'init_params_mu: {init_params_mu}', Loader=Loader), otherfile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(f'final_params_mu: {final_params_mu}', Loader=Loader), otherfile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(f'init_params_std: {init_params_std}', Loader=Loader), otherfile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(f'final_params_std: {final_params_std}', Loader=Loader), otherfile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(f'init_params_l2: {init_params_l2}', Loader=Loader), otherfile, explicit_start=explicit_start, default_flow_style=default_flow_style)
        yaml.dump(yaml.load(f'final_params_l2: {final_params_l2}', Loader=Loader), otherfile, explicit_start=explicit_start, default_flow_style=default_flow_style)
    #np.savez(meta_data_path, init_params=init_params, final_params=final_params, train_loss=train_loss, train_error=train_error, test_loss=test_loss, test_error=test_error)


class ModelDataGenerator():

    def __init__(self, path, trainloader, valloader, testloader, min_train_epochs=600, max_train_epochs=800):
        '''
        TODO: perhaps isntead of having a giant argument to mode data ModelDataGenerator
        have a config file with all that it needs...
        '''
        ## the paths to saves the data set
        self.path = path.expanduser()
        ## the dataset
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        ## epoch range for the sampler
        self.min_train_epochs = min_train_epochs
        self.max_train_epochs = max_train_epochs

    def generate_debug_dataset(self):
        '''
        Generates a debugging dataset to cifar10
        '''
        self.path.mkdir(exist_ok=True)
        iterations, train_iterations = 1, 1 # CAREFUL it might be returining 1, for real production we want it to be greater than 1
        mdls = get_debug_models()
        print()
        for i in range(len(mdls)):
            print(f'---> mdl_{i}')
            start = time.time()
            ## generate mdl data point
            mdl = mdls[i].to(device)
            epochs = random.randint(self.min_train_epochs, self.max_train_epochs) # CAREFUL it might be returining 1, for real production we want it to be greater than 1
            optimizer = torch.optim.Adam(mdl.parameters())
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1.0)
            criterion = nn.CrossEntropyLoss()
            error_criterion = metrics.error_criterion
            init_params = list(mdl.parameters())
            stats_collector = StatsCollector()
            trainer = Trainer(self.trainloader, self.valloader, self.testloader, optimizer, scheduler, criterion, error_criterion, stats_collector, device)
            train_loss, train_error, val_loss, val_error, test_loss, test_error = trainer.train_and_track_stats(mdl, epochs, iterations=iterations, train_iterations=train_iterations)
            final_params = list(mdl.parameters())
            ## save data point
            how_long, hours = timeSince(start)
            print(f'hours = {hours}')
            print(f'{how_long}')
            mdl_name = f'debug_{i}'
            other_data = trainer.stats_collector.get_stats_dict({'error_criterion':error_criterion.__name__})
            # TODO: fix later
            init_algorithm_list = 'default'
            init_hyerparam_list = torch.__version__
            number_parameters = count_nb_params(mdl)
            ##
            batch_size_train, batch_size_val, batch_size_test = self.trainloader.batch_size, self.valloader.batch_size, self.testloader.batch_size
            data_path = str(self.path)
            save_model_info(data_path, mdl, init_params, final_params,
                train_loss, train_error, val_loss, val_error, test_loss, test_error,
                optimizer, epochs, criterion, error_criterion,
                hours, mdl_name, init_algorithm_list, init_hyerparam_list,
                batch_size_train, batch_size_val, batch_size_test,
                number_parameters,
                scheduler=scheduler, other_data=other_data)
            print(f'--> mdl_{i} data point saved!\n')

## code to make data sets

def create_debug_data_set():
    '''
    Create a dummy/toy data set of models from hardcoded models with a trainer class
    '''
    trainloader, valloader, testloader = get_cifar10_for_data_point_mdl_gen()
    ## these two numbers are so that the debug data set runs fast, in real production they should NOT be 1.
    min_train_epochs, max_train_epochs = 1, 1 # CAREFUL it might be returining 1, for real production we want it to be greater than 1
    ##
    data_path = '~/predicting_generalization/automl/data/automl_dataset_debug' # note u can use: os.path.expanduser
    path = Path(data_path).expanduser()
    datagen = ModelDataGenerator(path, trainloader, valloader, testloader, min_train_epochs=1, max_train_epochs=1)
    datagen.generate_debug_dataset()
    ##
    other_data_path = os.path.join(str(path),'mdl_debug_0/other_data.yml')
    with open(other_data_path, "r") as metadata_file:
        metadata = yaml.load(metadata_file, Loader=Loader)
        test_errors = metadata['test_errors']
        print(f'test_errors = {test_errors}')

if __name__ == '__main__':
    #create_real_cifar10_meta_learning_dataset()
    create_debug_data_set()
    print('done \a')
