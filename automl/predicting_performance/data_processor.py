import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import yaml
from yaml import Loader
import os
from pathlib import Path

from pdb import set_trace as st

## Vocab code

class Vocab:

    def __init__(self):
        ## set architecture vocab data structures
        self.architecture_vocab, self.architecture2idx = self._init_architecture_vocab()
        ## set hyper param vocab data structures
        self.hparms_vocab, self.hptype2idx = self._init_layers_hp_vocab()
        ## Optimizer
        self.optimizer_vocab, self.opt2idx = self._init_optimizer_vocab()
        ## Optimizer hyperparams
        self.hparams_opt_vocab, self.hpopt2idx = self._init_layers_hp_optimizer_vocab()

    def _init_architecture_vocab(self):
        '''
        Initializes the architecture vocabulary
        '''
        architecture_vocab = ['PAD_TOKEN', 'SOS', 'EOS','Conv2d', 'Linear', 'MaxPool2d', 'BatchNorm2d', 'Dropout2d', 'ReLU', 'SELU', 'LeakyReLU', 'Flatten','Tanh','Dropout','BatchNorm1d','Softmax']
        architecture2idx = { architecture_vocab[i]:i for i in range(len(architecture_vocab)) } # faster than using python's list.index(element)
        return architecture_vocab, architecture2idx

    def _init_layers_hp_vocab(self):
        '''
        Initializes the hyper param layers vocab
        '''
        hparms_vocab = ['PAD_TOKEN','SOS', 'EOS','in_features', 'out_features', 'kernel_size', 'stride', 'padding', 'dilation', 'ceil_mode', 'eps', 'momentum', 'affine', 'track_running_stats', 'p', 'bias']
        hptype2idx = { hparms_vocab[i]:i for i in range(len(hparms_vocab))} # faster than using python's list.index(element)
        return hparms_vocab, hptype2idx

    def _init_optimizer_vocab(self):
        '''
        Initializes the hyper param layers vocab
        '''
        optimizer_vocab = ['PAD_TOKEN', 'SOS', 'EOS','SGD', 'Adam', 'Adadelta', 'Adagrad']
        opt2idx = { optimizer_vocab[i]:i for i in range(len(optimizer_vocab))} # faster than using python's list.index(element)
        return optimizer_vocab, opt2idx

    def _init_layers_hp_optimizer_vocab(self):
        '''
        Initializes the hyper param layers vocab
        '''
        hparams_opt_vocab = ['PAD_TOKEN', 'SOS', 'EOS', 'dampening', 'lr', 'momentum', 'nesterov', 'weight_decay', 'rho']
        hpopt2idx = { hparams_opt_vocab[i]:i for i in range(len(hparams_opt_vocab))} # faster than using python's list.index(element)
        return hparams_opt_vocab, hpopt2idx

def get_type(vocab, layer_str):
    '''
    Get's the string type of the layer.

    :param list vocab: a list of all the token types (probably as strings)
    :param str layer_str: a string of a splitted layer e.g. ' Linear(in_features=4, out_features=3, bias=True)\n  (1)'
    :return str arch_token: string representation of layer type.
    '''
    for arch_token in vocab:
        if arch_token in layer_str:
            return arch_token
    raise ValueError(f'The string you have {layer_str} doesn\'t match any of the architecture tokens in {vocab}')

def indices2onehot(indices, vocab_size):
    '''
    Returns the onehot matrix
    '''
    shape = (len(indices), vocab_size)
    matrix = np.zeros(shape)
    # for every symbol index i, place a 1 i the one hot vector in the vocab position symbol_idx
    for i, symbol_idx in enumerate(indices):
        matrix[i,symbol_idx] = 1
    return matrix

## DataProcessing code

class DataProcessor:
    '''
    Class for converting models into vector forms to be used by neural nets.
    '''

    def __init__(self, vocab):
        self.vocab = vocab

    def arch_parse_string(self, layer_str):
        """
        Parses the architecture layer string and gets all the parameters for each layer in list.

        :param str layer_str: the string representation of a layer of a model
        :return list param_vector: python list of parameters for the layer of the model
        """
        params = self.vocab.hptype2idx # dictionary from hyper param type to its index
        curr_word = ''
        param_vector = [0]*len(params)
        #go through the entire string and try to find keywords
        for i in range(len(layer_str)):
            #start a new sublayer_str if there is a space
            if layer_str[i] == ' ':
                curr_word = ''
            else:
                #add the next character to the substring
                curr_word += layer_str[i]
                #separate 'padding' from 'p'
                if layer_str[i] == 'p':
                    #continues if the substring is padding
                    if layer_str[i+1] == 'a':
                        continue
                #Separates function call from keywords
                if layer_str[i] == '(' and layer_str[i-1] != '=':
                    curr_word = ''
                #loop through the keys of the dictionary
                for param in params.keys():
                    #check if our substring is a possible parameter
                    if curr_word in params.keys():
                        #if there is a match then add to the index corresponding to the parameter
                        if curr_word == param:
                            # print(curr_word, params[curr_word])
                            #if there is a ( then add the next character
                            if layer_str[i+2] == '(' and layer_str[i+1] == '=':
                                index = int(params[curr_word])
                                param_vector[index] = int(layer_str[i+3])
                            else:
                                #add a 0 if the word is 'False'
                                if layer_str[i+2] == 'F':
                                    param_vector[int(params[curr_word])] = 0
                                #add a 1 if the word is 'True'
                                elif layer_str[i+2] == 'T':
                                    param_vector[int(params[curr_word])] = 1
                                else:
                                    val = ''
                                    i += 2
                                    #loop through the string until the entire value is found
                                    while layer_str[i] != ',' and layer_str[i] != ')':
                                        val += layer_str[i]
                                        i += 1
                                    param_vector[int(params[curr_word])] = eval(val)
        return param_vector

    def mdl_str2feature_vec(self, mdl_str):
        """
        Makes a one hot matrix from each layer of the architecture data (note doesn't include meta data)

        Note: the names of the layers have to be separated by colons for it to work

        :param str mdl_str: model string e.g. nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        :return np.array feature_matrix: model vector with for layer + meta data e.g [conv,filters...etc...]
        """
        ## arch 2 one-hot
        one_hot_arch_matrix = self.mdl_str2onehot(mdl_str)
        ## hparams 2 matrix
        hp_params_matrix = self.mdl_str2hp_matrix(mdl_str)
        ##append arch + hparam vecs
        feature_matrix = np.concatenate((one_hot_arch_matrix, hp_params_matrix),axis=1)
        return feature_matrix

    def parse_optimizer_string(self, opt_str):
        """
        Parses the optimizer string and gets all its parameters

        :param str opt_str: optimizer string for the model
        :return list param_vector: python list of optimizer parameters
        """
        params = self.vocab.hpopt2idx
        curr_word = ''
        param_vector = np.zeros(len(params))
        for i in range(len(opt_str)):
            #start a new substring if there is a space
            if opt_str[i] == ' ':
                curr_word = ''
            else:
                #add the next character to the substring
                curr_word += opt_str[i]
                for param in params.keys():
                    #check if our substring is a possible parameter
                    if curr_word in params.keys():
                        #if there is a match then add to the index corresponding to the parameter
                        if curr_word == param:
                            val = ''
                            i += 3
                            #loop through the string until the entire value is found
                            while opt_str[i] != ' ':
                                val += opt_str[i]
                                i += 1
                            if val == 'False':
                                param_vector[int(params[curr_word])] = int(0)
                            elif val == 'True':
                                param_vector[int(params[curr_word])] = int(1)
                            #if not true or false put the actual value
                            else:
                                try:
                                    param_vector[int(params[curr_word])] = int(val)
                                except:
                                    param_vector[int(params[curr_word])] = float(val)
        return param_vector

    def optimizer_feature_vec(self, opt_str, epochs):
        """
        Makes a feature_vec for the optimizer used in the model.

        param str opt_str: optimizer string for the model
        return list feature_vector: vector of one-hot and hp_param data

        TODO: its missing the epochs...
        """
        indices = optimizer2indices(opt_str)
        opt_onehot = indices2onehot(indices, len(self.vocab.optimizer_vocab))
        #parses optimizer info for its parameters
        params_vector = self.opt_parse_string(opt_str)
        #add parameters to the one hot vector
        feature_vector = np.concatenate( (opt_onehot, params_vector, [epochs]) )
        return feature_vector

    def calculate_weight_stats(self, weights):
        """
        Calculates the Statistics for the weights.

        param list weights: python list of weights (initial or final)
        return list weight_stats: python list of the statistics of the weights

        TODO: change these to torch ops so that they are done on GPU
        """
        length = len(weights)
        new_weights = []
        for i in range(length):
            #flatten each tensor
            flat_weights = weights[i].flatten()
            #convert each tensor to a numpy array and concatenates it to a a list
            new_weights.extend(flat_weights.cpu().detach().numpy())
        #calculates the stats for the weights
        sum_weights = np.sum(new_weights)
        max_weight = np.max(new_weights)
        min_weight = np.min(new_weights)
        average_weight = np.mean(new_weights)
        std_dev_weight = np.std(new_weights)
        weight_stats = [sum_weights,max_weight,min_weight,average_weight,std_dev_weight]
        return weight_stats

    def mdl_str2onehot(self, mdl_str):
        '''
        Makes a one-hot matrix for the arch from the (whole) model string

        :param str mdl_str: string of the model e.g. e.g. nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        :return np.array one_hot_arch_matrix: one-hot matrix representation of model (nb_layers, dim)
        '''
        indices = self.mdl_str2indices(mdl_str)
        one_hot_arch_matrix = self.indices2arch_onehot(indices)
        return one_hot_arch_matrix

    def mdl_str2hp_matrix(self, mdl_str):
        '''
        Makes a matrix for the hps from the (whole) model string

        :param str mdl_str: string of the model e.g. e.g. nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        :return np.array one_hot_arch_matrix: one hot vector representation of model
        '''
        data = mdl_str.split(':')[1:]
        nb_layers = len(data)
        hp_vocab_size = len(self.vocab.hparms_vocab)
        hp_params_matrix = np.zeros((nb_layers,hp_vocab_size))
        for i in range(nb_layers):
            hparam_vector = self.arch_parse_string(data[i])
            hp_params_matrix[i,:] = hparam_vector
        return hp_params_matrix

    def mdl_str2indices(self, mdl_str):
        '''
        Returns a list of indices corresponding to the model arch of given model string.

        :param str mdl_str: string of the model e.g. e.g. nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        :return list arch_indices: list of corresponding indicies in vocab of each arch layer type
        '''
        data = mdl_str.split(':')[1:]
        nb_layers = len(data)
        arch_vocab_size = len(self.vocab.architecture_vocab)
        arch_indices = []
        for i in range(nb_layers):
            layer_type = get_type(self.vocab.architecture_vocab, data[i] )
            idx_layer_type = self.vocab.architecture2idx[layer_type]
            arch_indices.append(idx_layer_type)
        return arch_indices

    def optimizer_str2indices(self, opt_str):
        """
        Returns a list of indices corresponding to the optimization of given optimization string

        param str opt_str: optimizer string for the model
        return list opt_indices: list of corresponding indicies in vocab of each optimizer type
        """
        #vocab
        opt_indices = []
        for opt_token in self.vocab.optimizer_vocab:
            #if vocab is in the optimizer info then append a 1
            if opt_token in opt_str:
                opt_indx = self.vocab.opt2idx[opt_token]
                opt_indices.append(opt_indx)
                break
        return opt_indices

    def tokens2arch_indices(self, tokens):
        '''

        :param list tokens: list of (string) of tokens
        :return list indicies: list of (ints) of indicies

        TODO:
            - add logic to receive things like torch.nn.Conv2d etc
        '''
        ## check if user passed a single string
        if isinstance(str(tokens), str):
            token_str = tokens
            return self.vocab.architecture2idx[token_str]
        indicies = [ self.vocab.architecture2idx[token_str] for token_str in tokens ]
        return indicies

    def indices2arch_onehot(self, indices):
        if isinstance(indices, int):
            return indices2onehot([indices], len(self.vocab.architecture_vocab))[0]
        one_hot_arch_matrix = indices2onehot(indices, len(self.vocab.architecture_vocab))
        return one_hot_arch_matrix

    def indices2hp_matrix(self, indices):
        '''
        TODO implement but we need to also change mdl_str2hp_matrix
        '''
        if isinstance(indices, int):
            return indices2onehot([indices], len(self.vocab.hparms_vocab))[0]
        one_hot_arch_hp_matrix = indices2onehot(indices, len(self.vocab.hparms_vocab))
        return one_hot_arch_hp_matrix

####

class MetaLearningDataset(Dataset):
    '''
    Data set for meta learning. It contains the architecture, hyperparams,
    optimizer, Weights init and final, and train & test error.

    note:
    __len__ so that len(dataset) returns the size of the dataset.
    __getitem__ to support the indexing such that dataset[i] can be used to get ith sample
    '''

    def __init__(self, data_path, vocab):
        '''
        '''
        self.path = Path(data_path).expanduser()
        print(str(data_path))
        self.model_folders = [ f for f in self.path.iterdir() if f.is_dir() ]
        self.data_processor = DataProcessor(vocab)

    def __len__(self):
        '''
        Returns the number of data points (size of data set).
        '''
        return len(self.model_folders)

    def __getitem__(self, idx):
        '''
        Gets you data point at the given index idx.
        '''
        ## look for the model indexed with idx
        mdl_name = ''
        for f in self.model_folders:
            # TODO fix
            mdl_names = str(f)
            if f'_{idx}' in mdl_names: # is this model the model # idx?
                mdl_name = mdl_names
                break
        ## generate strings to paths
        data_path = str(self.path)
        data_filepath = os.path.join(data_path, mdl_name)
        print("data_filepath",data_filepath)
        metadata_filepath = os.path.join(data_filepath, f'meta_data.yml')
        otherdata_filepath = os.path.join(data_filepath, f'other_data.yml')
        param_stats_filepath = os.path.join(data_filepath, f'param_stats.yml')
        #tensor_filepath = os.path.join(data_filepath, f'tensors.npz')
        ##
        data_item = {}
        with open(metadata_filepath, 'r') as f:
            # loader of data
            yamldata = yaml.load(f, Loader=Loader)
            # get raw data
            data_item['mdl_str'] = yamldata['arch_and_hp']
            mdl_str = data_item['mdl_str']
            data_item['opt_str'] = yamldata['optimizer']
            opt_str = data_item['opt_str']
            data_item['epochs'] = yamldata['epochs']
            epochs = data_item['epochs']
            # data_item['batch_size_test'] = yamldata['batch_size_test']
            # data_item['batch_size_train'] = yamldata['batch_size_train']
            # data_item['batch_size_val'] = yamldata['batch_size_val']
            data_item['batch_size_test'] = 1024
            data_item['batch_size_train'] = 512
            data_item['batch_size_val'] = 512
            try:
                criterion = yamldata['criteron']
            except:
                criterion = yamldata['criterion']
            opt_hp = self.data_processor.parse_optimizer_string(opt_str)
            opt_hp = np.concatenate(([epochs],opt_hp) )
            #
            data_item['train_error'] = yamldata['train_error']
            data_item['test_error'] = yamldata['test_error']
            data_item['train_loss'] = yamldata['train_loss']
            data_item['test_loss'] = yamldata['test_loss']
            ## get arch indices and hyperparams
            arch_indices = self.data_processor.mdl_str2indices(mdl_str)
            data_item['arch_indices'] = arch_indices
            arch_hp = self.data_processor.mdl_str2hp_matrix(mdl_str)
            data_item['arch_hp'] = arch_hp
            ## get hyperparams indices and hyperparams
            opt_indices = self.data_processor.optimizer_str2indices(opt_str)
            data_item['opt_indices'] = opt_indices
            opt_hp = self.data_processor.parse_optimizer_string(opt_str)
            data_item['opt_hp'] = opt_hp
        with open(otherdata_filepath, 'r') as f:
            yamldata = yaml.load(f, Loader=Loader)
            #
            data_item['test_accs'] = yamldata['test_accs']
            data_item['test_errors'] = yamldata['test_errors']
            data_item['test_losses'] = yamldata['test_losses']
            #
            data_item['train_accs'] = yamldata['train_accs']
            data_item['train_errors'] = yamldata['train_errors']
            data_item['train_losses'] = yamldata['train_losses']
            #
            data_item['val_accs'] = yamldata['val_accs']
            data_item['val_errors'] = yamldata['val_errors']
            data_item['val_losses'] = yamldata['val_losses']
        with open(param_stats_filepath, 'r') as f:
            yamldata = yaml.load(f, Loader=Loader)
            #
            data_item['init_params_mu'] = yamldata['init_params_mu']
            data_item['final_params_mu'] = yamldata['final_params_mu']
            #
            data_item['init_params_std'] = yamldata['init_params_std']
            data_item['final_params_std'] = yamldata['final_params_std']
            #
            data_item['init_params_l2'] = yamldata['init_params_l2']
            data_item['final_params_l2'] = yamldata['final_params_l2']
        ##
        return data_item

class Collate_fn_onehot_general_features(object):
    '''
    Custom collate function that gets onehot representation for Arch blocks
    and gets general features for the rest. General features are such that they
    are useful for any optimizer amnd initialization. e.g.
    Optimizer might be anything (even a RNN itself) so having symbolic representation for this
    even if its in onehot form isn't general (specially if a new unknown optimzer is used that the model has never seen).
    Thus its better to use the training/validation statistics during training (say the first 10).
    Similarly for initialization (or final weights). If we use the actual weights
    then we don't need a symbolic representation for the initialization algorithm.
    For (space) efficiency reasons we only use statistics of the initial (and final)
    weights. Mean, Std and L2 of the weights.

    Custom collate function to return everything in per batch as follow:
    - OneHot representation for symbols
    - Arch representation concate of OneHot for Arch and Arch hyperparams [A;A_hp]
    - Opt representation train history
    - Net stats representation
    '''

    def __init__(self, device, batch_first, vocab, padding_value=-1):
        '''

        NOTE: padding_value is -1 so to not get confused with 0 which stands for special characers (TODO: check this implemented correctly)
        '''
        self.device = device
        self.batch_first = batch_first
        self.data_processor = DataProcessor(vocab)
        self.padding_value = padding_value

    def arch2OneHot(self, indicies):
        '''
        Maps indices in the batch to tensor OneHot representation
        '''
        vocab_size = len(self.data_processor.vocab.architecture_vocab)
        return torch.Tensor(indices2onehot(indicies, vocab_size)).to(self.device)

    def opt2OneHot(self, indicies):
        '''
        Maps optimizer indicies in the batch to tensor OneHot representation
        '''
        vocab_size = len(self.data_processor.vocab.optimizer_vocab)
        return torch.Tensor(indices2onehot(indicies, vocab_size)).to(self.device)

    def Tensor(self, t):
        '''
        Maps to torch tensor + proper device (cpu or gpu)
        '''
        return torch.Tensor(t).to(self.device)

    def __call__(self, batch):
        '''
        Gets the batch in dictionary foorm ready to be processed by a NN (i.e. its a proper tensor)

        :param list batch: list of samples in a batch. Samples produced by Dataset, which is a dictionary with all the raw data of a data point model.

        :return torch.Tensor batch_arch_rep: OneHot for each layer type (batch_size, max_len, vocab_size)
        :return torch.Tensor arch_lengths: lenghts of each sequence in batch (i.e. # layers for each sample in the batch) (batch_size)
        :return torch.Tensor arch_mask: mask with 0 zeros on padding 1 elsewhere (batch_size, max_len, vocab_size)

        :return torch.Tensor batch_arch_hp_rep: vector form for arch hp (batch_size, max_len, vocab_size)
        :return torch.Tensor arch_hp_lengths: lenghts of each sequence in batch (i.e. # layers for each sample in the batch) (batch_size)
        :return torch.Tensor arch_hp_mask: mask with 0 zeros on padding 1 elsewhere (batch_size, max_len, vocab_size)

        :return torch.Tensor batch_opt: OneHot for which optimizer was used (batch_size, vocab_size)
        :returned torch.Tensor batch_opt_hp: vector form for opt hp (batch_size, vocab_size)

        :return torch.Tensor batch_W_init_rep: tensor with mean and std for each weight in the sequence. (batch_size, max_len, 2)
        :return torch.Tensor W_init_lengths: lenghts of each sequence in batch (i.e. # layers for each sample in the batch) (batch_size)
        :return torch.Tensor W_init_mask: mask with 0 zeros on padding 1 elsewhere (batch_size, max_len, vocab_size)

        :return torch.Tensor batch_W_final_rep: tensor with mean and std for each weight in the sequence. (batch_size, max_len, 2)
        :return torch.Tensor W_final_lengths: lenghts of each sequence in batch (i.e. # layers for each sample in the batch) (batch_size)
        :return torch.Tensor W_final_mask: mask with 0 zeros on padding 1 elsewhere (batch_size, max_len, vocab_size)

        :return torch.Tensor batch_train_errorr: tensor with train errors for each sample in the batch (batch_size)
        '''
        all_batch_info = {}
        ##
        batch_mdl_str = [ sample['mdl_str'] for sample in batch ]
        batch_mdl_str = {'mdl_str':batch_mdl_str}
        ## get arch representation, A
        batch_arch_rep, arch_lengths, arch_mask = self.get_arch_rep(batch)
        arch = {'batch_arch_rep':batch_arch_rep, 'arch_lengths':arch_lengths, 'arch_mask':arch_mask}
        ## get arch hyper param representation, Ahp
        batch_arch_hp_rep, arch_hp_lengths, arch_hp_mask = self.get_arch_hp_rep(batch)
        arch_hp ={'batch_arch_hp_rep':batch_arch_hp_rep, 'arch_hp_lengths':arch_hp_lengths, 'arch_hp_mask':arch_hp_mask}
        ## get opt representation, O
        # batch_opt = self.get_opt_rep(batch)
        # opt = {'batch_opt':batch_opt}
        ## get opt hp, Ohp
        # batch_opt_hp = self.get_opt_hp_rep(batch)
        # opt_hp = {'batch_opt_hp':batch_opt_hp}
        train_history, val_history = self.get_training_validation_history(batch)
        opt, opt_hp = {'train_history':train_history}, {'val_history':val_history}
        ## get W representation
        weight_stats = self.get_all_weight_stats(batch)
        ## get train errors for models
        batch_train_errorr = self.Tensor([ float(sample['train_error']) for sample in batch ])
        train_error = {'batch_train_error':batch_train_errorr}
        ##
        batch_test_errorr = self.Tensor([ float(sample['test_error']) for sample in batch ])
        #test_error = {'batch_test_error':batch_test_errorr}
        test_error = batch_test_errorr
        ## collect return batch
        new_batch = ({**batch_mdl_str, **arch, **arch_hp, **opt, **opt_hp, **weight_stats, **train_error}, test_error)
        #print(new_batch['train_history'])
        return new_batch
        #return batch_arch_rep, batch_arch_hp_rep, batch_opt, batch_W_init, batch_W_final, batch_train_errorr

    def get_arch_rep(self, batch):
        '''
        Converts archictecture indicies to OneHot.

        :param list batch: list of samples in a batch (in dictionary form)
        :return torch.Tensor batch_arch_rep: OneHot for each layer type (batch_size, max_len, vocab_size)
        :return torch.Tensor arch_lengths: lenghts of each sequence in batch (i.e. # layers for each sample in the batch) (batch_size)
        :return torch.Tensor arch_mask: mask with 0 zeros on padding 1 elsewhere (batch_size, max_len, vocab_size)
        '''
        ## get lengths of sequences for each sample in the batch
        arch_lengths = self.Tensor([ len(sample['arch_indices']) for sample in batch ]).long()
        ## make array of one hot tensors for each example in batch
        batch = [ self.arch2OneHot(sample['arch_indices']) for sample in batch ]
        ## padd (and concatenate) the tensors in the whole batch
        batch_arch_rep = torch.nn.utils.rnn.pad_sequence(batch, batch_first=self.batch_first, padding_value=self.padding_value)
        ## compute mask
        arch_mask = (batch_arch_rep != self.padding_value)
        ##
        return batch_arch_rep.to(self.device), arch_lengths.to(self.device), arch_mask.to(self.device)

    def get_arch_hp_rep(self, batch):
        '''
        Converts architecture hyperparams to tensor form (not OneHot, just stacks values)

        :param list batch: list of samples in a batch (in dictionary form)
        :return torch.Tensor batch_arch_hp_rep: vector form for arch hp (batch_size, max_len, vocab_size)
        :return torch.Tensor arch_hp_lengths: lenghts of each sequence in batch (i.e. # layers for each sample in the batch) (batch_size)
        :return torch.Tensor arch_hp_mask: mask with 0 zeros on padding 1 elsewhere (batch_size, max_len, vocab_size)
        '''
        ## get lengths of sequences for each sample in the batch
        arch_hp_lengths = self.Tensor([ len(sample['arch_hp']) for sample in batch ]).long()
        ## padd
        batch = [ self.Tensor(sample['arch_hp']) for sample in batch ]
        batch_arch_hp_rep = torch.nn.utils.rnn.pad_sequence(batch, batch_first=self.batch_first, padding_value=self.padding_value)
        ## compute mask
        arch_hp_mask = (batch_arch_hp_rep != self.padding_value)
        ##
        return batch_arch_hp_rep.to(self.device), arch_hp_lengths.to(self.device), arch_hp_mask.to(self.device)

    def get_opt_rep(self, batch):
        '''
        Get OneHot for optimizer.

        :param list batch: list of samples in a batch. Samples produced by Dataset, which is a dictionary with all the raw data of a data point model.
        :return torch.Tensor batch_opt: OneHot for which optimizer was used (batch_size, vocab_size)
        '''
        batch = [ self.opt2OneHot(sample['opt_indices']) for sample in batch ]
        batch_opt = torch.cat(batch,dim=0)
        return batch_opt.to(self.device)

    def get_opt_hp_rep(self, batch):
        '''
        Converts optimizer hyperparams to tensor form (not OneHot, just stacks values)

        :param list batch: list of samples in a batch. Samples produced by Dataset, which is a dictionary with all the raw data of a data point model.
        :returned torch.Tensor batch_opt_hp: vector form for opt hp (batch_size, vocab_size)
        '''
        batch = [ self.Tensor(sample['opt_hp']) for sample in batch ]
        batch_opt_hp = torch.cat(batch, dim=0)
        return batch_opt_hp.to(self.device)

    def get_training_validation_history(self,batch):
        Tensor = torch.Tensor
        train_history_batch = []
        val_history_batch = []
        for sample in batch:
            ##
            train_errors, train_losses = Tensor(sample['train_errors']), Tensor(sample['train_losses'])
            train = torch.stack((train_errors,train_losses)).t() # (2,seq_len)
            #train = train.unsqueeze(2) # so that convolution layers can take it (2,seq_len,1)
            train_history_batch.append(train)
            ##
            val_errors, val_losses = Tensor(sample['val_errors']), Tensor(sample['val_losses'])
            val = torch.stack((val_errors,val_losses)).t() # (2,seq_len)
            #val = val.unsqueeze(2) # so that convolution layers can take it (2,seq_len,1)
            val_history_batch.append(val)

        ##
        train_history_batch = torch.nn.utils.rnn.pad_sequence(train_history_batch, batch_first=self.batch_first, padding_value=self.padding_value)
        val_history_batch = torch.nn.utils.rnn.pad_sequence(val_history_batch, batch_first=self.batch_first, padding_value=self.padding_value)
        print(f'val_history_batch = {val_history_batch.size()}')
        if self.batch_first:
            train_history_batch = train_history_batch.transpose(1,2)
            val_history_batch = val_history_batch.transpose(1,2)
        else:
            train_history_batch = train_history_batch.transpose(0,2)
            val_history_batch = val_history_batch.transpose(0,2)
        return train_history_batch.to(self.device), val_history_batch.to(self.device)

    def get_all_weight_stats(self, batch):
        '''

        :param list batch: list of samples in a batch. Samples produced by Dataset, which is a dictionary with all the raw data of a data point model.

        :return torch.Tensor batch_W_rep: tensor with mean and std for each weight in the sequence. (batch_size, max_len, 2)
        :return torch.Tensor W_lengths: lenghts of each sequence in batch (i.e. # layers for each sample in the batch) (batch_size)
        :return torch.Tensor W_mask: mask with 0 zeros on padding 1 elsewhere (batch_size, max_len, vocab_size)
        '''
        weight_stats = {}
        with torch.no_grad():
            ##
            batch_init_params_mu_rep, init_params_mu_lengths, init_params_mu_mask = self.get_weight_stat(batch,'init_params_mu')
            batch_final_params_mu_rep, final_params_mu_lengths, final_params_mu_mask = self.get_weight_stat(batch,'final_params_mu')
            new_weights_stats_init = {'batch_init_params_mu_rep':batch_init_params_mu_rep,'init_params_mu_lengths':init_params_mu_lengths, 'init_params_mu_mask':init_params_mu_mask}
            new_weights_stats_final = {'batch_final_params_mu_rep':batch_final_params_mu_rep,'final_params_mu_lengths':final_params_mu_lengths,'final_params_mu_mask':final_params_mu_mask}
            weight_stats = dict(weight_stats, **new_weights_stats_init)
            weight_stats = dict(weight_stats, **new_weights_stats_final)
            ##
            batch_init_params_std_rep, init_params_std_lengths, init_params_std_mask = self.get_weight_stat(batch,'init_params_std')
            batch_final_params_std_rep, final_params_std_lengths, final_params_std_mask = self.get_weight_stat(batch,'final_params_std')
            new_weights_stats_init = {'batch_init_params_std_rep':batch_init_params_std_rep,'init_params_std_lengths':init_params_std_lengths, 'init_params_std_mask':init_params_std_mask}
            new_weights_stats_final = {'batch_final_params_std_rep':batch_final_params_std_rep,'final_params_std_lengths':final_params_std_lengths,'final_params_std_mask':final_params_std_mask}
            weight_stats = dict(weight_stats, **new_weights_stats_init)
            weight_stats = dict(weight_stats, **new_weights_stats_final)
            ##
            batch_init_params_l2_rep, init_params_l2_lengths, init_params_l2_mask = self.get_weight_stat(batch,'init_params_l2')
            batch_final_params_l2_rep, final_params_l2_lengths, final_params_l2_mask = self.get_weight_stat(batch,'final_params_l2')
            new_weights_stats_init = {'batch_init_params_l2_rep':batch_init_params_l2_rep,'init_params_l2_lengths':init_params_l2_lengths, 'init_params_l2_mask':init_params_l2_mask}
            new_weights_stats_final = {'batch_final_params_l2_rep':batch_final_params_l2_rep,'final_params_l2_lengths':final_params_l2_lengths,'final_params_l2_mask':final_params_l2_mask}
            weight_stats = dict(weight_stats, **new_weights_stats_init)
            weight_stats = dict(weight_stats, **new_weights_stats_final)
            ##
            return weight_stats

    def get_weight_stat(self, batch, W_type):
        ## get lengths of sequences for each sample in the batch
        weight_lengths = self.Tensor([ len(sample[W_type]) for sample in batch ]).long()
        ## padd
        #st()
        new_batch = []
        for i,sample in enumerate(batch):
            try:
                print(f'i = {i}')
                print(f'sample = {sample}')
                tensor_sample = self.Tensor(sample[W_type])
                print(f'tensor_sample = {tensor_sample}')
                new_batch.append(tensor_sample)
            except:
                print(f'\n ---- ERROR: i = {i}')
                print(f'sample = {sample}')
                st()
        ## padd batch sequences
        batch_weight_rep = torch.nn.utils.rnn.pad_sequence(new_batch, batch_first=self.batch_first, padding_value=self.padding_value)
        ## compute mask
        weight_mask = (batch_weight_rep != self.padding_value)
        ##
        return batch_weight_rep.to(self.device), weight_lengths.to(self.device), weight_mask.to(self.device)

def testing():
    pass

if __name__ == '__main__':
    testing()
