import unittest

from pathlib import Path

import numpy as np

import torch

from predicting_performance.data_processor import DataProcessor, Vocab
from predicting_performance.data_processor import get_type, MetaLearningDataset
from predicting_performance.data_processor import Collate_fn_onehot_general_features

import utils.utils as utils

from pdb import set_trace as st

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TestStringMethods(unittest.TestCase):

    def test_pkg_import(self):
        HelloWorld = utils.HelloWorld()
        self.assertEqual(HelloWorld,'HelloWorld')

    def test_get_layer_type_from(self):
        '''
        Makes sure we can extract layer types from strings
        '''
        ##
        vocab = Vocab()
        ##
        D_in = 4
        H = 3
        D_out = 2
        net = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
        )
        net_str = str(net)
        ##
        data = net_str.split(':')[1:]
        answer = ['Linear', 'ReLU', 'Linear']
        for i in range(len(data)):
            layer_type = get_type(vocab.architecture_vocab, data[i])
            self.assertEqual(answer[i], layer_type)

    def test_architecture_onehot(self):
        '''
        tests that we make an architecture to one hot (note: no meta data)
        '''
        #self.assertEqual('foo'.upper(), 'FOO')

    def test_make_full_arch_and_meta_data_input_vec(self):
        '''
        tests that the whole pipeline to make a processable input data from X = [model, meta data]
        '''
        vocab = Vocab()
        data_processor = DataProcessor(vocab)
        vocab_size = len(data_processor.vocab.architecture_vocab)
        hp_vocab_size = len(data_processor.vocab.hparms_vocab)
        ## model to test
        D_in = 4
        H = 3
        D_out = 2
        net = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
        )
        net_str = str(net)
        data = net_str.split(':')[1:] # e.g.
        nb_layers = len(data)
        ## convert model to feature vector
        feature_matrix = data_processor.mdl_str2feature_vec(net_str)
        ##
        for i in range(nb_layers):
            layer_type = get_type(data_processor.vocab.architecture_vocab, data[i] )
            layer_vec = feature_matrix[i,:]
            ##
            idx_layer_type = data_processor.vocab.architecture2idx[layer_type]
            arch_vec = layer_vec[:vocab_size]
            (nonzeros_arch,) = np.nonzero(arch_vec)
            self.assertEqual(idx_layer_type, nonzeros_arch[0])
            ## TODO, write test for this that doesn't totally re-use code we are testing but also is extendable to growing vocabs
            hp_vec = layer_vec[vocab_size:]
            (nonzeros_hp,) = np.nonzero(hp_vec)
        #print(feature_matrix)

    def test_loop_through_data(self):
        ## paths to automl data set
        data_path = '~/predicting_generalization/automl/data/automl_dataset_debug'
        path = Path(data_path).expanduser()
        ## create dataloader for meta learning data set
        vocab = Vocab()
        pred_test_dataset = MetaLearningDataset(data_path, vocab)
        dataloader = torch.utils.data.DataLoader(pred_test_dataset, batch_size=1)
        ## check dataset is of the right size
        self.assertEqual(len(pred_test_dataset),5)
        ## loop through data
        print()
        for data in dataloader:
            data
            st()
            pass
            #print(f'len(data)={len(data)}')
            #self.assertEqual(len(data),8)
        self.assertEqual(True,True)

    def test_loop_through_data(self):
        ## paths to automl data set
        data_path = '~/predicting_generalization/automl/data/automl_dataset_debug'
        path = Path(data_path).expanduser()
        ## create dataloader for meta learning data set
        vocab = Vocab()
        pred_test_dataset = MetaLearningDataset(data_path, vocab)
        batch_first = True
        collate_fn = Collate_fn_onehot_general_features(device, batch_first, vocab)
        batch_size = 1
        dataloader = torch.utils.data.DataLoader(pred_test_dataset, batch_size=batch_size, collate_fn=collate_fn)
        ## check dataset is of the right size
        self.assertEqual(len(pred_test_dataset),5)
        ## loop through data
        print()
        for data in dataloader:
            inputs,targets = data
            batch_size_input = inputs['batch_train_error'].size(0)
            batch_size_target = targets['batch_test_error'].size(0)
            self.assertEqual(batch_size_input, batch_size_target)
            self.assertEqual(batch_size_input, batch_size)
            pass
            #print(f'len(data)={len(data)}')
            #self.assertEqual(len(data),8)
        self.assertEqual(True,True)
        ##
        batch_size = 3
        dataloader = torch.utils.data.DataLoader(pred_test_dataset, batch_size=batch_size, collate_fn=collate_fn)
        ## check dataset is of the right size
        self.assertEqual(len(pred_test_dataset),5)
        ## loop through data
        print()
        for data in dataloader:
            inputs,targets = data
            batch_size_input = inputs['batch_train_error'].size(0)
            batch_size_target = targets['batch_test_error'].size(0)
            self.assertEqual(batch_size_input, batch_size_target)
            pass
            #print(f'len(data)={len(data)}')
            #self.assertEqual(len(data),8)
        self.assertEqual(True,True)

    #
    # def test_special_tokens(self):
    #     '''
    #     '''
    #     print('\n\n')
    #     print('special')
    #     data_processor = DataProcessor()
    #     ## check SOS
    #     SOS = 'SOS'
    #     sos_true_idx = 1
    #     sos_idx = data_processor.tokens2arch_indices(SOS)
    #     sos_tensor = data_processor.indices2arch_onehot(sos_idx)
    #     self.assertEqual(sos_idx, sos_true_idx)
    #     #
    #     (sos_idx_as_np_array,) = (sos_tensor == 1).nonzero()
    #     self.assertEqual(sos_true_idx, int(sos_idx_as_np_array))
    #     ## check EOS
    #     EOS = 'EOS'
    #     eos_true_idx = 2
    #     eos_idx = data_processor.tokens2arch_indices(EOS)
    #     eos_tensor = data_processor.indices2arch_onehot(eos_idx)
    #     self.assertEqual(eos_idx, eos_true_idx)
    #     #
    #     (eos_idx_as_np_array,) = (eos_tensor == 1).nonzero()
    #     self.assertEqual(eos_true_idx, int(eos_idx_as_np_array))

if __name__ == '__main__':
    unittest.main()
