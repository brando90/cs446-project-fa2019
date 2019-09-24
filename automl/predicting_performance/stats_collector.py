import numpy as np

#from maps import NamedDict

import utils

from pdb import set_trace as st

class StatsCollector:
    '''
    Class that has all the stats collected during training.

    check the fields it has to see what stats its tracking.
    '''
    def __init__(self):
        ''' create empty list of loss & errors '''
        self.reset()

    def append_losses_errors_accs(self, train_loss, train_error, val_loss, val_error, test_loss, test_error):
        '''
        Appends the losses and errors for the current epoch for the current model.
        Append -1 if you're not using that dataset/loader (e.g. val_error = -1 if there is no val data set)

        :param number train_loss: saves the train loss in the list of train losses tracked during training.
        '''
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))
        self.test_losses.append(float(test_loss))

        self.train_errors.append(float(train_error))
        self.val_errors.append(float(val_error))
        self.test_errors.append(float(test_error))

        self.train_accs.append(float(1.0-train_error))
        self.val_accs.append(float(1.0-val_error))
        self.test_accs.append(float(1.0-test_error))

    def reset(self):
        '''
        Create empty list of loss & errors
        '''
        self.train_losses, self.val_losses, self.test_losses = [], [], []
        self.train_errors, self.val_errors, self.test_errors = [], [], []
        self.train_accs, self.val_accs, self.test_accs = [], [], []

    def get_stats_dict(self, other_data_to_append={}):
        '''
        Return the dictionary

        :param dict other_data_to_append: other dict data to append
        '''
        ## TODO: loop through fields?
        stats = {'train_losses':self.train_losses, 'val_losses':self.val_losses, 'test_losses':self.test_losses,
            'train_errors':self.train_errors, 'val_errors':self.val_errors, 'test_errors':self.test_errors,
            'train_accs':self.train_accs, 'val_accs':self.val_accs, 'test_accs':self.test_accs}
        stats = {**stats, **other_data_to_append}
        return stats
