import unittest

import shutil
import os
from pathlib import Path

from utils.utils import HelloWorld
from utils.utils import make_and_check_dir

from predicting_performance.data_generators.debug_model_gen import create_debug_data_set
from predicting_performance.data_point_models.debug_models import get_debug_models
from predicting_performance.data_loader_cifar import get_cifar10_for_data_point_mdl_gen

from pdb import set_trace as st

class TestStringMethods(unittest.TestCase):

    def test_pkg_import(self):
        helloworld = HelloWorld()
        self.assertEqual(helloworld,'HelloWorld')

    def test_download_cifar_put_in_right_folder(self):
        data_path = '~/predicting_generalization/automl/data/cifar-10-batches-py' # note you can use: os.path.expanduser
        path = Path(data_path).expanduser()
        print(f'---> {str(path)}')
        ## try to remove it, if there is nothing to remove then do nothing
        try:
            shutil.rmtree(str(path))
        except OSError:
            # do nothing if removing fails
            pass
        ## check if indeed we removed the cifar data set properly
        cifar_folder_exists = path.exists()
        self.assertEqual(cifar_folder_exists, False)
        ##
        trainloader, valloader, testloader = get_cifar10_for_data_point_mdl_gen()
        cifar_folder_exists = path.exists()
        self.assertEqual(cifar_folder_exists, True)
        for i,data in enumerate(trainloader,0):
            inputs,targets = data
            if i >= 2:
                break
            print(f'batch_size = {inputs.size(0)}')
            print(inputs.size())
        ##
        cifar_folder_exists = path.exists()
        self.assertEqual(cifar_folder_exists, True)
        ##
        files_in_cifar10_folder = 8
        all_f = [f for f in path.iterdir()]
        #empty_model_folders = [f for f in path.iterdir() if f.is_dir()]
        self.assertEqual(len(all_f), files_in_cifar10_folder)

    def test_create_debug_data_set(self):
        '''
        tests that we can create a debug data set
        '''
        print('---> Running: unittest.test_create_debug_data_set')
        ## make sure debug data set is clean
        data_path = '~/predicting_generalization/automl/data/automl_dataset_debug' # note you can use: os.path.expanduser
        path = Path(data_path).expanduser()
        ## try to remove it, if there is nothing to remove then do nothing
        try:
            shutil.rmtree(str(path))
        except OSError:
            # do nothing if removing fails
            pass
        ## check it was removed
        debug_data_set_exists = path.exists()
        self.assertEqual(debug_data_set_exists, False)
        ## since we definitively removed the debug data set folder, make it (or throw an error if it exists, shouldn't exists!)
        path.mkdir()
        ## check there is no data point model (i.e. no mdl folders)
        empty_model_folders = [f for f in path.iterdir() if f.is_dir()] # listdir
        self.assertEqual(len(empty_model_folders), 0)
        ## check dataset is of the right size
        create_debug_data_set()
        ## check it created the right amount of models
        mdls = get_debug_models()
        model_folders = [f for f in path.iterdir() if f.is_dir()]
        self.assertEqual(len(model_folders), len(mdls))
        print(f'len(model_folders), len(mdls) = {len(model_folders), len(mdls)}')


if __name__ == '__main__':
    unittest.main()
    print('Done\a')
