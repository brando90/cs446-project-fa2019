'''
Utils class with useful helper functions

utils: https://www.quora.com/What-do-utils-files-tend-to-be-in-computer-programming-documentation
'''

import time
import os
import math

import os

import time
import numpy as np

import torch

import socket

from email.message import EmailMessage
import smtplib
import os

from pdb import set_trace as st

def HelloWorld():
    return 'HelloWorld'

def remove_folders_recursively(path):
    print('WARNING: HAS NOT BEEN TESTED')
    path.expanduser()
    try:
        shutil.rmtree(str(path))
    except OSError:
        # do nothing if removing fails
        pass

def oslist_for_path(path):
    return [f for f in path.iterdir() if f.is_dir()]

def make_and_check_dir(path):
    '''
    tries to make dir/file, if it exists already does nothing else creates it.

    https://docs.python.org/3/library/pathlib.html

    :param path object path: path where the data will be saved
    '''
    path = os.path.expanduser(path)
    print(path)
    st()
    try:
        os.makedirs(path)
    except OSError:
        print(OSError)
        return OSError
        pass

def timeSince(start):
    '''
    How much time has passed since the time "start"

    :param float start: the number representing start (usually time.time())
    '''
    now = time.time()
    s = now - start
    ## compute how long it took in hours
    h = s/3600
    ## compute numbers only for displaying how long in took
    m = math.floor(s / 60) # compute amount of whole integer minutes it took
    s -= m * 60 # compute how much time remaining time was spent in seconds
    ##
    msg = f'time passed: hours:{h}, minutes={m}, seconds={s}'
    return msg, h

def report_times(start, verbose=False):
    '''
    How much time has passed since the time "start"

    :param float start: the number representing start (usually time.time())
    '''
    meta_str=''
    ## REPORT TIMES
    start_time = start
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    if verbose:
        print(f"--- {seconds} {'seconds '+meta_str} ---")
        print(f"--- {minutes} {'minutes '+meta_str} ---")
        print(f"--- {hours} {'hours '+meta_str} ---")
        print('\a')
    ##
    msg = f'time passed: hours:{hours}, minutes={minutes}, seconds={seconds}'
    return msg, seconds, minutes, hours

def is_NaN(value):
    '''
    Checks is value is problematic by checking if the value:
    is not finite, is infinite or is already NaN
    '''
    return not np.isfinite(value) or np.isinf(value) or np.isnan(value)

##

def make_and_check_dir(path):
    '''
        tries to make dir/file, if it exists already does nothing else creates it.
    '''
    try:
        os.makedirs(path)
    except OSError:
        pass

####

def save_pytorch_mdl(path_to_save,net):
    ##http://pytorch.org/docs/master/notes/serialization.html
    ##The first (recommended) saves and loads only the model parameters:
    torch.save(net.state_dict(), path_to_save)

def restore_mdl(path_to_save,mdl_class):
    # TODO
    the_model = TheModelClass(*args, **kwargs)
    the_model.load_state_dict(torch.load(PATH))

def save_entire_mdl(path_to_save,the_model):
    torch.save(the_model, path_to_save)

def restore_entire_mdl(path_to_restore):
    '''
    NOTE: However in this case, the serialized data is bound to the specific
    classes and the exact directory structure used,
    so it can break in various ways when used in other projects, or after some serious refactors.
    '''
    the_model = torch.load(path_to_restore)
    return the_model

def get_hostname():
    hostname = socket.gethostname()
    if 'polestar-old' in hostname or hostname=='gpu-16' or hostname=='gpu-17':
        return 'polestar-old'
    elif 'openmind' in hostname:
        return 'OM'
    else:
        return hostname

def count_nb_params(net):
    count = 0
    for p in net.parameters():
        count += p.data.nelement()
    return count

####

'''
Greater than 4 I get this error:

ValueError: Seed must be between 0 and 2**32 - 1
'''

RAND_SIZE = 4

def get_random_seed():
    '''

    source: https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
    '''
    random_data = os.urandom(RAND_SIZE) # Return a string of size random bytes suitable for cryptographic use.
    random_seed = int.from_bytes(random_data, byteorder="big")
    return random_seed

def seed_everything(seed=42):
    '''
    https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

####

def send_email(message, destination):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    # not a real email account nor password, its all ok!
    server.login('slurm.miranda@gmail.com', 'dummy123!@#$321')

    ## SLURM Job_id=374_* (374) Name=flatness_expts.py Ended, Run time 10:19:54, COMPLETED, ExitCode [0-0]
    msg = EmailMessage()
    msg.set_content(message)

    msg['Subject'] = get_hostname()
    msg['From'] = 'slurm.miranda@gmail.com'
    msg['To'] = destination
    server.send_message(msg)

if __name__ == '__main__':
    send_email('msg','miranda9@illinois.edu')
