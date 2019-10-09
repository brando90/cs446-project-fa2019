import torch
import torch.nn as nn

#from torch.nn.functional import softmax as softmax
from torch.nn.functional import sigmoid as sigmoid

from pathlib import Path

from predicting_performance.data_processor import DataProcessor, Vocab
from predicting_performance.data_processor import get_type, MetaLearningDataset
from predicting_performance.data_processor import Collate_fn_onehot_general_features
from predicting_performance.trainer import Trainer

from predicting_performance.stats_collector import StatsCollector

from pdb import set_trace as st

def get_init_hidden(batch_size, hidden_size, n_layers, bidirectional, device):
    '''
    Gets initial hidden states for all cells depending on # batches, nb_layers, directions, etc.

    Details:
    We have to have a hidden initial state of size (hidden_size) for:
    - each sequence in the X_batch
    - each direction the RNN process the sequence
    - each layer of the RNN (note we are stacking RNNs not mini-NN layers)

    NOTE: notice that we don't have seq_len anywhere because the first hidden
    state is only needed to start the computation

    :param int batch_size: size of batch
    :return torch.Tensor hidden: initial hidden state (n_layers*nb_directions, batch_size, hidden_size)
    '''
    nb_directions = 2 if bidirectional else 1
    h_n = torch.randn(n_layers*nb_directions, batch_size, hidden_size).to(device)
    c_n = torch.randn(n_layers*nb_directions, batch_size, hidden_size).to(device)
    hidden = (h_n, c_n)
    return hidden

class ChainLSTM(nn.Module):

    def __init__(self,
    arch_input_size, arch_hidden_size, arch_num_layers,
    arch_hp_input_size, arch_hp_hidden_size, arch_hp_num_layers,
    opt_input_size, opt_hidden_size, opt_num_layers,
    weight_stats_input_size, weight_stats_hidden_size, weight_stats_layers,
    train_err_input_size, train_err_hidden_size, num_layers_num_layers,
    bidirectional=False, batch_first=True, device="cpu"):
        '''
        '''
        super().__init__()
        
        self.device = device
        ##
        self.batch_first = batch_first #If True, then the input and output tensors are provided as (batch, seq, feature).
        ## LSTM unit for processing the Architecture
        self.arch = nn.LSTM(input_size=arch_input_size,
                            hidden_size=arch_hidden_size,
                            num_layers=arch_num_layers,
                            batch_first=batch_first).to(device)
        ## LSTM unit for processing the Architecture HyperParams
        self.arch_hp = nn.LSTM(input_size=arch_hp_input_size,
                            hidden_size=arch_hp_hidden_size,
                            num_layers=arch_num_layers,
                            batch_first=batch_first).to(device)
        ## LSTM for processing raw features related to Optimization
        self.opt = nn.LSTM(input_size=opt_input_size,
                            hidden_size=opt_hidden_size,
                            num_layers=opt_num_layers,
                            batch_first=batch_first).to(device)
        # ## LSTM for processing raw features related to init and final weights
        self.weight_stats = nn.LSTM(input_size=weight_stats_input_size,
                            hidden_size=weight_stats_hidden_size,
                            num_layers=weight_stats_layers,
                            batch_first=batch_first).to(device)
        # ## LSTM for processing train error
        self.train_err = nn.LSTM(input_size=train_err_input_size,
                            hidden_size=train_err_hidden_size,
                            num_layers=num_layers_num_layers,
                            batch_first=batch_first).to(device)
        self.predict_performance = nn.Linear(in_features=train_err_hidden_size,out_features=1).to(device)
        ##


    def forward(self, input):
        '''
        '''
        print()
        ##
        batch_first = self.batch_first
        batch_size = input['batch_arch_rep'].size(0)
        ##
        h_a, c_a = get_init_hidden(batch_size, self.arch.hidden_size, self.arch.num_layers, self.arch.bidirectional, self.device)
        ## forward pass through Arch
        arch_lengths = input['arch_lengths']
        batch_arch_rep = input['batch_arch_rep'] # (batch_size,max_seq_len,dim) e.g. torch.Size([3, 6, 12])
        # print()
        # print(f'arch_lengths = {arch_lengths}')
        print(f'batch_arch_rep.size() = {batch_arch_rep.size()}')
        batch_arch_rep = nn.utils.rnn.pack_padded_sequence(batch_arch_rep, arch_lengths, batch_first=batch_first, enforce_sorted=False)
        out, (h_a, c_a) = self.arch(input=batch_arch_rep, hx=(h_a, c_a)) # lstm
        ## forward pass through Arch
        arch_hp_lengths = input['arch_hp_lengths']
        batch_arch_hp_rep = input['batch_arch_hp_rep']
        batch_arch_hp_rep = nn.utils.rnn.pack_padded_sequence(batch_arch_hp_rep, arch_hp_lengths, batch_first=batch_first, enforce_sorted=False)
        out, (h_a, c_a) = self.arch_hp(input=batch_arch_hp_rep, hx=(h_a, c_a)) # lstm
        ## forward pass through Opt (training/val stats)
        train_history = input['train_history']
        val_history = input['val_history']
        history = torch.cat((train_history,val_history),dim=1)
        out, (h_a, c_a) = self.opt(input=history, hx=(h_a, c_a)) # lstm
        ## forward pass through Weight stats
        batch_init_params_mu_rep, init_params_mu_lengths = input['batch_init_params_mu_rep'], input['init_params_mu_lengths']
        batch_init_params_std_rep, _ = input['batch_init_params_std_rep'], input['init_params_std_lengths']
        batch_init_params_l2_rep, _ = input['batch_init_params_l2_rep'], input['init_params_l2_lengths']
        print(f'\nbatch_init_params_mu_rep = {batch_init_params_mu_rep.size()}')
        print(f'init_params_mu_lengths = {init_params_mu_lengths}')
        weight_stats = torch.stack((batch_init_params_mu_rep,batch_init_params_std_rep,batch_init_params_l2_rep),dim=2)
        weight_lengths = init_params_mu_lengths # note you only need 1 of them lengths array cuz they all have the same shapes since they are stats from the same arch for each sample in the batch
        print(f'weight_stats.size() = {weight_stats.size()}')
        print(f'weight_lengths = {weight_lengths}')
        weight_stats = nn.utils.rnn.pack_padded_sequence(weight_stats, weight_lengths, batch_first=batch_first, enforce_sorted=False)
        out, (h_a, c_a) = self.weight_stats(input=weight_stats, hx=(h_a, c_a)) # lstm
        ## forwad pass through train error
        batch_train_error = input['batch_train_error'].view(batch_size,1,1)
        out, (h_a, c_a) = self.train_err(input=batch_train_error, hx=(h_a, c_a)) # lstm
        score = self.predict_performance(out).squeeze()
        print(f'score = {score}')
        test_error_prediction = torch.sigmoid(score)
        print(f'test_error_prediction = {test_error_prediction}')
        ##
        return test_error_prediction

def main():
    '''
    '''
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    ## paths to automl data set
    data_path = '~/predicting_generalization/automl/data/automl_dataset_debug'
    path = Path(data_path).expanduser()
    ## Vocab
    vocab = Vocab()
    V_a, V_hp = len(vocab.architecture_vocab), len(vocab.hparms_vocab)
    ## create dataloader for meta learning data set
    batch_first = True
    dataset = MetaLearningDataset(data_path, vocab)
    collate_fn = Collate_fn_onehot_general_features(device, batch_first, vocab)
    batch_size = 3
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    ## instantiate Meta Learner
    # arch hps
    arch_input_size = V_a
    arch_hidden_size = 16
    arch_num_layers = 1
    # arch_hp hps
    arch_hp_input_size = V_hp
    arch_hp_hidden_size = arch_hidden_size
    arch_hp_num_layers = 1
    # opt hps
    # st()
    # input1 = dataset[0]['train_history'].view(batch_size,-1)
    # input2 = dataset[0]['val_history'].view(batch_size,-1)
    # st()
    seq_len = len(dataset[0]['test_errors']) # since they all have the same seq_len
    input_dim = 4 # 4 is cuz we have 2 losses, errors CSEloss but we have val and train so 2*2=4
    opt_input_size = input_dim # so that it process one time step of the history at a time [train_err,train_loss,val_loss,val_err]
    opt_hidden_size = arch_hidden_size
    opt_num_layers = 1
    # weight stats
    weight_stats_input_size = 3 # 3 because we are only processing init params stats mu, std l2. if we also process all final param stats this would be 6
    weight_stats_hidden_size = arch_hidden_size
    weight_stats_layers = 1
    ## train error hps
    train_err_input_size = 1
    train_err_hidden_size = arch_hidden_size
    num_layers_num_layers = 1
    # meta-learner chain lstm
    meta_learner = ChainLSTM(arch_input_size=arch_input_size, arch_hidden_size=arch_hidden_size, arch_num_layers=1,
        arch_hp_input_size=arch_hp_input_size, arch_hp_hidden_size=arch_hp_hidden_size, arch_hp_num_layers=1,
        weight_stats_input_size=weight_stats_input_size, weight_stats_hidden_size=weight_stats_hidden_size, weight_stats_layers=weight_stats_layers,
        opt_input_size=opt_input_size, opt_hidden_size=opt_hidden_size, opt_num_layers=opt_num_layers,
        train_err_input_size = train_err_input_size, train_err_hidden_size = train_err_hidden_size, num_layers_num_layers = num_layers_num_layers,
        device=device
    )
    ##
    trainloader, valloader, testloader = dataloader, dataloader, dataloader # TODO this is just for the sake of an example!
    optimizer = torch.optim.Adam(meta_learner.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1.0)
    criterion = torch.nn.MSELoss()
    error_criterion = criterion # TODO: implement epsilon classification loss
    stats_collector = StatsCollector()
    device = device
    trainer = Trainer(trainloader, valloader, testloader,
        optimizer, scheduler, criterion, error_criterion,
        stats_collector,
        device)
    ##
    nb_epochs = 2
    train_iterations = 4 # TODO: CHANGE for model to be fully trained!!!
    trainer.train_and_track_stats(meta_learner, nb_epochs, iterations=4, train_iterations=4)

if __name__ == '__main__':
    main()
