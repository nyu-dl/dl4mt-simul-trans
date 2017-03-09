"""
Configuration for Simultaneous Neural Machine Translation
"""
from collections import OrderedDict

# data_home  = '/home/thoma/scratch/un16/'
# model_home = '/home/thoma/scratch/simul/'
# data_home  = '/mnt/scratch/un16/'
# model_home = '/mnt/scratch/simul/'

data_home   = '/misc/kcgscratch1/ChoGroup/thoma_data/simul_trans/un16/'
model_home  = '/misc/kcgscratch1/ChoGroup/thoma_data/simul_trans/'


def pretrain_config():

    """Configuration for pretraining underlining NMT model."""

    config = dict()

    # training set (source, target)
    config['datasets'] = [data_home + 'train.un16.en-zh.zh.c0.tok.clean.bpe20k.np',
                          data_home + 'train.un16.en-zh.en.c0.tok.clean.bpe20k.np']

    # validation set (source, target)
    config['valid_datasets'] = [data_home + 'devset.un16.en-zh.zh.c0.tok.bpe20k.np',
                                data_home + 'devset.un16.en-zh.en.c0.tok.bpe20k.np']

    # vocabulary (source, target)
    config['dictionaries']   = [data_home + 'train.un16.en-zh.zh.c0.tok.clean.bpe20k.vocab.pkl',
                                data_home + 'train.un16.en-zh.en.c0.tok.clean.bpe20k.vocab.pkl']

    # save the model to
    config['saveto']      = data_home + 'pretraining/model_un16_bpe2k_uni_zh-en.npz'
    config['reload_']     = True

    # model details
    config['dim_word']    = 512
    config['dim']         = 1028
    config['n_words']     = 20000
    config['n_words_src'] = 20000

    # learning details
    config['decay_c']     = 0
    config['clip_c']      = 1.
    config['use_dropout'] = False
    config['lrate']       = 0.0001
    config['optimizer']   = 'adadelta'
    config['patience']    = 1000
    config['maxlen']      = 50
    config['batch_size']  = 32
    config['valid_batch_size'] =  64
    config['validFreq']   = 1000
    config['dispFreq']    = 50
    config['saveFreq']    = 1000
    config['sampleFreq']  = 99

    return config


def rl_config():
    """Configuration for training the agent using REINFORCE algorithm."""

    config = OrderedDict()  # general configuration

    # work-space
    config['workspace'] = model_home

    # training set (source, target); or leave it None, agent will use the same corpus saved in the model
    config['datasets'] = [data_home + 'train.un16.en-zh.en.c0.tok.clean.bpe20k.np',
                          data_home + 'train.un16.en-zh.zh.c0.tok.clean.bpe20k.np']

    # validation set (source, target); or leave it None, agent will use the same corpus saved in the model
    config['valid_datasets'] = [data_home + 'devset.un16.en-zh.en.c0.tok.bpe20k.np',
                                data_home + 'devset.un16.en-zh.zh.c0.tok.bpe20k.np']

    # vocabulary (source, target); or leave it None, agent will use the same dictionary saved in the model
    config['dictionaries']   = [data_home + 'train.un16.en-zh.en.c0.tok.clean.bpe20k.vocab.pkl',
                                data_home + 'train.un16.en-zh.zh.c0.tok.clean.bpe20k.vocab.pkl']

    # pretrained model
    config['model']  = model_home + '.pretrained/model_un16_bpe2k_uni_en-zh.npz'
    config['option'] = model_home + '.pretrained/model_un16_bpe2k_uni_en-zh.npz.pkl'

    # critical training parameters.
    config['sample']    = 10
    config['batchsize'] = 10
    config['rl_maxlen'] = 100
    config['target_ap'] = 0.8   # 0.75  # target delay if using AP as reward.
    config['target_cw'] = 8     # if cw > 0 use cw mode

    # under-construction
    config['forget']    = False

    # learning rate
    config['lr_policy'] = 0.0002
    config['lr_model']  = 0.00002

    # policy parameters
    config['prop']      = 0.5   # leave it default
    config['recurrent'] = True  # use a recurrent agent
    config['layernorm'] = False # layer normalalization for the GRU agent.
    config['updater']   = 'REINFORCE'  # 'TRPO' not work well.
    config['act_mask']  = True  # leave it default

    # old model parameters (maybe useless, leave them default)
    config['step']     = 1
    config['peek']     = 1
    config['s0']       = 1
    config['gamma']    = 1
    config['Rtype']    = 10
    config['maxsrc']   = 10
    config['pre']      = False
    config['coverage'] = False
    config['upper']    = False
    config['finetune'] = True
    config['train_gt'] = True   # when training with GT, fix the agent??
    config['full_att'] = False

    return config










