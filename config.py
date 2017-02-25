"""
Configuration for Simultaneous Neural Machine Translation
"""
from collections import OrderedDict

def pretrain_config():
    """Configuration for pretraining underlining NMT model."""

    config = dict()

    # training set (source, target)
    config['datasets'] = ['/misc/kcgscratch1/ChoGroup/thoma_data/un16/train.un16.en-zh.zh.c0.tok.clean.bpe20k.np',
                          '/misc/kcgscratch1/ChoGroup/thoma_data/un16/train.un16.en-zh.en.c0.tok.clean.bpe20k.np']

    # validation set (source, target)
    config['valid_datasets'] = ['/misc/kcgscratch1/ChoGroup/thoma_data/un16/devset.un16.en-zh.zh.c0.tok.bpe20k.np',
                                '/misc/kcgscratch1/ChoGroup/thoma_data/un16/devset.un16.en-zh.en.c0.tok.bpe20k.np']

    # vocabulary (source, target)
    config['dictionaries']   = ['/misc/kcgscratch1/ChoGroup/thoma_data/un16/train.un16.en-zh.zh.c0.tok.clean.bpe20k.vocab.pkl',
                                '/misc/kcgscratch1/ChoGroup/thoma_data/un16/train.un16.en-zh.en.c0.tok.clean.bpe20k.vocab.pkl']

    # save the model to
    config['saveto']      = '.pretraining/model_un16_bpe2k_uni_zh-en.npz'
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
    config['batch_size']  = 64
    config['valid_batch_size'] =  64
    config['validFreq']   = 1000
    config['dispFreq']    = 50
    config['saveFreq']    = 1000
    config['sampleFreq']  = 99

    return config


def rl_config():
    """Configuration for training the agent using REINFORCE algorithm."""

    config = OrderedDict()  # general configuration
    policy = OrderedDict()  # configuration for policy

    # work-space
    config['workspace'] = './'


    # training set (source, target); or leave it None, agent will use the same corpus saved in the model
    config['datasets'] = ['/misc/kcgscratch1/ChoGroup/thoma_data/un16/train.un16.en-zh.en.c0.tok.clean.bpe20k.np',
                          '/misc/kcgscratch1/ChoGroup/thoma_data/un16/train.un16.en-zh.zh.c0.tok.clean.bpe20k.np']

    # validation set (source, target); or leave it None, agent will use the same corpus saved in the model
    config['valid_datasets'] = ['/misc/kcgscratch1/ChoGroup/thoma_data/un16/devset.un16.en-zh.en.c0.tok.bpe20k.np',
                                '/misc/kcgscratch1/ChoGroup/thoma_data/un16/devset.un16.en-zh.zh.c0.tok.bpe20k.np']

    # vocabulary (source, target); or leave it None, agent will use the same dictionary saved in the model
    config['dictionaries']   = ['/misc/kcgscratch1/ChoGroup/thoma_data/un16/train.un16.en-zh.en.c0.tok.clean.bpe20k.vocab.pkl',
                                '/misc/kcgscratch1/ChoGroup/thoma_data/un16/train.un16.en-zh.zh.c0.tok.clean.bpe20k.vocab.pkl']

    # pretrained model
    config['model']  = '.pretrained/model_un16_bpe2k_uni_en-zh.npz'
    config['option'] = '.pretrained/model_un16_bpe2k_uni_en-zh.npz.pkl'

    # critical training parameters.
    config['sample']    = 10
    config['batchsize'] = 10
    config['target_ap'] = 1.0   # 0.75  # target delay if using AP as reward.
    config['target_cw'] = 8     # if cw > 0 use cw mode

    # under-construction
    config['forget']    = False

    # policy parameters
    policy['prop']      = 0.5   # leave it default
    policy['recurrent'] = True  # use a recurrent agent
    policy['layernorm'] = False # layer normalalization for the GRU agent.
    policy['updater']   = 'REINFORCE'  # 'TRPO' not work well.
    policy['act_mask']  = True  # leave it default


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
    config['finetune'] = 'nope'


    return policy, config










