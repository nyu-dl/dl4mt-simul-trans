"""
Configuration for Simultaneous Neural Machine Translation
"""

def pretrain_config():

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



