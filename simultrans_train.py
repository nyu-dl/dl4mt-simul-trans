"""
Simultaneous Machine Translateion: Training with Policy Gradient

"""
import argparse
import os
import cPickle as pkl

from bleu import *
from nmt_uni import *
from policy import Controller as Policy
from utils import Progbar, Monitor

from simultrans_model import simultaneous_decoding
from simultrans_model import _seqs2words, _bpe2words, PIPE, _padding

import time

numpy.random.seed(19920206)
timer = time.time

# run training function:: >>>
def run_simultrans(model,
                   options_file=None,
                   config=None,
                   policy=None,
                   id=None,
                   remote=False):

    WORK = config['workspace']

    # check hidden folders
    paths = ['.policy', '.pretrained', '.log', '.config', '.images', '.translate']
    for p in paths:
        p = WORK + p
        if not os.path.exists(p):
            os.mkdir(p)


    if id is not None:
        fcon = WORK + '.config/{}.conf'.format(id)
        if os.path.exists(fcon):
            print 'load config files'
            policy, config = pkl.load(open(fcon, 'r'))

    # ============================================================================== #
    # load model model_options
    # ============================================================================== #
    _model = model.split('/')[-1]

    if options_file is not None:
        with open(options_file, 'rb') as f:
            options = pkl.load(f)
    else:
        with open('%s.pkl' % model, 'rb') as f:
            options = pkl.load(f)

    print 'merge configuration into options'
    for w in config:
        if (w in options) and (config[w] is not None):
            options[w] = config[w]


    print 'load options...'
    for w, p in sorted(options.items(), key=lambda x: x[0]):
        print '{}: {}'.format(w, p)

    # load detail settings from option file:
    dictionary, dictionary_target = options['dictionaries']

    def _iter(fname):
        with open(fname, 'r') as f:
            for line in f:
                words = line.strip().split()
                x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x = map(lambda ii: ii if ii < options['n_words'] else 1, x)
                x += [0]
                yield x

    def _check_length(fname):
        f = open(fname, 'r')
        count = 0
        for _ in f:
            count += 1
        f.close()

        return count

    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    ## use additional input for the policy network
    options['pre'] = config['pre']

    # ========================================================================= #
    # Build a Simultaneous Translator
    # ========================================================================= #

    # allocate model parameters
    params = init_params(options)
    params = load_params(model, params)
    tparams = init_tparams(params)

    # print 'build the model for computing cost (full source sentence).'
    trng, use_noise, \
    _x, _x_mask, _y, _y_mask, \
    opt_ret, \
    cost, f_cost = build_model(tparams, options)
    print 'done'

    # functions for sampler
    f_sim_ctx, f_sim_init, f_sim_next = build_simultaneous_sampler(tparams, options, trng)

    # function for finetune the underlying model
    if config['finetune']:
        ff_init, ff_cost, ff_update = build_simultaneous_model(tparams, options, rl=True)
        funcs = [f_sim_ctx, f_sim_init, f_sim_next, f_cost, ff_init, ff_cost, ff_update]

    else:
        funcs = [f_sim_ctx, f_sim_init, f_sim_next, f_cost]


    def _translate(src, trg, train=False, samples=config['sample'], greedy=False):
        ret = simultaneous_decoding(
            funcs,
            _policy,
            src, trg, word_idict_trg,
            step=config['step'], peek=config['peek'], sidx=config['s0'],
            n_samples=samples,
            reward_config={'target': config['target_ap'],
                            'cw':    config['target_cw'],
                           'gamma':  config['gamma'],
                           'Rtype':  config['Rtype'],
                           'maxsrc': config['maxsrc'],
                           'greedy': greedy,
                           'upper':  config['upper'],
                           'finetune': config['finetune']},
            train=train,
            use_forget=config['forget'],
            use_newinput=config['pre'],
            use_coverage=config['coverage'],
            on_groundtruth = 0)

        return ret


    # check the ID:
    policy['base'] = _model
    _policy = Policy(trng, options, policy, config,
                     n_in=options['readout_dim'] + 1 if config['coverage'] else options['readout_dim'],
                     n_out=3 if config['forget'] else 2,
                     recurrent=policy['recurrent'], id=id)

    # make the dataset ready for training & validation
    # train_    = options['datasets'][0]
    # train_num = _check_length
    trainIter = TextIterator(options['datasets'][0], options['datasets'][1],
                             options['dictionaries'][0], options['dictionaries'][1],
                             n_words_source=options['n_words_src'], n_words_target=options['n_words'],
                             batch_size=config['batchsize'],
                             maxlen=options['maxlen'])

    train_num = trainIter.num

    validIter = TextIterator(options['valid_datasets'][0], options['valid_datasets'][1],
                             options['dictionaries'][0], options['dictionaries'][1],
                             n_words_source=options['n_words_src'], n_words_target=options['n_words'],
                             batch_size=64, cache=10,
                             maxlen=1000000)

    valid_num = validIter.num

    valid_ = options['valid_datasets'][0]
    valid_num = _check_length(valid_)
    print 'training set {} lines / validation set {} lines'.format(train_num, valid_num)
    print 'use the reward function {}'.format(chr(config['Rtype'] + 65))

    # ================================================================================= #
    # Main Loop: Run
    # ================================================================================= #
    print 'Start Simultaneous Translator...'
    probar = Progbar(train_num / config['batchsize'], with_history=False)
    monitor = None
    if remote:
        monitor = Monitor(root='http://localhost:9000')

    # freqs
    save_freq     = 200
    sample_freq   = 10
    valid_freq    = 200
    valid_size    = 200
    display_freq  = 50
    finetune_freq = 5

    history, last_it = _policy.load()
    action_space = ['W', 'C', 'F']
    Log_avg = {}
    time0 = timer()
    pipe = PIPE(['x', 'x_mask', 'y', 'y_mask', 'c_mask'])

    for it, (srcs, trgs) in enumerate(trainIter):  # only one sentence each iteration
        if it < last_it:  # go over the scanned lines.
            continue

        # for validation
        # doing the whole validation!!
        reference = []
        system    = []

        reference2 = []
        system2    = []

        if it % valid_freq == (valid_freq-1):
            print 'start validation'

            collections = [[], [], [], [], []]
            probar_v = Progbar(valid_num / 64 + 1)
            for ij, (srcs, trgs) in enumerate(validIter):

                statistics = _translate(srcs, trgs, train=False, samples=1, greedy=True)

                quality, delay, reward = zip(*statistics['track'])
                reference += statistics['Ref']
                system    += statistics['Sys']

                # print ' '.join(reference[-1][0])
                # print ' '.join(system[-1])


                # compute the average consective waiting length
                def _consective(action):
                    waits = []
                    temp = 0
                    for a in action:
                        if a == 0:
                            temp += 1
                        elif temp > 0:
                            waits += [temp]
                            temp = 0

                    if temp > 0:
                        waits += [temp]

                    mean = numpy.mean(waits)
                    gec = numpy.max(waits)  # numpy.prod(waits) ** (1./len(waits))
                    return mean, gec

                def _max_length(action):
                    _cur = 0
                    _end = 0
                    _max = 0
                    for it, a in enumerate(action):
                        if a == 0:
                            _cur += 1
                        elif a == 2:
                            _end += 1

                        temp = _cur - _end
                        if temp > _max:
                            _max = temp
                    return _max

                maxlen = [_max_length(action) for action in statistics['action']]
                means, gecs = zip(*(_consective(action) for action in statistics['action']))

                collections[0] += quality
                collections[1] += delay
                collections[2] += means
                collections[3] += gecs
                collections[4] += maxlen

                values = [('quality', numpy.mean(quality)), ('delay', numpy.mean(delay)),
                          ('wait_mean', numpy.mean(means)), ('wait_max', numpy.mean(gecs)),
                          ('max_len', numpy.mean(maxlen))]
                probar_v.update(ij + 1, values=values)


            validIter.reset()
            valid_bleu, valid_delay, valid_wait, valid_wait_gec, valid_mx = [numpy.mean(a) for a in collections]
            print 'Iter = {}: AVG BLEU = {}, DELAY = {}, WAIT(MEAN) = {}, WAIT(MAX) = {}, MaxLen={}'.format(
                it, valid_bleu, valid_delay, valid_wait, valid_wait_gec, valid_mx)

            print 'Compute the Corpus BLEU={} (greedy)'.format(corpus_bleu(reference, system))

            with open(WORK + '.translate/test.txt', 'w') as fout:
                for sys in system:
                    fout.write('{}\n'.format(' '.join(sys)))

            with open(WORK + '.translate/ref.txt', 'w') as fout:
                for ref in reference:
                    fout.write('{}\n'.format(' '.join(ref[0])))

            history += [collections]


        if config['upper']:
            print 'done'
            import sys; sys.exit(-1)


        # training set sentence tuning
        new_srcs, new_trgs = [], []
        for src, trg in zip(srcs, trgs):
            if len(src) <= config['s0']:
                continue  # ignore when the source sentence is less than sidx. we don't use the policy\
            else:
                new_srcs += [src]
                new_trgs += [trg]

        if len(new_srcs) == 0:
            continue

        srcs, trgs = new_srcs, new_trgs
        #try:
        statistics, info, pipe_t = _translate(srcs, trgs, train=True)

        #except Exception:
        #    print 'translate a empty sentence. bug.'
        #    continue


        # samples, scores, actions, rewards, info, pipe_t = _translate(srcs, trgs, train=True)
        # print pipe_t


        if config['finetune'] != 'nope':

            for idx, act in enumerate(pipe_t['action']):
                _start = 0
                _end = 0
                _mask = [0 for _ in srcs[0]]
                _cmask = []

                pipe.messages['x'] += srcs
                pipe.messages['y'] += [pipe_t['sample'][idx]]

                for a in act:
                    # print _start, _end
                    if a == 0:
                        _mask[_start] = 1
                        _start += 1
                    elif a == 2:
                        _mask[_end] = 0
                        _end += 1
                    else:
                        _cmask.append(_mask)
                # print numpy.asarray(_cmask).shape

                pipe.messages['c_mask'].append(_cmask)

            if it % finetune_freq == (finetune_freq - 1):
                num = len(pipe.messages['x'])
                max_x = max([len(v) for v in pipe.messages['x']])
                max_y = max([len(v) for v in pipe.messages['y']])

                xx, xx_mask = _padding(pipe.messages['x'], shape=(max_x, num), return_mask=True, dtype='int64')
                yy, yy_mask = _padding(pipe.messages['y'], shape=(max_y, num), return_mask=True, dtype='int64')
                cc_mask = _padding(pipe.messages['c_mask'], shape=(max_y, num, max_x)).transpose([0, 2, 1])

                # fine-tune the EncDec of translation
                if config['finetune'] == 'full':
                    cost = f_fine_cost(xx, xx_mask, yy, yy_mask, cc_mask)
                elif config['finetune'] == 'decoder':
                    cost = f_fine_cost(xx, xx_mask, yy, yy_mask, cc_mask)
                else:
                    raise NotImplementedError

                print '\nIter={} || cost = {}'.format(it, cost[0])
                f_fine_update(0.00001)
                pipe.reset()

        if it % sample_freq == 0:

            print '\nModel:{} has been trained for {} hours'.format(_policy.id, (timer() - time0) / 3600.)
            print 'source: ', _bpe2words(_seqs2words([srcs[0]], word_idict))[0]
            print 'target: ', _bpe2words(_seqs2words([trgs[0]], word_idict_trg))[0]

            # obtain the translation results
            samples = _bpe2words(_seqs2words(statistics['sample'], word_idict_trg))

            # obtain the delay (normalized)
            # delays = _action2delay(srcs[0], statistics['action'])

            c  = 0
            for j in xrange(len(samples)):

                if statistics['secs'][j][0] == 0:
                    if c < (config['sample']/2.):
                        c += 1
                        continue

                    print '---ID: {}'.format(_policy.id)
                    print 'sample: ', samples[j]
                    print 'action: ', ','.join(
                        ['{}'.format(action_space[t])
                         for t in statistics['action'][j]])

                    print 'quality:', statistics['track'][j][0]
                    print 'delay:',   statistics['track'][j][1]
                    print 'reward:',  statistics['track'][j][2]
                    break

        values = [(w, info[w]) for w in info]
        probar.update(it + 1, values=values)


        # NaN detector
        for w in info:
            if numpy.isnan(info[w]) or numpy.isinf(info[w]):
                raise RuntimeError, 'NaN/INF is detected!! {} : ID={}'.format(w, id)

        # remote display
        if remote:
            logs = {'R': info['R'], 'Q': info['Q'],
                    'D': info['D'], 'P': float(info['P'])}
            # print logs
            for w in logs:
                Log_avg[w] = Log_avg.get(w, 0) + logs[w]

            if it % display_freq == (display_freq - 1):
                for w in Log_avg:
                    Log_avg[w] /= display_freq

                monitor.display(it + 1, Log_avg)
                Log_avg = dict()

        # save the history & model
        history += [info]
        if it % save_freq == 0:
            _policy.save(history, it)


if __name__ == "__main__":
    from config import rl_config
    policy, config = rl_config()

    run_simultrans(config['model'],
                   options_file=config['option'],
                   config=config,
                   policy=policy,
                   id=None,
                   remote=False)




