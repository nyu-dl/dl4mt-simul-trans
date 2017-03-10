'''
Build a neural machine translation model with soft attention
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
#import ipdb
import numpy
import copy

import os

import sys
import time

from collections import OrderedDict
from data_iterator import TextIterator
from layers import *
from optimizer import *

profile = False
TINY    = 1e-7

# -----------------------------------------------------------------------------#
# Build the Unidirectional Attention-based Neural Machine Translation

# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    params['Wemb']     = norm_weight(options['n_words_src'], options['dim_word'])
    params['Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'])

    # encoder: uni-directional RNN
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'])

    if options.get('birnn', False):
        params = get_layer(options['encoder'])[0](options, params,
                                                  prefix='encoder_r',
                                                  nin=options['dim_word'],
                                                  dim=options['dim'])

    ctxdim = options['dim'] if not options.get('birnn', False) else 2 * options['dim']

    # init_state, init_cell
    params = get_layer('ff')[0](options, params, prefix='ff_state',
                                nin=ctxdim, nout=options['dim'])
    # decoder
    params = get_layer(options['decoder'])[0](options, params,
                                              prefix='decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              dimctx=ctxdim)
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx',
                                nin=ctxdim, nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'])
    return params


# build a training model: uni-directional encoding
def build_model(tparams, options):
    opt_ret   = dict()

    trng      = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x      = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y      = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')

    # time_steps
    n_timesteps     = x_mask.shape[0]
    n_timesteps_trg = y_mask.shape[0]
    n_samples       = x_mask.shape[1]

    # word embedding for forward rnn (source)
    emb    = tparams['Wemb'][x.flatten()]
    emb    = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    proj   = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask)

    # for reverse RNN: bi-directional RNN encoder
    if options.get('birnn', False):
        xr      = x[::-1]
        xr_mask = x_mask[::-1]

        embr    = tparams['Wemb'][xr.flatten()]
        embr    = embr.reshape([n_timesteps, n_samples, options['dim_word']])
        projr   = get_layer(options['encoder'])[1](tparams, embr, options,
                                                prefix='encoder_r',
                                                mask=xr_mask)
        ctx     = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    else:
        ctx     = proj[0]  # context vectors

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean    = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    # initial decoder state
    init_state  = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    emb         = tparams['Wemb_dec'][y.flatten()]
    emb         = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb         = emb_shifted

    # decoder - pass through the decoder conditional gru with attention
    proj   = get_layer(options['decoder'])[1](tparams, emb, options,
                                              prefix='decoder',
                                              mask=y_mask, context=ctx,
                                              context_mask=x_mask,
                                              one_step=False,
                                              init_state=init_state)
    # hidden states of the decoder gru
    proj_h = proj[0]

    # weighted averages of context, generated by attention module
    ctxs   = proj[1]

    # weights (alignment matrix)
    opt_ret['dec_alphas'] = proj[2]  # --> to show the attenion weights

    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx  = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit      = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

    # dropout (noise)
    if options['use_dropout']:
        logit  = dropout_layer(logit, use_noise, trng)
    logit      = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')
    logit_shp  = logit.shape
    probs      = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

    # compute the cost (negative loglikelihood)
    y_flat     = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat

    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    # we will build an additional function for computing costs
    f_cost = theano.function([ctx, x_mask, y, y_mask], cost)
    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost, f_cost


# build a simultaneous model
def build_simultaneous_model(tparams, options, rl=True):

    # ------------------- ENCODER ------------------------------------------ #

    opt_ret   = dict()

    trng      = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x      = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y      = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')

    # time_steps
    n_timesteps     = x_mask.shape[0]
    n_timesteps_trg = y_mask.shape[0]
    n_samples       = x_mask.shape[1]

    # word embedding for forward rnn (source)
    emb    = tparams['Wemb'][x.flatten()]
    emb    = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    proj   = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask)

    # for reverse RNN: bi-directional RNN encoder
    if options.get('birnn', False):
        xr      = x[::-1]
        xr_mask = x_mask[::-1]

        embr    = tparams['Wemb'][xr.flatten()]
        embr    = embr.reshape([n_timesteps, n_samples, options['dim_word']])
        projr   = get_layer(options['encoder'])[1](tparams, embr, options,
                                                prefix='encoder_r',
                                                mask=xr_mask)
        ctx     = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    else:
        ctx     = proj[0]  # context vectors

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean    = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    # initial decoder state
    init_state  = get_layer('ff')[1](tparams, ctx_mean, options,
                                     prefix='ff_state', activ='tanh')

    print 'compile the initializer'
    f_init      = theano.function([x, x_mask], [ctx, init_state])
    print 'encoder done.'

    # ------------------- DECODER ------------------------------------------ #

    c_mask      = tensor.tensor3('c_mask', dtype='float32') # seq_t x seq_s x batches

    emb         = tparams['Wemb_dec'][y.flatten()]
    emb         = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb         = emb_shifted

    # decoder - pass through the decoder conditional gru with attention
    def _step(_emb, _y_mask, _c_mask, _init_state, _ctx):
        return get_layer(options['decoder'])[1](tparams, _emb, options,
                                                prefix='decoder',
                                                mask=_y_mask, context=_ctx,
                                                context_mask=_c_mask,
                                                one_step=True,
                                                init_state=_init_state)

    proj, _ = theano.scan(_step,
                          sequences=[emb, y_mask, c_mask],
                          outputs_info=[init_state, None, None],
                          non_sequences=[ctx])

    # hidden states of the decoder gru
    proj_h = proj[0]

    # weighted averages of context, generated by attention module
    ctxs   = proj[1]

    # weights (alignment matrix)
    opt_ret['dec_alphas'] = proj[2]  # --> to show the attenion weights

    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx  = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit      = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

    # dropout (noise)
    if options['use_dropout']:
        logit  = dropout_layer(logit, use_noise, trng)
    logit      = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')
    logit_shp  = logit.shape
    probs      = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

    # compute the cost (negative loglikelihood)
    y_flat     = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat

    cost       = -tensor.log(probs.flatten()[y_flat_idx] + TINY)
    cost       = cost.reshape([y.shape[0], y.shape[1]])

    if rl:
        advantages = tensor.matrix('advantages')
        a_cost = tensor.mean((y_mask * cost * advantages).sum(0))
    else:
        a_cost = tensor.mean((cost * y_mask).sum(0))

    lr    = tensor.scalar(name='lr')
    if not rl:
        print 'build MLE optimizer for the whole NMT model:'
        a_grad = grad_clip(theano.grad(a_cost, wrt=itemlist(tparams)))
        inps   = [x, x_mask, y, y_mask, c_mask]
        outps  = [a_cost, cost]
        f_cost, f_update = adam(lr, tparams, a_grad, inps, outps)

    else:
        print 'build REINFORCE optimizer for the whole NMT model:'
        a_grad = grad_clip(theano.grad(a_cost, wrt=itemlist(tparams)))
        inps   = [x, x_mask, y, y_mask, c_mask, advantages]
        outps  = [a_cost, cost]
        f_cost, f_update = adam(lr, tparams, a_grad, inps, outps)

    print 'done.'
    return f_init, f_cost, f_update


# build a sampler for NMT
def build_sampler(tparams, options, trng):

    x    = tensor.matrix('x', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples   = x.shape[1]

    # word embedding (source), forward and backward
    emb  = tparams['Wemb'][x.flatten()]
    emb  = emb.reshape([n_timesteps, n_samples, options['dim_word']])

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder')

    # bi-rnn
    if options.get('birnn', False):
        xr    = x[::-1]

        embr  = tparams['Wemb'][xr.flatten()]
        embr  = embr.reshape([n_timesteps, n_samples, options['dim_word']])
        projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                                 prefix='encoder_r')

        ## concatenate forward and backward rnn hidden states
        ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    else:
        ctx  = proj[0]

    # get the input for decoder rnn initializer mlp
    ctx_mean   = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    print 'Building f_init...',
    outs       = [init_state, ctx]
    f_init     = theano.function([x], outs, name='f_init', profile=profile)
    print 'Done.'

    # ..........................................................................
    # x: 1 x 1
    y          = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')
    use_noise  = theano.shared(numpy.float32(0.))


    # if it's the first word, emb should be all zero and it is indicated by -1
    emb  = tensor.switch(y[:, None] < 0,
                         tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                         tparams['Wemb_dec'][y])

    # apply one step of conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=ctx,
                                            one_step=True,
                                            init_state=init_state)
    # get the next hidden state
    next_state = proj[0]

    # get the weighted averages of context for this target word y
    ctxs       = proj[1]

    logit_lstm = get_layer('ff')[1](tparams, next_state, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx  = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit      = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

    if options['use_dropout']:
        logit  = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')

    # compute the softmax probability
    next_probs  = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print 'Building f_next..',
    inps = [y, ctx, init_state]
    outs = [next_probs, next_sample, next_state]
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done.'

    return f_init, f_next

def build_partial(tparams, options, trng):

    assert options.get('birnn', False), 'must used in uni-directional mode'

    x           = tensor.matrix('x', dtype='int64')
    prev_state  = tensor.matrix('prev_state', dtype='float32')
    n_timesteps = x.shape[0]
    n_samples   = x.shape[1]

    # word embedding (source), forward and backward
    emb  = tparams['Wemb'][x.flatten()]
    emb  = emb.reshape([n_timesteps, n_samples, options['dim_word']])

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            one_step=True,
                                            _init_state=prev_state,
                                            prefix='encoder')
    next_state = proj[0]


    print 'Building f_partial...',
    outs      = [next_state]
    f_partial = theano.function([x, prev_state], outs, name='f_partial', profile=profile)
    print 'Done'

    return f_partial


# ----------------------------------------------------------------------- #
# Simultaneous NMT
def build_simultaneous_sampler(tparams, options, trng):
    x = tensor.matrix('x', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples   = x.shape[1]

    # word embedding (source), forward and backward
    emb   = tparams['Wemb'][x.flatten()]
    emb   = emb.reshape([n_timesteps, n_samples, options['dim_word']])

    # encoder
    proj  = get_layer(options['encoder'])[1](tparams, emb, options, prefix='encoder')

    # bi-rnn
    if options.get('birnn', False):
        xr    = x[::-1]

        embr  = tparams['Wemb'][xr.flatten()]
        embr  = embr.reshape([n_timesteps, n_samples, options['dim_word']])
        projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                                 prefix='encoder_r')
        ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    else:
        ctx  = proj[0]

    # get the input for decoder rnn initializer mlp
    ctx_mean   = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    print 'Building f_ctx/init...',

    f_sim_ctx  = theano.function([x], [init_state, ctx, ctx_mean], name = 'f_sim_ctx')
    f_sim_init = theano.function([ctx], init_state, name='f_sim_init')

    print 'Done.'

    # -------------------------------------------------------------------------------- #
    y          = tensor.vector('y_sampler', dtype='int64')
    ctx        = tensor.tensor3('context_vectors', dtype='float32')
    mask       = tensor.matrix('context_mask', dtype='float32')
    init_state = tensor.matrix('init_state', dtype='float32')
    use_noise  = theano.shared(numpy.float32(0.))

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb  = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    # apply one step of conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=ctx,
                                            one_step=True,
                                            init_state=init_state,
                                            context_mask=mask)

    # get the next hidden state
    next_state  = proj[0]

    # get the weighted averages of context for this target word y
    ctxs        = proj[1]
    attention   = proj[2]

    logit_lstm  = get_layer('ff')[1](tparams, next_state, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev  = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx   = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit       = tensor.tanh(logit_lstm + logit_prev + logit_ctx)

    if options['use_dropout']:
        logit   = dropout_layer(logit, use_noise, trng)

    logit       = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')

    # compute the softmax probability
    next_probs  = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # ***== special care: use additional inforamtion ====*** #
    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print 'Building f_sim_next..',
    inps = [y, ctx, mask, init_state]
    ctxdim = options['dim'] if not options.get('birnn', False) else 2 * options['dim']

    if 'pre' in options and options['pre']:
        assert not options.get('birnn', False), 'should not use birnn for SimulTrans'

        read_head   = tensor.ivector('read_head')
        forget_head = tensor.ivector('forget_head')
        inps += [read_head, forget_head]

        def _grab(contexts, index):
            assert contexts.ndim == 3

            batch_size = contexts.shape[1]
            return contexts[index, tensor.arange(batch_size), :]

        last_ctx   = _grab(ctx, read_head)
        first_ctx  = _grab(ctx, forget_head)
        next_max_w = tparams['Wemb_dec'][next_probs.argmax(1)]

        readout    = tensor.concatenate([next_state, ctxs, last_ctx, first_ctx, next_max_w], axis=-1)
        options['readout_dim'] = options['dim_word'] + ctxdim * 3 + options['dim']

    else:
        print 'with normal input'
        next_max_w = tparams['Wemb_dec'][next_probs.argmax(1)]
        readout = tensor.concatenate([next_state, ctxs, next_max_w], axis=-1)  # the obersavtion for each step.
        options['readout_dim'] = options['dim_word'] + options['dim'] + ctxdim

    outs = [next_probs, next_sample, next_state, readout, attention, emb]
    f_sim_next = theano.function(inps, outs, name='f_sim_next', profile=profile)
    print 'Done.'

    return f_sim_ctx, f_sim_init, f_sim_next


# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams, f_init, f_next, x, options, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False, sigma=-1.):

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    # get initial state of decoder rnn and encoder context
    ret = f_init(x)
    next_state, ctx0 = ret[0], ret[1]
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])

        if sigma > 0.:
            next_state_inp = next_state + numpy.float32((sigma/(ii+1)) * numpy.random.randn(*next_state.shape))
        else:
            next_state_inp = next_state

        inps = [next_w, ctx, next_state_inp]
        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score += next_p[0, nw]
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True):
    probs = []

    n_done = 0

    for x, y in iterator:
        n_done += len(x)

        x, x_mask, y, y_mask = prepare_data(x, y,
                                            n_words_src=options['n_words_src'],
                                            n_words=options['n_words'])

        pprobs = f_log_probs(x, x_mask, y, y_mask)
        for pp in pprobs:
            probs.append(pp)

        #if numpy.isnan(numpy.mean(probs)):
        #    ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs)


# -----------------------------------------------------------------------------#
# Batch preparation
def prepare_data(seqs_x,
                 seqs_y,
                 maxlen=None,
                 n_words_src=30000,
                 n_words=30000):

    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x    = []
        new_seqs_y    = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x    = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y    = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x  = numpy.max(lengths_x) + 1
    maxlen_y  = numpy.max(lengths_y) + 1

    x      = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y      = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask


#-----------------------------------------------------------------------------#
# Training Function:

def train(dim_word     = 100,  # word vector dimensionality
          dim          = 1000,  # the number of RNN units
          encoder      = 'gru',
          decoder      = 'gru_cond',
          patience     = 10,  # early stopping patience
          max_epochs   = 5000,
          finish_after = 10000000,  # finish after this many updates
          dispFreq     = 100,
          decay_c      = 0.,  # L2 regularization penalty
          alpha_c      = 0.,  # alignment regularization
          clip_c       = -1.,  # gradient clipping threshold
          lrate        = 0.01,  # learning rate
          n_words_src  = 100000,  # source vocabulary size
          n_words      = 100000,  # target vocabulary size
          maxlen       = 100,  # maximum length of the description
          optimizer    = 'rmsprop',
          batch_size   = 16,
          valid_batch_size = 16,
          saveto       = 'model.npz',
          validFreq    = 1000,
          saveFreq     = 1000,   # save the parameters after every saveFreq updates
          sampleFreq   = 100,   # generate some samples after every sampleFreq
          datasets     =[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'],

          valid_datasets=['../data/dev/newstest2011.en.tok',
                          '../data/dev/newstest2011.fr.tok'],

          dictionaries=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok.pkl'],

          use_dropout  = False,
          reload_      = False,
          overwrite    = False):

    # Model options
    model_options = locals().copy()

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        with open(dd, 'rb') as f:
            worddicts[ii] = pkl.load(f)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'Reloading model options'
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)

    print 'Loading data'
    train = TextIterator(datasets[0], datasets[1],
                         dictionaries[0], dictionaries[1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen)
    valid = TextIterator(valid_datasets[0], valid_datasets[1],
                         dictionaries[0], dictionaries[1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=valid_batch_size,
                         maxlen=maxlen)

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        print 'Reloading model parameters'
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost, f_cost = \
        build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask]

    print 'Building sampler'
    f_init, f_next = build_sampler(tparams, model_options, trng)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    print 'Optimization'

    best_p = None
    bad_counter = 0
    uidx = 0
    estop = False
    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        rmodel = numpy.load(saveto)
        history_errs = list(rmodel['history_errs'])
        if 'uidx' in rmodel:
            uidx = rmodel['uidx']

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    for eidx in xrange(max_epochs):
        n_samples = 0

        for x, y in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen,
                                                n_words_src=n_words_src,
                                                n_words=n_words)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask, y, y_mask)

            # do the update on parameters
            f_update(lrate)

            ud = time.time() - ud_start

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            # save the best model so far, in addition, save the latest model
            # into a separate file with the iteration number for external eval
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving the best model...',
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, uidx=uidx, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                params = unzip(tparams)
                numpy.savez('%s.current'%(saveto), history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.current.pkl' % saveto, 'wb'))
                print 'Done'

                # save with uidx
                if not overwrite:
                    print 'Saving the model at iteration {}...'.format(uidx),
                    saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], uidx)
                    numpy.savez(saveto_uidx, history_errs=history_errs,
                                uidx=uidx, **unzip(tparams))
                    print 'Done'

            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(numpy.minimum(5, x.shape[1])):
                    stochastic = False
                    sample, score = gen_sample(tparams, f_init, f_next,
                                               x[:, jj][:, None],
                                               model_options, trng=trng, k=1,
                                               maxlen=30,
                                               stochastic=stochastic,
                                               argmax=True)
                    print 'Source ', jj, ': ',
                    ss = []
                    for vv in x[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[0]:
                            ss.append(worddicts_r[0][vv])
                        else:
                            ss.append('UNK')
                    print ' '.join(ss).replace('@@ ', '')
                    print 'Truth ', jj, ' : ',
                    ss = []
                    for vv in y[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[1]:
                            ss.append(worddicts_r[1][vv])
                        else:
                            ss.append('UNK')
                    print ' '.join(ss).replace('@@ ', '')
                    print 'Sample ', jj, ': ',
                    tt = []

                    score = score / numpy.array([len(s) for s in sample])
                    ss = sample[score.argmin()]
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in worddicts_r[1]:
                            tt.append(worddicts_r[1][vv])
                        else:
                            tt.append('UNK')
                    print ' '.join(tt).replace('@@ ', '')

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                #if numpy.isnan(valid_err):
                #    ipdb.set_trace()

                print 'Valid ', valid_err

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid_err = pred_probs(f_log_probs, prepare_data,
                           model_options, valid).mean()

    print 'Valid ', valid_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                uidx=uidx,
                **params)

    return valid_err


if __name__ == '__main__':
    pass
