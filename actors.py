"""
Deterministic Actor Functions:
"""
from layers import *

TINY = 1e-7

# -------------------------------------------------------------------------#
# Noise
def ou_noise(trng, x, mu=0., theta=0.15, sigma=0.01):
    dx = theta * (mu - x) + sigma * trng.normal(x.shape)
    return x + dx


def gaussian_noise(trng, x, mu=0, sigma=0.01):
    dx = mu + sigma * trng.normal(x.shape)
    return dx


# -------------------------------------------------------------------------#
# Actors:
actors    = dict()
actors['dumb']  = ('param_init_dumb', 'dumb_actor')
actors['const'] = ('param_init_constant', 'constant_actor')
actors['ff']    = ('param_init_ff',   'ff_actor')
actors['gru']   = ('param_init_gru',  'gru_actor', 'gru_actor_hard')
actors['gru2']  = ('param_init_gru2', 'gru_actor2')
actors['gg']    = ('param_init_gg',   'gg_actor')


def get_actor(name):
    fns = actors[name]
    return tuple([eval(f) for f in fns])


def _p(pp, name):
    return '%s_%s' % (pp, name)


# -------------------------------------------------------------------------#
# Dump Actors:
def param_init_dumb(options, prefix='db', nin=None, nout=None):
    params = OrderedDict()
    if nin is None:
        nin = options['dim'] + options['ctxdim']

    if nout is None:
        nout = options['dim']

    return params


def dumb_actor(tparams, options,h1, ctx=None, act=None, prefix='db'):

    action = tensor.zeros_like(h1)
    hidden = act
    return action, hidden


# constant Actors:
def param_init_constant(options, prefix='ct', nin=None, nout=None):
    params = OrderedDict()
    if nin is None:
        nin = options['dim'] + options['ctxdim']

    if nout is None:
        nout = options['dim']

    params[_p(prefix, 'a')] = numpy.zeros((nout,)).astype('float32')
    return params


def constant_actor(tparams, options, h1, ctx=None, act=None, prefix='ct'):
    action = tensor.zeros_like(h1)
    if action.ndim == 2:
        action += tparams[_p(prefix, 'a')][None, :]
    elif action.ndim == 3:
        action += tparams[_p(prefix, 'a')][None, None, :]
    else:
        action += tparams[_p(prefix, 'a')]

    hidden = act
    return action, hidden


# Feedforward Actors:
def param_init_ff(options, prefix='ff', nin=None, nout=None, nhid=None):

    params = OrderedDict()

    if nin is None:
        nin = options['dim'] + options['ctxdim']

    if nout is None:
        nout = options['dim']

    if nhid is None:
        nhid = options['act_hdim']

    params = get_layer('ff')[0](options, params, prefix=prefix + '_in',
                                nin=nin, nout=nhid, scale=0.001)

    params = get_layer('ff')[0](options, params, prefix=prefix + '_out',
                                nin=nhid, nout=nout, scale=0.001)

    return params


def ff_actor(tparams, options, h1, ctx=None, act=None, prefix='ff'):

    hidden = get_layer('ff')[1](tparams, concatenate([h1, ctx], axis=1),
                                options, prefix=prefix + '_in',  activ='tanh')
    action = get_layer('ff')[1](tparams, hidden,
                                options, prefix=prefix + '_out', activ='tanh')

    return action, hidden


# Recurrent Actors:
def param_init_gru(options, prefix='ff', nin=None, nout=None, nhid=None):

    params = OrderedDict()

    if nin is None:
        nin = 2 * options['dim'] + options['ctxdim']

    if nout is None:
        nout = options['dim']

    if nhid is None:
        nhid = options['act_hdim']

    # params = get_layer('lngru')[0](options, params, prefix=prefix + '_in',
    #                                nin=nin, dim=nhid, scale=0.001)
    params = get_layer('gru')[0](options, params, prefix=prefix + '_in',
                                   nin=nin, dim=nhid, scale=0.001)

    params = get_layer('ff')[0](options, params, prefix=prefix + '_out',
                                nin=nhid, nout=nout, scale=0.001)

    return params


def gru_actor(tparams, options, h1, ctx=None, act=None, prefix='ff'):

    pre_state, pre_action = act[:, :options['act_hdim']], act[:, options['act_hdim']:]
    # hidden = get_layer('lngru')[1](tparams, concatenate([h1, ctx, pre_action], axis=1),
    #                             options, prefix=prefix + '_in',
    #                             one_step=True, _init_state=pre_state)[0]
    hidden = get_layer('gru')[1](tparams, concatenate([h1, ctx, pre_action], axis=1),
                                options, prefix=prefix + '_in',
                                one_step=True, _init_state=pre_state)[0]

    action = get_layer('ff')[1](tparams, hidden,
                                options, prefix=prefix + '_out', activ='tanh')
    cur_act = concatenate([hidden, action], axis=1)
    return action, cur_act


# Recurrent Actor2
def param_init_gru2(options, prefix='ff', nin=None, nout=None, nhid=None):

    params = OrderedDict()

    if nin is None:
        nin  = options['dim']

    if nout is None:
        nout = options['dim']

    if nhid is None:
        nhid = options['act_hdim']

    # params = get_layer('lngru')[0](options, params, prefix=prefix + '_in',
    #                                nin=nin, dim=nhid, scale=0.001)
    params = get_layer('gru')[0](options, params, prefix=prefix + '_in',
                                 nin=nin, dim=nhid, scale=0.001)

    params = get_layer('ff')[0](options, params, prefix=prefix + '_out',
                                nin=nhid, nout=nout, scale=0.001)

    return params


def gru_actor2(tparams, options, h1, act=None, prefix='ff'):

    # hidden = get_layer('lngru')[1](tparams, concatenate([h1, ctx, pre_action], axis=1),
    #                             options, prefix=prefix + '_in',
    #                             one_step=True, _init_state=pre_state)[0]
    hidden = get_layer('gru')[1](tparams, h1,
                                options, prefix=prefix + '_in',
                                one_step=True, _init_state=act)[0]

    action = get_layer('ff')[1](tparams, hidden,
                                options, prefix=prefix + '_out', activ='tanh')
    return action, hidden


def gru_actor_hard(tparams, options, h1, ctx=None, act=None, prefix='ff', bound=0.1):

    pre_state, pre_action = act[:, :options['act_hdim']], act[:, options['act_hdim']:]
    # hidden = get_layer('lngru')[2](tparams, concatenate([h1, ctx, pre_action], axis=1),
    #                             options, prefix=prefix + '_in',
    #                             one_step=True, _init_state=pre_state)[0]
    hidden = get_layer('gru')[1](tparams, concatenate([h1, ctx, pre_action], axis=1),
                                options, prefix=prefix + '_in',
                                one_step=True, _init_state=pre_state)[0]

    action = get_layer('ff')[1](tparams, hidden,
                                options, prefix=prefix + '_out', activ='tanh')

    a_norm = tensor.sqrt(tensor.sum(action ** 2, axis=-1, keepdims=True))
    action = tensor.switch(a_norm > bound, action / a_norm * bound, action)  # add a hard boundary of actions

    cur_act = concatenate([hidden, action], axis=1)
    return action, cur_act


# Recurrent Actors:
def param_init_gg(options, prefix='ff', nin=None, nout=None, nhid=None):

    params = OrderedDict()

    if nin is None:
        nin = 2 * options['dim'] + options['ctxdim']

    if nout is None:
        nout = options['dim']

    if nhid is None:
        nhid = options['act_hdim']

    # params = get_layer('lngru')[0](options, params, prefix=prefix + '_in',
    #                                nin=nin, dim=nhid, scale=0.001)
    params = get_layer('gru')[0](options, params, prefix=prefix + '_in',
                                   nin=nin, dim=nhid, scale=0.001)

    params = get_layer('ff')[0](options, params, prefix=prefix + '_out',
                                nin=nhid, nout=nout, scale=0.001)

    # params = get_layer('ff')[0](options, params, prefix=prefix + '_gate',
    #                             nin=nhid + nout, nout=1)
    # params = get_layer('ff')[0](options, params, prefix=prefix + '_gate',
    #                             nin=nin + nout, nout=1)
    params = get_layer('ff')[0](options, params, prefix=prefix + '_gate',
                                nin=nin + nout, nout=nout)

    return params


def gg_actor(tparams, options, h1, ctx=None, act=None, prefix='ff'):

    pre_state, pre_action = act[:, :options['act_hdim']], act[:, options['act_hdim']:]
    # hidden = get_layer('lngru')[1](tparams, concatenate([h1, ctx, pre_action], axis=1),
    #                             options, prefix=prefix + '_in',
    #                             one_step=True, _init_state=pre_state)[0]
    hidden  = get_layer('gru')[1](tparams, concatenate([h1, ctx, pre_action], axis=1),
                                options, prefix=prefix + '_in',
                                one_step=True, _init_state=pre_state)[0]

    output  = get_layer('ff')[1](tparams, hidden,
                                options, prefix=prefix + '_out', activ='tanh')
    # gate    = get_layer('ff')[1](tparams, concatenate([hidden, output], axis=1), options, prefix=prefix + '_gate', activ='sigmoid')[:, 0]
    # gate    = get_layer('ff')[1](tparams, concatenate([h1, ctx, pre_action, output], axis=1), options, prefix=prefix + '_gate', activ='sigmoid')[:, 0]
    # action  = output * gate[:, None]
    gate    = get_layer('ff')[1](tparams, concatenate([h1, ctx, pre_action, output], axis=1), options, prefix=prefix + '_gate', activ='sigmoid')
    action  = output * gate
    cur_act = concatenate([hidden, action], axis=1)
    return action, cur_act




