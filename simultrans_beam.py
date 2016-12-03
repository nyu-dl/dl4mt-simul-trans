"""
Simultaneous Machine Translateion

"""

from nmt_uni import *
from reward  import return_reward

import time
import sys

timer = time.time

# utility functions
def _seqs2words(caps, idict):
    capsw = []
    for cc in caps:
        ww = []
        for w in cc:
            if w == 0:
                break
            ww.append(idict[w])
        capsw.append(' '.join(ww))
    return capsw

def _bpe2words(capsw):
    capw = []
    for cc in capsw:
        capw += [cc.replace('@@ ', '')]
    return capw

def _action2delay(src, actions):
    delays = []
    X = len(src)
    for act in actions:
        A = numpy.array(act, dtype='float32')
        Y = numpy.sum(act)
        S = numpy.sum(numpy.cumsum(1 - A) * A)

        assert (X > 0) and (Y > 0), 'avoid NAN {}, {}'.format(X, Y)

        tau = S / (Y * X)
        delays.append([tau, X, Y, S])

    return delays


# padding for computing policy gradient
def _padding(arrays, shape, dtype='float32', return_mask=False, sidx=0):
    B = numpy.zeros(shape, dtype=dtype)

    if return_mask:
        M = numpy.zeros((shape[0], shape[1]), dtype='float32')

    for it, arr in enumerate(arrays):
        arr = numpy.asarray(arr, dtype=dtype)
        # print arr.shape

        steps = arr.shape[0]

        if arr.ndim < 2:
            B[sidx: steps + sidx, it] = arr
        else:
            steps2 = arr.shape[1]
            B[sidx: steps + sidx, it, : steps2] = arr

        if return_mask:
            M[sidx: steps + sidx, it] = 1.

    if return_mask:
        return B, M
    return B


class PIPE(object):
    def __init__(self, keys=None):
        self.messages          = OrderedDict()
        self.hyp_messages      = OrderedDict()
        self.new_hyp_messages  = OrderedDict()
        for key in keys:
            self.messages[key] = []

    def reset(self):
        for key in self.messages:
            self.messages[key] = []

        self.hyp_messages = OrderedDict()
        self.new_hyp_messages = OrderedDict()

    def clean(self):
        for key in self.messages:
            self.messages[key] = []

    def clean_hyp(self):
        self.hyp_messages      = OrderedDict()

    def clean_new_hyp(self):
        self.new_hyp_messages  = OrderedDict()

    def init_hyp(self, key, live_k=None):
        if live_k is not None:
            self.hyp_messages[key] = [[] for _ in xrange(live_k)]
        else:
            self.hyp_messages[key] = []

    def init_new_hyp(self, key, use_copy=False):
        if use_copy:
            self.new_hyp_messages[key] = copy.copy(self.hyp_messages[key])
        else:
            self.new_hyp_messages[key] = []

    def append(self, key, new, idx=None, use_hyp=False):
        if not use_hyp:
            self.new_hyp_messages[key].append(new)
        else:
            self.new_hyp_messages[key].append(self.hyp_messages[key][idx] + [new])

    def append_new(self, key, idx, hyper=True):
        if hyper:
            self.hyp_messages[key].append(self.new_hyp_messages[key][idx])
        else:
            # print self.messages['sample']
            self.messages[key].append(self.new_hyp_messages[key][idx])

    def add(self, key, new, idx):
        self.new_hyp_messages[key][idx] += new

    def asarray(self, key, replace=False):
        if replace:
            self.hyp_messages[key] = numpy.array(self.hyp_messages[key])
        else:
            return numpy.array(self.hyp_messages[key], dtype='float32')

    def split(self):
        truth  = OrderedDict()
        sample = OrderedDict()


        for key in self.messages:
            if key == 'source':
                continue

            truth[key]  = []
            sample[key] = []

            if key == 'mask':
                for idx in xrange(len(self.messages['source'])):
                    if self.messages['source'][idx] < 0:
                        sample[key].append(self.messages[key][:, idx])
                    else:
                        truth[key].append(self.messages[key][:, idx])
            else:
                for idx in xrange(len(self.messages['source'])):
                    if self.messages['source'][idx] < 0:
                        sample[key].append(self.messages[key][idx])
                    else:
                        truth[key].append(self.messages[key][idx])

        self.messages = sample
        return truth

# ==============================================================
# Simultaneous Translation in Batch-mode
# ==============================================================
def simultaneous_decoding(f_sim_ctx,
                          f_sim_init,
                          f_sim_next,
                          f_cost,
                          _policy,
                          srcs,     # source sentences
                          trgs,     # taeget sentences
                          t_idict = None,
                          step=1, peek=1, sidx=3,
                          n_samples=10,
                          maxlen=200,
                          reward_config=None,
                          train=False,
                          use_forget=False,
                          forget_left=True,
                          use_newinput=False,
                          full_attention=False,
                          use_coverage=False,
                          on_groundtruth=0,
                          src_eos=True,
                          B=5):
    """
    :param f_init:     initializer using the first "sidx" words.
    :param f_sim_next:
    :param f_partial:
    :param src:        the original input needed to be translated (just for the speed)
    :param step:       step_size for each wait
    :param peek:
    :param sidx:       pre-read sidx words from the source
    :return:
    """
    Statistcs     = OrderedDict()

    n_sentences   = len(srcs)
    n_out         = 3 if use_forget else 2
    max_steps     = -1

    _probs        = numpy.zeros((3, )) if use_forget else numpy.zeros((2, ))
    _total        = 0

    # check
    assert n_sentences == 1, 'only works for one sentence'
    assert n_samples == 1,  'only works for one sample'

    # ================================================================================================= #
    # Generating Trajectories based on Current Policy
    # ================================================================================================= #

    live_k    = 1 # (n_samples + on_groundtruth) * n_sentences
    live_all  = live_k

    # Critical! add the <eos>
    srcs = [src + [0] for src in srcs]

    src_max   = max([len(src) for src in srcs])
    if src_max < sidx:
        sidx  = src_max

    x, ctx0, z0, secs0 = [], [], [], []

    # data initialization
    for id, (src, trg) in enumerate(zip(srcs, trgs)):

        _x    = numpy.array(src, dtype='int64')[:, None]
        _, _ctx0, _ = f_sim_ctx(_x)
        _z0   = f_sim_init(_ctx0[:sidx, :])

        x.append(_x[:, 0])
        ctx0.append(_ctx0[:, 0, :])
        z0.append(_z0.flatten())
        secs0.append([id, len(src), 0])  # word id / source length / correctness


    # pad the results
    x, x_mask = _padding(x, (src_max, n_sentences), dtype='int64', return_mask=True)
    ctx       = _padding(ctx0, (src_max, n_sentences, ctx0[0].shape[-1]))
    z0        = numpy.asarray(z0)
    mask      = numpy.asarray([1.] * sidx + [0.] * (src_max - sidx), dtype='float32')[:, None]

    # hidden states
    hidden0   = _policy.init_hidden()
    secs      = []
    for _ in xrange(live_k / n_sentences):
        secs += copy.deepcopy(secs0)


    # ====================================================================================== #
    # PIPE for message passing
    # ====================================================================================== #
    pipe      = PIPE(['sample', 'score', 'action', 'obs', 'attentions',
                      'old_attend', 'coverage', 'source', 'forgotten','secs'])

    # Build for the temporal results: hyp-message
    for key in ['sample', 'obs', 'attentions', 'hidden', 'old_attend']:
        pipe.init_hyp(key, live_k)

    # special care
    pipe.hyp_messages['source']    = [-1 for _ in xrange(n_samples)] + [0 for _ in xrange(on_groundtruth)]
    pipe.hyp_messages['source']    = [si for si in pipe.hyp_messages['source'] for _ in xrange(n_sentences)]

    pipe.hyp_messages['score']     = numpy.zeros(live_k).astype('float32')
    pipe.hyp_messages['action']    = [[0] * sidx for _ in xrange(live_k)]
    pipe.hyp_messages['coverage']  = numpy.zeros((live_k, ctx.shape[0])).astype('float32')

    pipe.hyp_messages['mask']      = mask
    pipe.hyp_messages['ctx']       = ctx
    pipe.hyp_messages['secs']      = secs
    pipe.hyp_messages['states']    = z0
    pipe.hyp_messages['heads']     = numpy.asarray([[sidx, 0, 0]] * live_k)  # W C F

    # these are inputs that needs to be updated
    prev_w     = -1 * numpy.ones((live_k, )).astype('int64')
    prev_z     = z0
    prev_hid   = hidden0
    step       = 0

    # =======================================================================
    # ROLLOUT: Iteration until all the samples over.
    # Action space:
    # 0: Read,
    # 1: Commit,
    # 2: Forget,
    # =======================================================================

    beamsize   = B
    FLAG       = 0
    beamwords  = [[]]
    beamscores = numpy.zeros((1)).astype('float32')
    beamz      = prev_z
    while live_k > 0:
        step += 1
        # if step > 10:
        #     import sys; sys.exit(111)

        mask2        = numpy.tile(mask, [1, prev_z.shape[0]])
        ctx2         = numpy.tile(ctx,  [1, prev_z.shape[0], 1])
        inps         = [prev_w, ctx2, mask2, prev_z]

        # print mask
        next_p, _, next_z, next_o, next_a, cur_emb = f_sim_next(*inps)
        cand_scores  = beamscores[:, None] - numpy.log(next_p)
        cand_flat    = cand_scores.flatten()
        ranks_flat = cand_flat.argsort()[:beamsize]
        # if len(beamwords) < beamsize:
        #     beamwords = [beamwords[0] for _ in xrange(beamsize)]

        # print ranks_flat
        voc_size      = next_p.shape[1]
        trans_indices = ranks_flat / voc_size
        word_indices  = ranks_flat % voc_size
        costs         = cand_flat[ranks_flat]

        _cand, _scores, _states = [], [], []

        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            _cand.append(beamwords[ti] + [wi])
            _scores.append(costs[idx])
            _states.append(next_z[ti])

        # new place-holders for temporal results: new-hyp-message
        pipe.clean_new_hyp()


        for key in ['sample', 'score', 'heads', 'attentions', 'old_attend', 'coverage', 'source',
                    'mask', 'secs', 'states']:
            pipe.init_new_hyp(key, use_copy=True)

        for key in ['action','obs', 'hidden']:
            pipe.init_new_hyp(key, use_copy=False)

        _observes = []
        for ti in trans_indices:
            _observes.append(next_o[ti:ti+1])


        # choose the best observation for next step action
        next_o = _observes[numpy.argmin(_scores)]

        # make the source length as an input
        slens  = (pipe.new_hyp_messages['heads'][:, 0] - pipe.new_hyp_messages['heads'][:, 2]).astype('float32')
        next_o = numpy.concatenate([next_o, slens[:, None]], axis=-1)


        # Rollout the action.
        _actions, _aprop, _hidden, _z = _policy.action(next_o, prev_hid)  # input the current observation

        # print _actions.shape
        if reward_config['greedy']:
            _actions = _aprop.argmax(-1)
            # print _actions.shape

        _total += _aprop.shape[0]
        _probs += _aprop.sum(axis=0)

        idx  = 0
        a    = _actions[idx]

        # *****  Evaluate the Action !!! *****
        # for wait:
        if reward_config.get('upper', False):
            # a = 1 - pipe.hyp_messages['action'][idx][-1]
            a = 0 # testing upper bound: only wait

        if reward_config['greedy'] and (pipe.new_hyp_messages['heads'][idx, 0]
                                     >= pipe.new_hyp_messages['secs'][idx][1]):
            a = 1 # in greedy mode. must end.

        if reward_config['greedy'] and (pipe.new_hyp_messages['heads'][idx, 2]
                                     >= pipe.new_hyp_messages['heads'][idx, 0]):
            a = 1 # in greedy mode. must end.

        # message appending
        pipe.append('obs',       next_o[idx],   idx=idx, use_hyp=True)
        pipe.append('action',    a,             idx=idx, use_hyp=True)   # collect action.
        pipe.append('hidden',    _hidden[idx])

        # print pipe.hyp_messages['heads'][idx]

        # print 'action', a
        if a == 0:
            # read-head move on one step
            # print 'p', pipe.hyp_messages['heads'][idx, 0], pipe.hyp_messages['secs'][idx]

            if pipe.new_hyp_messages['heads'][idx, 0] < pipe.new_hyp_messages['secs'][idx][1]:
                pipe.new_hyp_messages['mask'][pipe.new_hyp_messages['heads'][idx, 0], idx] = 1
                pipe.new_hyp_messages['heads'][idx, 0] += 1

            # pipe.append('forgotten', -1, idx=idx, use_hyp=True)

            # if the first word is still waiting for decoding
            if numpy.sum(pipe.new_hyp_messages['action'][idx]) == 0:
                temp_sidx = pipe.new_hyp_messages['heads'][idx, 0]
                _ctx0     = ctx0[pipe.new_hyp_messages['secs'][idx][0]][:, None, :]
                _z0       = f_sim_init(_ctx0[:temp_sidx])  # initializer
                pipe.new_hyp_messages['states'][idx] = _z0
                beamz     = _z0

            # output if it is the first wait.
            if FLAG == 1:
                # clean the true buffer
                best_idx   = numpy.argmin(beamscores)
                beamwords  = [beamwords[best_idx]]
                beamscores = beamscores[best_idx: best_idx+1]
                beamz      = beamz[best_idx: best_idx+1]

                if len(pipe.messages['sample']) > 0:
                    values = [v / (len(s)) for s, v in zip(pipe.messages['sample'], pipe.messages['score'])]
                    # values = [v for s, v in zip(pipe.messages['sample'], pipe.messages['score'])]
                    best_v = numpy.argmin(values)

                    """
                    if values[best_v] < (beamscores[0] / len(beamwords[0])):
                        break
                    else:
                        pipe.clean()
                    """
                    pipe.clean()

                FLAG = 0

        # for commit:
        elif a == 1:
            beamscores = numpy.array(_scores, dtype='float32')
            beamwords  = _cand
            beamz      = numpy.array(_states, dtype='float32')
            head_t     = pipe.new_hyp_messages['source'][idx]

            # always save the best
            best_idx = numpy.argmin(beamscores)
            pipe.new_hyp_messages['sample'] = [beamwords[best_idx]]

            pipe.add('score', beamscores[best_idx], idx)
            pipe.new_hyp_messages['states'][idx]    = _states[best_idx]
            pipe.new_hyp_messages['heads'][idx, 1] += 1

            if FLAG == 0:
                beamsize   = B
                FLAG = 1

        else:
            raise NotImplementedError


        # check the correctness, or given a very negative reward
        # print pipe.new_hyp_messages['heads'][:, 0], pipe.new_hyp_messages['secs']
        for idx in xrange(live_k):
            if pipe.new_hyp_messages['heads'][idx, 0] >= pipe.new_hyp_messages['secs'][idx][1]:  # the read head already reached the end.
                pipe.new_hyp_messages['secs'][idx][2] = -1


        #  kill the completed samples, so I need to build new hyp-messages
        pipe.clean_hyp()

        for key in ['sample', 'score', 'heads', 'mask', 'states', 'coverage', 'forgotten',
                    'action', 'obs', 'secs', 'attentions', 'hidden', 'old_attend', 'source']:
            pipe.init_hyp(key)


        # trash
        trash_idx = []
        for idx in xrange(len(beamwords)):
            if (len(beamwords[idx]) > 0) and \
                  ((beamwords[idx][-1] == 0)         # translate over
                   or (pipe.new_hyp_messages['heads'][0][1] >= maxlen)   # exceed the maximum length
                   or (step > (1.5 * maxlen))):

                trash_idx.append(idx)
                beamsize -= 1

                if beamsize == 0:
                    live_k -= 1

        remain_idx = [i for i in xrange(len(beamwords)) if i not in trash_idx]

        if len(trash_idx) > 0:
            for w in trash_idx:
                pipe.messages['sample'].append(beamwords[w])
                pipe.messages['score'].append(beamscores[w])
                pipe.messages['action'].append(pipe.new_hyp_messages['action'][0])

        if (len(trash_idx) > 0) and (len(remain_idx) > 0) :
            beamwords  = [beamwords[w] for w in remain_idx]
            beamscores = numpy.concatenate([beamscores[w: w+1] for w in remain_idx], axis=0)
            beamz      = numpy.concatenate([beamz[w: w+1] for w in remain_idx], axis=0)


        for key in ['sample', 'score', 'heads', 'states', 'action',
                    'obs', 'attentions', 'hidden',
                    'old_attend', 'coverage', 'source']:
            pipe.append_new(key, 0, hyper=True)

        # *** special care ***
        pipe.hyp_messages['secs'].append(pipe.new_hyp_messages['secs'][0])
        pipe.hyp_messages['mask'].append(pipe.new_hyp_messages['mask'][:, 0])

        # make it numpy array
        for key in ['heads', 'score', 'coverage', 'mask', 'states', 'hidden']:
            pipe.asarray(key, True)

        pipe.hyp_messages['mask'] = pipe.hyp_messages['mask'].T

        prev_z    = beamz
        prev_hid  = pipe.hyp_messages['hidden']
        mask      = pipe.hyp_messages['mask']

        prev_w    = numpy.array([w[-1] if len(w) > 0
                                 else -1 for w in beamwords],
                                dtype='int64')

    best_trans = numpy.argmin([v/ (len(w)) for (w, v) in zip(pipe.messages['sample'], pipe.messages['score'])])
    # best_trans = numpy.argmin([v for (w, v) in zip(pipe.messages['sample'], pipe.messages['score'])])

    # finalize the output
    pipe.hyp_messages['action'][0]
    pipe.messages['sample'] = [pipe.messages['sample'][best_trans]]
    pipe.messages['score']  = [pipe.messages['score'][best_trans]]
    pipe.messages['action'] = [pipe.messages['action'][best_trans]]
    # pipe.messages['secs']   = pipe.hyp_messages['secs']

    # =======================================================================
    # Collecting Rewards.
    # =======================================================================
    R     = []
    track = []

    Ref   = []
    Sys   = []

    sp, sc, act = [pipe.messages[key][0] for key in ['sample', 'score', 'action']]
    reference   = [_bpe2words(_seqs2words([trgs[0]], t_idict))[0].split()]
    y           = numpy.asarray(sp,  dtype='int64')[:, None]
    y_mask      = numpy.ones_like(y, dtype='float32')
    steps       = len(act)

    # turn back to sentence level
    words       = _seqs2words([sp], t_idict)[0]
    decoded     = _bpe2words([words])[0].split()

    Ref += [reference]
    Sys += [decoded]

    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # reward configs
    keys = {"steps": steps, "y":y, "y_mask": y_mask, "x_mask": x_mask,
            "act": act, "src_max": src_max, "ctx0": ctx0, "sidx": sidx,
            "f_cost": f_cost, "alpha": 0.5, "gamma": 1,
            "sample": decoded,
            "reference": reference,
            "words": words,
            "source_len": len(srcs[0])}

    # add additional configs
    for r in reward_config:
        keys[r] = reward_config[r]

    ret = return_reward(**keys)
    Rk, quality, delay, instant_reward = ret
    reward = numpy.mean(instant_reward) # the terminal reward

    if steps > max_steps:
        max_steps = steps

    # Rk  += sec_info[2] * 10000
    R     += [Rk]
    track += [(quality, delay, reward)]

    pipe.messages['R']     = R
    pipe.messages['track'] = track
    pipe.messages['Ref']   = Ref
    pipe.messages['Sys']   = Sys

    # --------------------------------------------------- #
    # collect information
    keywords     = ['sample', 'action', 'obs', 'forgotten', 'secs',
                    'attentions', 'old_attend', 'score', 'track', 'R', 'Ref', 'Sys']
    for k in keywords:
        if k not in Statistcs:
            Statistcs[k]  = pipe.messages[k]
        else:
            Statistcs[k] += pipe.messages[k]


    # If not train, End here
    if not train:
        return Statistcs


    # ================================================================================================= #
    # Policy Gradient over Trajectories
    # ================================================================================================= #
    # print Act_masks
    # p rint Actions

    p_obs, p_mask   \
            = _padding(Statistcs['obs'],
                       shape=(max_steps, n_samples * n_sentences, _policy.n_in),
                       return_mask=True, sidx=sidx)
    p_r     = _padding(Statistcs['R'],
                       shape=(max_steps, n_samples * n_sentences))
    p_act   = _padding(Statistcs['action'],
                       shape=(max_steps, n_samples * n_sentences), dtype='int64')

    # learning
    info    = _policy.get_learner()([p_obs, p_mask], p_act, p_r)

    # add the reward statistics
    q, d, r = zip(*Statistcs['track'])
    info['Quality']   = numpy.mean(q)
    info['Delay']     = numpy.mean(d)
    info['StartR']    = numpy.mean(r)

    _probs     /= float(_total)
    info['p(WAIT)']   = _probs[0]
    info['p(COMMIT)'] = _probs[1]

    if use_forget:
        info['F']   = _probs[2]


    return Statistcs, info, pipe_t


