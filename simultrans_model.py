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
def simultaneous_decoding(funcs,
                          _policy,
                          srcs,     # source sentences
                          trgs,     # taeget sentences
                          t_idict = None,
                          step=1, peek=1, sidx=3,
                          n_samples=10,
                          maxlen=120,
                          reward_config=None,
                          train=False,
                          use_forget=False,
                          forget_left=True,
                          use_newinput=False,
                          full_attention=False,
                          use_coverage=False,
                          on_groundtruth=0,
                          src_eos=True):

    # unzip functions
    f_sim_ctx     = funcs[0]
    f_sim_init    = funcs[1]
    f_sim_next    = funcs[2]
    f_cost        = funcs[3]

    if reward_config['finetune']:
        ff_init   = funcs[4]
        ff_cost   = funcs[5]
        ff_update = funcs[6]


    Statistcs     = OrderedDict()

    n_sentences   = len(srcs)
    n_out         = 3 if use_forget else 2
    max_steps     = -1

    _probs        = numpy.zeros((n_out, ))
    _total        = 0

    # check
    # if reward_config['greedy']:
    #     print 'use greedy policy'


    # ============================================================================ #
    # Generating Trajectories based on Current Policy
    # ============================================================================ #

    live_k    = (n_samples + on_groundtruth) * n_sentences
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
        # _z0   = f_sim_init(_ctx0)

        # print 'state', init
        # print 'state', _z0
        # print 'ctx0', _ctx0, _ctx0.shape
        # print 'ctx_mean', m

        x.append(_x[:, 0])
        ctx0.append(_ctx0[:, 0, :])
        z0.append(_z0.flatten())
        secs0.append([id, len(src), 0])  # word id / source length / correctness


    # pad the results
    x, x_mask = _padding(x, (src_max, n_sentences), dtype='int64', return_mask=True)
    ctx       = _padding(ctx0, (src_max, n_sentences, ctx0[0].shape[-1]))
    z0        = numpy.asarray(z0)

    mask      = numpy.asarray([1.] * sidx + [0.] * (src_max - sidx), dtype='float32')[:, None]
    one       = numpy.asarray([1.] * src_max, dtype='float32')[:, None]

    # hidden states
    hidden0   = _policy.init_hidden()

    # if we have multiple samples for one input sentence
    mask      = numpy.tile(mask, [1, live_k])
    z0        = numpy.tile(z0,   [live_k / n_sentences, 1])
    ctx       = numpy.tile(ctx,  [1, live_k / n_sentences, 1])
    hidden0   = numpy.tile(hidden0, [live_k, 1])

    secs      = []
    for _ in xrange(live_k / n_sentences):
        secs += copy.deepcopy(secs0)


    #============================================================================ #
    # PIPE for message passing
    # =========================================================================== #
    pipe      = PIPE(['sample', 'score', 'action', 'obs', 'attentions',
                      'old_attend', 'coverage', 'source', 'forgotten', 'secs', 'cmask'])

    # Build for the temporal results: hyp-message
    for key in ['sample', 'obs', 'attentions', 'hidden', 'old_attend', 'cmask']:
        pipe.init_hyp(key, live_k)

    # special care
    pipe.hyp_messages['source']    = [-1 for _ in xrange(n_samples)] + [0 for _ in xrange(on_groundtruth)]
    pipe.hyp_messages['source']    = [si for si in pipe.hyp_messages['source'] for _ in xrange(n_sentences)]

    pipe.hyp_messages['score']     = numpy.zeros(live_k).astype('float32')
    pipe.hyp_messages['action']    = [[0] * sidx for _ in xrange(live_k)]
    pipe.hyp_messages['forgotten'] = [[-1] * sidx for _ in xrange(live_k)]
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
    while live_k > 0:
        step += 1

        inps         = [prev_w, ctx, mask, prev_z]
        # print mask
        next_p, _, next_z, next_o, next_a, cur_emb = f_sim_next(*inps)

        if full_attention:
            old_mask = numpy.tile(one,  [1, live_k])
            inps2    = inps
            inps2[2] = old_mask
            _, _, _, _, next_fa, _ = f_sim_next(*inps2)

        # obtain the candidate and the accumulated score.
        _cand          = next_p.argmax(axis=-1)  # live_k
        _score         = next_p[range(live_k), _cand]

        # new place-holders for temporal results: new-hyp-message
        pipe.clean_new_hyp()


        for key in ['sample', 'score', 'heads', 'attentions', 'old_attend', 'coverage', 'source',
                    'mask', 'ctx', 'secs', 'states', 'cmask']:
            pipe.init_new_hyp(key, use_copy=True)

        for key in ['action', 'forgotten', 'obs', 'hidden']:
            pipe.init_new_hyp(key, use_copy=False)

        cov    = pipe.new_hyp_messages['coverage'] * pipe.new_hyp_messages['mask'].T \
               + next_a  # clean that has been forgotten

        # current maximum
        cid    = cov.argmax(axis=-1)


        # Rollout the action.
        _actions, _aprop, _hidden, _z = _policy.action(next_o, prev_hid)  # input the current observation

        # print _actions.shape
        if reward_config['greedy']:
            _actions = _aprop.argmax(-1)
            # print _actions.shape

        _total += _aprop.shape[0]
        _probs += _aprop.sum(axis=0)

        # check each candidate
        for idx, wi in enumerate(_cand):

            # collect the action
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

            # must read the whole sentence
            #if pipe.new_hyp_messages['heads'][idx, 0] < pipe.new_hyp_messages['secs'][idx][1]:
            #    if wi == 0: # end before read the last source words --->  wait!!
            #        a = 0


            # message appending
            pipe.append('obs',       next_o[idx],   idx=idx, use_hyp=True)
            pipe.append('action',    a,             idx=idx, use_hyp=True)   # collect action.
            pipe.append('hidden',    _hidden[idx])

            # print pipe.hyp_messages['heads'][idx]
            if a == 0:
                # read-head move on one step
                # print 'p', pipe.hyp_messages['heads'][idx, 0], pipe.hyp_messages['secs'][idx]

                if pipe.new_hyp_messages['heads'][idx, 0] < pipe.new_hyp_messages['secs'][idx][1]:
                    pipe.new_hyp_messages['mask'][pipe.new_hyp_messages['heads'][idx, 0], idx] = 1
                    pipe.new_hyp_messages['heads'][idx, 0] += 1

                pipe.append('forgotten', -1, idx=idx, use_hyp=True)

                # if the first word is still waiting for decoding
                # """
                if numpy.sum(pipe.new_hyp_messages['action'][idx]) == 0:
                    temp_sidx = pipe.new_hyp_messages['heads'][idx, 0]
                    _ctx0     = ctx0[pipe.new_hyp_messages['secs'][idx][0]][:, None, :]
                    _z0       = f_sim_init(_ctx0[:temp_sidx])  # initializer
                    pipe.new_hyp_messages['states'][idx] = _z0
                # """

            # for commit:
            elif a == 1:
                # print mask
                # update new_hyp_message
                head_t = pipe.new_hyp_messages['source'][idx]
                if head_t == -1:  # use generated samples
                    pipe.add('sample', [wi], idx)
                    pipe.add('cmask',[mask], idx)

                else:
                    pipe.add('sample', [trg[head_t] if head_t < len(trg) else 0], idx)  # use ground-truth
                    pipe.new_hyp_messages['source'][idx] += 1

                pipe.add('score',        _score[idx], idx)
                pipe.add('attentions', [next_a[idx]], idx)
                pipe.append('forgotten', -1, idx=idx, use_hyp=True)

                if full_attention:
                    pipe.add('old_attend', [next_fa[idx]], idx)

                # *** special care
                pipe.new_hyp_messages['states'][idx]    = next_z[idx]
                pipe.new_hyp_messages['heads'][idx, 1] += 1
                pipe.new_hyp_messages['coverage'][idx]  = cov[idx]


            # for forget:
            elif a == 2:
                # move the forget head.
                if forget_left:
                    _idx = pipe.new_hyp_messages['heads'][idx, 2]
                    if pipe.new_hyp_messages['heads'][idx, 2] < pipe.new_hyp_messages['heads'][idx, 0]:
                        pipe.new_hyp_messages['mask'][_idx, idx] = 0
                        pipe.new_hyp_messages['heads'][idx, 2] += 1
                    pipe.append('forgotten', _idx, idx=idx, use_hyp=True)
                else:
                    pipe.new_hyp_messages['mask'][cid[idx], idx] = 0
                    pipe.new_hyp_messages['heads'][idx, 2] = cid[idx]
                    pipe.append('forgotten', cid[idx], idx=idx, use_hyp=True)

            else:
                raise NotImplementedError


        # check the correctness, or given a very negative reward
        # print pipe.new_hyp_messages['heads'][:, 0], pipe.new_hyp_messages['secs']
        for idx in xrange(live_k):
            if pipe.new_hyp_messages['heads'][idx, 0] >= pipe.new_hyp_messages['secs'][idx][1]:  # the read head already reached the end.
                pipe.new_hyp_messages['secs'][idx][2] = -1


        #  kill the completed samples, so I need to build new hyp-messages
        pipe.clean_hyp()

        for key in ['sample', 'score', 'heads', 'mask',
                    'states', 'coverage', 'forgotten',
                    'action', 'obs', 'ctx', 'secs',
                    'attentions', 'hidden', 'old_attend', 'source', 'cmask']:
            pipe.init_hyp(key)

        for idx in xrange(len(pipe.new_hyp_messages['sample'])):


            if (len(pipe.new_hyp_messages['sample'][idx]) > 0) and \
                  ((pipe.new_hyp_messages['sample'][idx][-1] == 0)         # translate over
                   or (pipe.new_hyp_messages['heads'][idx][1] >= maxlen)   # exceed the maximum length
                   or (step > (1.5 * maxlen))):
                   # or (pipe.new_hyp_messages['secs'][idx][2]==-1)):        # get into something wrong

                for key in ['sample', 'score', 'action', 'obs', 'attentions',
                            'old_attend', 'coverage', 'source', 'forgotten', 'cmask']:
                    pipe.append_new(key, idx, hyper=False)

                pipe.messages['secs'].append(pipe.new_hyp_messages['secs'][idx])

                live_k -= 1

            else:

                for key in ['sample', 'score', 'heads', 'states', 'action',
                            'obs', 'attentions', 'hidden',
                            'old_attend', 'coverage', 'source', 'forgotten', 'cmask']:
                    pipe.append_new(key, idx, hyper=True)

                # *** special care ***
                pipe.hyp_messages['secs'].append(pipe.new_hyp_messages['secs'][idx])
                pipe.hyp_messages['mask'].append(pipe.new_hyp_messages['mask'][:, idx])
                pipe.hyp_messages['ctx'].append(pipe.new_hyp_messages['ctx'][:, idx])

        # make it numpy array
        for key in ['heads', 'score', 'coverage', 'mask', 'ctx', 'states', 'hidden']:
            pipe.asarray(key, True)

        pipe.hyp_messages['mask'] = pipe.hyp_messages['mask'].T

        if pipe.hyp_messages['ctx'].ndim == 3:
            pipe.hyp_messages['ctx']  = pipe.hyp_messages['ctx'].transpose(1, 0, 2)
        elif pipe.hyp_messages['ctx'].ndim == 2:
            pipe.hyp_messages['ctx']  = pipe.hyp_messages['ctx'][:, None, :]

        prev_z    = pipe.hyp_messages['states']
        prev_hid  = pipe.hyp_messages['hidden']
        mask      = pipe.hyp_messages['mask']
        ctx       = pipe.hyp_messages['ctx']

        prev_w    = numpy.array([w[-1] if len(w) > 0
                                 else -1 for w in pipe.hyp_messages['sample']],
                                dtype='int64')


    # =======================================================================
    # Collecting Rewards.
    # =======================================================================
    R     = []
    track = []

    Ref   = []
    Sys   = []

    Words = []
    SWord = []

    for k in xrange(live_all):
        sp, sc, act, sec_info = [pipe.messages[key][k] for key in ['sample', 'score', 'action', 'secs']]
        reference   = [_bpe2words(_seqs2words([trgs[sec_info[0]]], t_idict))[0].split()]
        y           = numpy.asarray(sp,  dtype='int64')[:, None]
        y_mask      = numpy.ones_like(y, dtype='float32')
        steps       = len(act)


        # turn back to sentence level
        words       = _seqs2words([sp], t_idict)[0]
        decoded     = _bpe2words([words])[0].split()

        Ref        += [reference]
        Sys        += [decoded]
        Words      += [words]
        SWord      += [srcs[sec_info[0]]]

        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
        # reward configs
        keys = {"steps": steps, "y":y, "y_mask": y_mask, "x_mask": x_mask,
                "act": act, "src_max": src_max, "ctx0": ctx0, "sidx": sidx,
                "f_cost": f_cost, "alpha": 0.5,
                "sample": decoded,
                "reference": reference,
                "words": words,
                "source_len": sec_info[1]}

        # add additional configs
        for r in reward_config:
            keys[r] = reward_config[r]

        ret = return_reward(**keys)
        Rk, quality, delay, instant_reward = ret
        reward = numpy.mean(instant_reward) # the terminal reward

        if steps > max_steps:
            max_steps = steps

        R     += [Rk]
        track += [(quality, delay, reward)]

    pipe.messages['R']     = R
    pipe.messages['track'] = track
    pipe.messages['Ref']   = Ref
    pipe.messages['Sys']   = Sys

    # --------------------------------------------------- #
    # add to global lists.
    pipe_t       = pipe.split()

    # --------------------------------------------------- #
    # collect information
    keywords     = ['sample', 'action', 'obs', 'forgotten', 'secs',
                    'attentions', 'old_attend', 'score', 'track',
                    'R', 'Ref', 'Sys', 'cmask']
    for k in keywords:
        if k not in Statistcs:
            Statistcs[k]  = pipe.messages[k]
        else:
            Statistcs[k] += pipe.messages[k]

    Statistcs['Words'] = Words
    Statistcs['SWord'] = SWord

    # If not train, End here
    if not train:
        return Statistcs


    print len(Statistcs['cmask'])
    print len(Statistcs['cmask'][0])
    print Statistcs['cmask'][0][0].shape
    sys.exit(1)
    # ================================================================= #
    # Policy Gradient over Trajectories
    # ================================================================= #
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


    # ================================================================ #
    # Policy Gradient for the underlying NMT model
    if reward_config['finetune']:
        pass



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


