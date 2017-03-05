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
                          src_eos=True):

    # --- unzip functions
    f_sim_ctx     = funcs[0]
    f_sim_init    = funcs[1]
    f_sim_next    = funcs[2]
    f_cost        = funcs[3]

    if reward_config['finetune']:
        ff_init   = funcs[4]
        ff_cost   = funcs[5]
        ff_update = funcs[6]

    n_sentences   = len(srcs)
    n_out         = 3 if use_forget else 2


    _probs        = numpy.zeros((n_out, ))
    _total        = 0

    live_k    = n_samples * n_sentences
    live_all  = live_k

    # ============================================================================ #
    # Initialization Before Generating Trajectories
    # ============================================================================ #

    # Critical! add the <eos> ------------------
    srcs = [src + [0] for src in srcs]

    src_max   = max([len(src) for src in srcs])
    if src_max < sidx:
        sidx  = src_max

    x, ctx0, z0, seq_info0 = [], [], [], []

    # data initialization
    for id, (src, trg) in enumerate(zip(srcs, trgs)):

        _x          = numpy.array(src, dtype='int64')[:, None]
        _, _ctx0, _ = f_sim_ctx(_x)
        _z0         = f_sim_init(_ctx0[:sidx, :])

        x.append(_x[:, 0])
        ctx0.append(_ctx0[:, 0, :])
        z0.append(_z0.flatten())
        seq_info0.append([id, len(src), 0])  # word id / source length / correctness

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
    z         = numpy.tile(z0,   [n_samples, 1])
    ctx       = numpy.tile(ctx,  [1, n_samples, 1])
    x         = numpy.tile(x,    [1, n_samples])
    hidden    = numpy.tile(hidden0, [live_k, 1])

    seq_info  = []
    for _ in range(live_k / n_sentences):
        seq_info += copy.deepcopy(seq_info0)

    # =========================================================================== #
    # PIPE for message passing
    # =========================================================================== #
    pipe   = OrderedDict()
    h_pipe = OrderedDict()

    # initialize pipes
    for key in ['sample', 'score', 'action',
                'obs', 'attentions', 'old_attend',
                'coverage', 'forgotten',
                'seq_info',
                'cmask', 'source', 'i_mask']:
        pipe[key] = []

    # initialize h-pipe
    for key in ['sample', 'obs', 'attentions',
                'hidden', 'old_attend', 'cmask']:
        h_pipe[key] = [[] for _ in range(live_k)]

    h_pipe['score']     = numpy.zeros(live_k).astype('float32')
    h_pipe['action']    = [[0]  * sidx for _ in xrange(live_k)]
    h_pipe['forgotten'] = [[-1] * sidx for _ in xrange(live_k)]
    h_pipe['coverage']  = numpy.zeros((live_k, ctx.shape[0])).astype('float32')

    h_pipe['mask']      = mask
    h_pipe['ctx']       = ctx    # contexts
    h_pipe['source']    = x      # source words
    h_pipe['seq_info']  = seq_info
    h_pipe['heads']     = numpy.asarray([[sidx, 0, 0]] * live_k)  # W C F
    h_pipe['i_mask']    = mask

    h_pipe['prev_w']    = -1 * numpy.ones((live_k, )).astype('int64')
    h_pipe['prev_z']    = z
    h_pipe['prev_hid']  = hidden

    # these are inputs that needs to be updated
    step       = 0

    #
    # =======================================================================
    # ROLLOUT: Iteration until all the samples over.
    # Action space:
    # 0: Read,
    # 1: Commit,
    # 2: Forget (optional)
    # =======================================================================
    while live_k > 0:
        step += 1

        # ------------------------------------------------------------------
        # Run one-step translation
        # ------------------------------------------------------------------

        inps  = [h_pipe[v] for v in ['prev_w', 'ctx', 'mask', 'prev_z']]
        next_p, _, next_z, next_o, next_a, cur_emb = f_sim_next(*inps)

        if full_attention:
            old_mask = numpy.tile(one,  [1, live_k])
            inps2    = inps
            inps2[2] = old_mask
            _, _, _, _, next_fa, _ = f_sim_next(*inps2)

        # obtain the candidate and the accumulated score.
        _cand          = next_p.argmax(axis=-1)  # live_k, candidate words
        _score         = next_p[range(live_k), _cand]

        # new place-holders for temporal results: new-hyp-message
        n_pipe = OrderedDict()

        for key in ['sample', 'score', 'heads', 'attentions',
                    'old_attend', 'coverage', 'mask', 'ctx',
                    'seq_info', 'cmask', 'obs',
                    'prev_z', 'source', 'i_mask',
                    'action', 'forgotten']:

            n_pipe[key] = copy.copy(h_pipe[key])
        n_pipe['hidden'] = []

        cov    = n_pipe['coverage'] * n_pipe['mask'].T + next_a  # clean that has been forgotten
        cid    = cov.argmax(axis=-1)

        # ------------------------------------------------------------------
        # Run one-step agent action.
        # ------------------------------------------------------------------
        _actions, _aprop, _hidden, _z = _policy.action(next_o, h_pipe['prev_hid'])  # input the current observation
        if reward_config['greedy']:
            _actions = _aprop.argmax(-1)

        _total += _aprop.shape[0]
        _probs += _aprop.sum(axis=0)

        # ------------------------------------------------------------------
        # Evaluate the action
        # ------------------------------------------------------------------
        for idx, wi in enumerate(_cand):

            # action
            a      = _actions[idx]
            c_mask = n_pipe['mask'][:, idx]

            if reward_config.get('upper', False):
                a = 0  # testing upper bound: only wait

            if reward_config['greedy'] and \
                    (n_pipe['heads'][idx, 0] >= n_pipe['seq_info'][idx][1]):
                a = 1  # in greedy mode. must end.

            if reward_config['greedy'] and \
                    (n_pipe['heads'][idx, 2] >= n_pipe['heads'][idx, 0]):
                a = 1  # in greedy mode. must end.

            # must read the whole sentence
            # pass

            # message appending
            n_pipe['obs'][idx].append(next_o[idx])
            n_pipe['action'][idx].append(a)
            n_pipe['hidden'].append(_hidden[idx])

            # change the behavior of NMT model
            if a == 0:
                # read-head move on one step
                if n_pipe['heads'][idx, 0] < n_pipe['seq_info'][idx][1]:
                    n_pipe['mask'][n_pipe['heads'][idx, 0], idx] = 1
                    n_pipe['heads'][idx, 0] += 1

                n_pipe['forgotten'][idx].append(-1)

                # if the first word is still waiting for decoding
                if numpy.sum(n_pipe['action'][idx]) == 0:
                    temp_sidx = n_pipe['heads'][idx, 0]
                    _ctx0     = ctx0[n_pipe['seq_info'][idx][0]][:, None, :]
                    _z0       = f_sim_init(_ctx0[:temp_sidx])  # initializer
                    n_pipe['prev_z'][idx] = _z0
                    n_pipe['i_mask'][temp_sidx, idx] = 1

            # for write:
            elif a == 1:
                n_pipe['sample'][idx].append(wi)
                n_pipe['cmask'][idx].append(c_mask)
                n_pipe['score'][idx] += _score[idx]
                n_pipe['attentions'][idx].append(next_a[idx])
                n_pipe['forgotten'][idx].append(-1)

                if full_attention:
                    n_pipe['old_attend'][idx].append(next_fa[idx])

                n_pipe['prev_z'][idx]    = next_z[idx]  # update the decoder's hidden states
                n_pipe['heads'][idx, 1] += 1
                n_pipe['coverage'][idx]  = cov[idx]

            # for forget:
            elif a == 2:
                if forget_left:
                    _idx = n_pipe['heads'][idx, 2]
                    if n_pipe['heads'][idx, 2] < n_pipe['heads'][idx, 0]:
                        n_pipe['mask'][_idx, idx] = 0
                        n_pipe['heads'][idx, 2] += 1
                    n_pipe['forgotten'][idx].append(_idx)
                else:
                    n_pipe['mask'][cid[idx], idx] = 0
                    n_pipe['heads'][idx, 2] = cid[idx]
                    n_pipe['forgotten'][idx].append(cid[idx])
            else:
                raise NotImplementedError

        # ------------------------------------------------------------------
        # Check the correctness!
        # ------------------------------------------------------------------
        for idx in xrange(live_k):
            if n_pipe['heads'][idx, 0] >= n_pipe['seq_info'][idx][1]:
                # the read head already reached the end.
                n_pipe['seq_info'][idx][2] = -1

        # ------------------------------------------------------------------
        # Collect the trajectories
        # ------------------------------------------------------------------
        #  kill the completed samples, so I need to build new hyp-messages
        h_pipe = OrderedDict()

        for key in ['sample', 'score', 'heads', 'mask',
                    'prev_z', 'coverage', 'forgotten',
                    'action', 'obs', 'ctx', 'seq_info',
                    'attentions', 'hidden', 'old_attend',
                    'cmask', 'source', 'i_mask']:
            h_pipe[key] = []

        for idx in xrange(len(n_pipe['sample'])):

            if (len(n_pipe['sample'][idx]) > 0) and \
                  ((n_pipe['sample'][idx][-1] == 0)         # translate over
                   or (n_pipe['heads'][idx][1] >= maxlen)   # exceed the maximum length
                   or (step > (1.5 * maxlen))):

                for key in ['sample', 'score', 'action', 'obs',
                            'attentions', 'old_attend', 'coverage',
                            'forgotten', 'cmask', 'seq_info']:
                    pipe[key].append(n_pipe[key][idx])

                pipe['i_mask'].append(n_pipe['i_mask'][:, idx])
                pipe['source'].append(n_pipe['source'][:, idx])
                live_k -= 1

            else:

                for key in ['sample', 'score', 'heads',
                            'prev_z', 'action',
                            'obs', 'attentions', 'hidden',
                            'old_attend', 'coverage',
                            'forgotten', 'cmask', 'seq_info']:
                    h_pipe[key].append(n_pipe[key][idx])

                h_pipe['mask'].append(n_pipe['mask'][:, idx])
                h_pipe['ctx'].append(n_pipe['ctx'][:, idx])
                h_pipe['i_mask'].append(n_pipe['i_mask'][:, idx])
                h_pipe['source'].append(n_pipe['source'][:, idx])

        # make it numpy array
        for key in ['heads', 'score', 'coverage',
                    'mask', 'ctx', 'prev_z', 'hidden',
                    'source', 'i_mask']:
            h_pipe[key] = numpy.asarray(h_pipe[key])
        h_pipe['mask']   = h_pipe['mask'].T
        h_pipe['source'] = h_pipe['source'].T
        h_pipe['i_mask'] = h_pipe['i_mask'].T

        if h_pipe['ctx'].ndim == 3:
            h_pipe['ctx']    = h_pipe['ctx'].transpose(1, 0, 2)

        elif h_pipe['ctx'].ndim == 2:
            h_pipe['ctx']  = h_pipe['ctx'][:, None, :]

        h_pipe['prev_hid'] = h_pipe['hidden']
        h_pipe['prev_w']   = numpy.array([w[-1] if len(w) > 0
                                          else -1 for w in h_pipe['sample']], dtype='int64')

    # =======================================================================
    # Collecting Rewards.
    # =======================================================================
    R     = []
    track = []

    Ref   = []
    Sys   = []

    Words = []
    SWord = []

    max_steps   = -1
    max_w_steps = -1

    for k in xrange(live_all):
        sp, sc, act, sec_info = [pipe[key][k] for key in ['sample', 'score', 'action', 'seq_info']]
        reference   = [_bpe2words(_seqs2words([trgs[sec_info[0]]], t_idict))[0].split()]
        y           = numpy.asarray(sp,  dtype='int64')[:, None]
        y_mask      = numpy.ones_like(y, dtype='float32')

        steps       = len(act)
        w_steps     = len(sp)

        # turn back to sentence level
        words       = _seqs2words([sp], t_idict)[0]
        decoded     = _bpe2words([words])[0].split()

        Ref        += [reference]
        Sys        += [decoded]
        Words      += [words]
        SWord      += [srcs[sec_info[0]]]

        # ----------------------------------------------------------------
        # reward configs
        # ----------------------------------------------------------------
        keys = {"steps": steps,
                "y": y, "y_mask": y_mask,
                "x_mask": x_mask,
                "act": act, "src_max": src_max,
                "ctx0": ctx0, "sidx": sidx,
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
        reward = numpy.mean(instant_reward)  # the terminal reward

        if steps > max_steps:
            max_steps = steps

        if w_steps > max_w_steps:
            max_w_steps = w_steps

        R     += [Rk]
        track += [(quality, delay, reward)]

    pipe['R']     = R
    pipe['track'] = track
    pipe['Ref']   = Ref
    pipe['Sys']   = Sys

    pipe['Words'] = Words
    pipe['SWord'] = SWord

    # If not train, End here
    if not train:
        return pipe

    # ================================================================= #
    # Policy Gradient over Trajectories for the Agent
    # ================================================================= #
    p_obs, p_mask = _padding(pipe['obs'],
                             shape=(max_steps, n_samples * n_sentences, _policy.n_in),
                             return_mask=True, sidx=sidx)
    p_r           = _padding(pipe['R'],
                             shape=(max_steps, n_samples * n_sentences))
    p_act         = _padding(pipe['action'],
                             shape=(max_steps, n_samples * n_sentences), dtype='int64')

    # learning
    info    = _policy.get_learner()([p_obs, p_mask], p_act, p_r)

    # ================================================================ #
    # Policy Gradient for the underlying NMT model
    # ================================================================ #

    if reward_config['finetune']:
        p_y, p_y_mask = _padding(pipe['sample'],
                                 shape=(max_w_steps, n_samples * n_sentences),
                                 return_mask=True, dtype='int64')
        p_x = numpy.asarray(pipe['source']).T
        p_i_mask = numpy.asarray(pipe['i_mask']).T
        p_c_mask = _padding(pipe['cmask'],
                            shape=(max_w_steps, n_samples * n_sentences, p_x.shape[0]))
        p_adv = info['advantages']
        new_adv = [p_adv[p_act[:, s] == 1, s] for s in range(p_adv.shape[1])]
        new_adv = _padding(new_adv, shape=(max_w_steps, n_samples * n_sentences))

        a_cost, _ = ff_cost(p_x, p_i_mask, p_y, p_y_mask, p_c_mask.transpose(0, 2, 1), new_adv)
        ff_update(2e-5)

        info['a_cost'] = a_cost

    # add the reward statistics
    q, d, r = zip(*pipe['track'])
    info['Quality']   = numpy.mean(q)
    info['Delay']     = numpy.mean(d)
    info['StartR']    = numpy.mean(r)

    _probs     /= float(_total)
    info['p(WAIT)']   = _probs[0]
    info['p(COMMIT)'] = _probs[1]

    if use_forget:
        info['F']   = _probs[2]

    return pipe, info


